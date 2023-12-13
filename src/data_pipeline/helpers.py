import math
import re

import pyarrow as pa
import pyarrow.compute as pc

from pyarrow import compute as pc

ordinal_re = re.compile(r'^.*/expanded/ordinals/([^/.]+)\.parquet$')
table_re = re.compile(r'^.*/expanded/([^/.]+)\.parquet$')

def create_update_status(total):
    print(f'0% complete (0 / {total})')
    previous = 0
    def update_status(current):
        nonlocal previous
        new_percent = (current + 1) * 100 / total
        if new_percent > previous + 1 or (current + 1) == total:
            print(f'{math.floor(new_percent)}% complete ({current + 1} / {total})')
            previous = new_percent
    return update_status

def _map_ordinal(column: pa.array, ordinals: dict[str,pa.array], ordinal: str) -> pa.array:
    unique = pc.unique(column).drop_null()
    mask = pc.if_else(pc.is_in(unique, ordinals[ordinal]), False, True)
    ordinals[ordinal] = pa.concat_arrays([ordinals[ordinal], pc.filter(unique, mask)])
    return pc.index_in(column, ordinals[ordinal])


def map_ordinals(table: pa.table, ordinals: dict[str,pa.array]) -> pa.table:
    return pa.table({
                    **{ordinal: _map_ordinal(table.column(ordinal), ordinals, ordinal) for ordinal in ordinals.keys() if ordinal in table.column_names},
                    **{ non_ordinal: table.column(non_ordinal) for non_ordinal in table.column_names if non_ordinal not in ordinals.keys() }})

def get_file_paths_of_cached_files(mapping):
    '''Accepts the mapping returned by s3.get_all_with_prefix and returns a list of the paths to the extracted files in the s3 cache'''
    file_paths: list[str] = [path for
                             _, v in mapping.items()
                             for path in (v if isinstance(v, list) else (v,))]
    return file_paths

def get_sort_columns(fixed_priority_columns: list[str], all_columns: list[str]):
    '''Helper function to create deterministic list of sort columns.  Prioritizes fixed_priority_columns, then appends the remaining columns in sorted order'''
    sorted_columns = [*all_columns]
    sorted_columns.sort()
    return [*fixed_priority_columns,
            *[c for c in sorted_columns if c not in fixed_priority_columns]]
