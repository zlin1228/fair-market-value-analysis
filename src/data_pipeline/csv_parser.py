from io import BytesIO
import os
import pyarrow as pa
import pyarrow.csv as csv
from typing import Any, Callable, Dict, List, Literal, Union
import traceback
from performance_helper import create_profiler

_invalid_row_handler = lambda _: 'error'


def _parse_csv_file(file: Union[str,BytesIO], column_types: Dict[str,pa.DataType], delimiter: str=',',
                quote_char: Union[str,Literal[False]]='"',
                invalid_row_handler:Union[None,Callable[[Any],Union[Literal['skip'],Literal['error']]]]=None) -> pa.table:
    return csv.read_csv(file,
        parse_options=csv.ParseOptions(
            delimiter=delimiter,
            invalid_row_handler=invalid_row_handler or _invalid_row_handler,
            quote_char=quote_char
        ),
        convert_options=csv.ConvertOptions(
            column_types=column_types,
            include_columns=[k for k in column_types.keys()],
            strings_can_be_null=True
        ))

def parse_csv_file(file: Union[str,BytesIO], column_types: Dict[str,pa.DataType],
                delimiter: str=',', quote_char: Union[str,Literal[False]]='"',
                process:Union[None,Callable[[pa.table],pa.table]]=None,
                invalid_row_handler:Union[None,Callable[[Any],Union[Literal['skip'],Literal['error']]]]=None) -> pa.table:
    """Parses either the csv file at the path provided or the csv contained in the bytes buffer provided and returns a pyarrow table containing the data

    Keyword arguments:
    file -- path to csv file or bytes buffer
    column_types -- dict mapping any columns needed for processing to their associated pyarrow type
    delimiter -- optional, defaults to ',', csv delimiter character
    quote_char -- optional, defaults to '"', character to use for quotes or False if quoted data is not allowed
    process -- optional, defaults to None, function that receives a pyarrow table representing a parsed csv file and returns a new pyarrow table based on the parsed table
    invalid_row_handler -- optional, defaults to a row handler that throws on any invalid row, function that receives an InvalidRow (https://arrow.apache.org/docs/python/generated/pyarrow.csv.InvalidRow.html#pyarrow-csv-invalidrow) and returns 'skip' or 'error'
    """
    table = _parse_csv_file(file, column_types, delimiter, quote_char, invalid_row_handler)
    table = process(table) if process else table
    return table

def parse_csv_files(file_paths: List[str], column_types: Dict[str,pa.DataType],
                delimiter: str=',', quote_char: Union[str,Literal[False]]='"',
                process:Union[None,Callable[[pa.table],pa.table]]=None,
                invalid_row_handler:Union[None,Callable[[Any],Union[Literal['skip'],Literal['error']]]]=None) -> pa.table:
    profile = create_profiler('parse_csv_files')
    """Parses the list of csv files and returns a pyarrow table containing the concatenated data

    Keyword arguments:
    file_paths -- list of csv file paths
    column_types -- dict mapping any columns needed for processing to their associated pyarrow type
    delimiter -- optional, defaults to ',', csv delimiter character
    quote_char -- optional, defaults to '"', character to use for quotes or False if quoted data is not allowed
    process -- optional, defaults to None, function that receives a pyarrow table representing a parsed csv file and returns a new pyarrow table based on the parsed table
    invalid_row_handler -- optional, defaults to a row handler that throws on any invalid row, function that receives an InvalidRow (https://arrow.apache.org/docs/python/generated/pyarrow.csv.InvalidRow.html#pyarrow-csv-invalidrow) and returns 'skip' or 'error'
    """
    with profile() as p:
        tables = []
        for i, file_path in enumerate(file_paths):
            try:
                tables.append(_parse_csv_file(file_path, column_types, delimiter, quote_char, invalid_row_handler))
            except Exception as e:
                print(f'EXCEPTION WHILE PROCESSING {file_path}')
                print(e)
                traceback.print_exc()
                raise e
            p.print(f'{i + 1}/{len(file_paths)} - {os.path.basename(file_path)}')
        p.print(f'avg: { round(p.elapsed_time() / len(file_paths), 3) }s')
        with profile('concat_tables'):
            table = pa.concat_tables(tables)
        with profile('process'):
            table = process(table) if process else table
        return table

def parse_csv(csv_bytes: bytes, column_types: Dict[str,pa.DataType],
                delimiter: str=',', quote_char: Union[str,Literal[False]]='"',
                process:Union[None,Callable[[pa.table],pa.table]]=None,
                invalid_row_handler:Union[None,Callable[[Any],Union[Literal['skip'],Literal['error']]]]=None) -> pa.table:
    """Parses the csv contained in the provided bytes and returns a pyarrow table containing the data

    example: parse_csv(b'a,b,c\n1,2,3', ...)

    Keyword arguments:
    file_paths -- list of csv file paths
    column_types -- dict mapping any columns needed for processing to their associated pyarrow type
    delimiter -- optional, defaults to ',', csv delimiter character
    quote_char -- optional, defaults to '"', character to use for quotes or False if quoted data is not allowed
    process -- optional, defaults to None, function that receives a pyarrow table representing a parsed csv file and returns a new pyarrow table based on the parsed table
    invalid_row_handler -- optional, defaults to a row handler that throws on any invalid row, function that receives an InvalidRow (https://arrow.apache.org/docs/python/generated/pyarrow.csv.InvalidRow.html#pyarrow-csv-invalidrow) and returns 'skip' or 'error'
    """
    return parse_csv_file(BytesIO(csv_bytes), column_types, delimiter, quote_char, process, invalid_row_handler)
