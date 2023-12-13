import settings
settings.override_if_main(__name__, 1)

import datetime
import json
import tempfile
import time
from urllib import parse

import pyarrow as pa
import pyarrow.compute as pc

from data_pipeline.helpers import get_sort_columns
from load_data import get_ordinals, get_table, parse_coupon_string
from performance_helper import create_profiler
from pyarrow_helpers import filter_by_value_in_column
import s3

def _update_bond_index():
    profile = create_profiler('update_bond_index')
    with profile():
        with profile('create_bonds'):
            bonds = get_table('bonds')
            # filter out bonds that don't exist in trades
            bonds = filter_by_value_in_column(bonds, 'cusip', get_table('trades'), ['cusip'])
            # filter out bonds that don't have a rating
            bonds = filter_by_value_in_column(bonds, 'cusip', get_table('ratings'), ['cusip'])
            # apply ordinals
            ordinals = get_ordinals()
            bonds = pa.table({
                        **{ ordinal: ordinals[ordinal].take(bonds[ordinal]) for ordinal in ordinals.keys() if ordinal in bonds.column_names },
                        **{ non_ordinal: bonds[non_ordinal] for non_ordinal in bonds.column_names if non_ordinal not in ordinals.keys() }})
            # extract ticker
            # TODO: improve this to include more bloomberg ticker formats
            ticker_symbol = pc.extract_regex(bonds['ticker'], r'^(?P<ticker_symbol>.*?)(?: \d+\.?\d*| (?:F|PERP))? \d\d/\d\d/\d\d')
            ticker_symbol = pc.struct_field(ticker_symbol, [0])
            bonds = bonds.append_column('ticker_symbol', ticker_symbol)
            # parse coupon_string
            parsed_coupon_string = parse_coupon_string(bonds['coupon_string'])
            bonds = bonds.append_column('parsed_coupon_string', parsed_coupon_string)
            # filter out bonds with a coupon_string we can't parse
            bonds = bonds.filter(pc.true_unless_null(bonds['parsed_coupon_string']))
            # filter out bonds with null in any of these columns
            for column in ['figi','cusip','issuer','issue_date','outstanding','maturity','coupon_string','sector','industry']:
                bonds = bonds.filter(pc.true_unless_null(bonds[column]))
            # filter to only keep bonds with outstanding > 0
            bonds = bonds.filter(pc.greater(bonds['outstanding'], 0))
            # sort bonds
            sort_columns = get_sort_columns([   'ticker',
                                                'maturity'], bonds.column_names)
            bonds = bonds.sort_by([(column, 'ascending') for column in sort_columns])

            def get_obj(table, index):
                f = table['figi'][index].as_py()
                c = table['cusip'][index].as_py()
                i = table['issuer'][index].as_py()
                m = table['maturity'][index].as_py()
                if f is None or c is None or i is None or m is None:
                    return None
                if m < time.time() * 1000000000:
                    print('already matured')
                    return None
                else:
                    print('not matured')
                return {
                    'T': table['ticker'][index].as_py() or '',
                    'F': f,
                    'C': c,
                    'I': table['isin'][index].as_py() or '',
                    't': table['ticker_symbol'][index].as_py() or '',
                    'c': table['parsed_coupon_string'][index].as_py() or '',
                    'm': datetime.datetime.fromtimestamp(m / 1e9).strftime('%Y-%m-%d'),
                    'i': i
                }

            with tempfile.TemporaryFile() as temp_file:
                obj_list = [get_obj(bonds, i) for i in range(len(bonds))]
                obj_list = [obj for obj in obj_list if obj is not None]
                temp_file.write(json.dumps(obj_list, indent=2).encode('utf-8'))
                temp_file.seek(0)
                s3.upload_fileobj(  temp_file,
                                    settings.get('$.bond_index_path'),
                                    {'Tagging': parse.urlencode({'public': 'true'})})

if __name__ == "__main__":
    _update_bond_index()
