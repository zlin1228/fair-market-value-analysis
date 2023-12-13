import settings
settings.override_if_main(__name__, 1)

import os
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
import zipfile

from data_pipeline.create_bonds import create_bonds
from data_pipeline.create_quotes import create_quotes
from data_pipeline.create_ratings import create_ratings
from data_pipeline.create_trades import create_trades
from data_pipeline.helpers import map_ordinals, ordinal_re
from performance_helper import create_profiler
import s3

# value to use when an ordinal value is unspecified (missing/unknown, deliberately left out, etc.)
UNSPECIFIED_ORDINAL_VALUE = '__unspecified_ordinal_value__'

def _get_initial_ordinals():
    # default to empty ordinals seeded with the UNSPECIFIED_ORDINAL_VALUE
    ordinals = {
        'as_of_indicator': pa.array([UNSPECIFIED_ORDINAL_VALUE], type=pa.string(), size=1),
        'ats_indicator': pa.array([UNSPECIFIED_ORDINAL_VALUE], type=pa.string(), size=1),
        'buy_sell': pa.array([UNSPECIFIED_ORDINAL_VALUE], type=pa.string(), size=1),
        'coupon_string': pa.array([UNSPECIFIED_ORDINAL_VALUE], type=pa.string(), size=1),
        'currency': pa.array([UNSPECIFIED_ORDINAL_VALUE], type=pa.string(), size=1),
        'cusip': pa.array([UNSPECIFIED_ORDINAL_VALUE], type=pa.string(), size=1),
        'day_count_convention': pa.array([UNSPECIFIED_ORDINAL_VALUE], type=pa.string(), size=1),
        'entry_id': pa.array([UNSPECIFIED_ORDINAL_VALUE], type=pa.string(), size=1),
        'exchCode': pa.array([UNSPECIFIED_ORDINAL_VALUE], type=pa.string(), size=1),
        'figi': pa.array([UNSPECIFIED_ORDINAL_VALUE], type=pa.string(), size=1),
        'industry': pa.array([UNSPECIFIED_ORDINAL_VALUE], type=pa.string(), size=1),
        'isin': pa.array([UNSPECIFIED_ORDINAL_VALUE], type=pa.string(), size=1),
        'issuer': pa.array([UNSPECIFIED_ORDINAL_VALUE], type=pa.string(), size=1),
        'issuer_id': pa.array([UNSPECIFIED_ORDINAL_VALUE], type=pa.string(), size=1),
        'marketSector': pa.array([UNSPECIFIED_ORDINAL_VALUE], type=pa.string(), size=1),
        'message_category': pa.array([UNSPECIFIED_ORDINAL_VALUE], type=pa.string(), size=1),
        'message_type': pa.array([UNSPECIFIED_ORDINAL_VALUE], type=pa.string(), size=1),
        'party_id': pa.array([UNSPECIFIED_ORDINAL_VALUE], type=pa.string(), size=1),
        'rating': pa.array([UNSPECIFIED_ORDINAL_VALUE], type=pa.string(), size=1),
        'sector': pa.array([UNSPECIFIED_ORDINAL_VALUE], type=pa.string(), size=1),
        'side': pa.array([UNSPECIFIED_ORDINAL_VALUE], type=pa.string(), size=1),
        'securityType': pa.array([UNSPECIFIED_ORDINAL_VALUE], type=pa.string(), size=1),
        'securityType2': pa.array([UNSPECIFIED_ORDINAL_VALUE], type=pa.string(), size=1),
        'sub_product': pa.array([UNSPECIFIED_ORDINAL_VALUE], type=pa.string(), size=1),
        'ticker': pa.array([UNSPECIFIED_ORDINAL_VALUE], type=pa.string(), size=1),
        'trade_original_format': pa.array([UNSPECIFIED_ORDINAL_VALUE], type=pa.string(), size=1),
        'trade_status': pa.array([UNSPECIFIED_ORDINAL_VALUE], type=pa.string(), size=1),
        'trade_source': pa.array([UNSPECIFIED_ORDINAL_VALUE], type=pa.string(), size=1),
        'when_issued_indicator': pa.array([UNSPECIFIED_ORDINAL_VALUE], type=pa.string(), size=1)
    }
    # overwrite default empty ordinals with existing ordinals (if any)
    if s3.object_size(settings.get('$.data_path')):
        file_list = s3.get_object(settings.get('$.data_path'))
        ordinals: dict[str, pa.array] = {}
        for file in file_list:
            if match := ordinal_re.search(file):
                ordinal = match.group(1)
                ordinals[ordinal] = pq.read_table(file).column(ordinal).combine_chunks()
    return ordinals

def _update_model_data():
    profile = create_profiler('update_model_data')
    with profile():
        with profile('create_bonds'):
            bonds = create_bonds()
        with profile('create_quotes'):
            quotes = create_quotes()
        with profile('create_ratings'):
            ratings = create_ratings()
        with profile('create_trades'):
            trades = create_trades()
        with profile('get initial ordinals'):
            ordinals = _get_initial_ordinals()
        with profile('map ordinals'):
            bonds = map_ordinals(bonds, ordinals)
            quotes = map_ordinals(quotes, ordinals)
            ratings = map_ordinals(ratings, ordinals)
            trades = map_ordinals(trades, ordinals)
        with profile('creating and uploading zip') as p:
            # Create the zip file and upload it to S3
            with tempfile.TemporaryDirectory() as temp_dir:
                zip_path = os.path.join(temp_dir, '_new_data_.zip')
                with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED,
                                        compresslevel=settings.get('$.data_compression_level')) as zip:
                    def write_table(table, arcname):
                        '''Helper function to write a table to the zip file'''
                        p.print(f'writing {arcname}')
                        file_path = os.path.join(temp_dir, os.path.basename(arcname))
                        pq.write_table(table,
                                    file_path,
                                    row_group_size=64 * 1024,
                                    version="2.6",
                                    use_dictionary=False,
                                    compression='NONE')
                        zip.write(file_path, arcname=arcname)
                    # write the tables
                    write_table(bonds, 'bonds.parquet')
                    write_table(quotes, 'quotes.parquet')
                    write_table(ratings, 'ratings.parquet')
                    write_table(trades, 'trades.parquet')
                    # write the ordinals
                    for ordinal in ordinals.keys():
                        write_table(pa.table({ ordinal: ordinals[ordinal] }), f'ordinals/{ordinal}.parquet')
                s3.upload_file(zip_path, settings.get('$.data_path'))

if __name__ == "__main__":
    _update_model_data()
