import pyarrow as pa

from data_pipeline.parse_finra_historical import parse_finra_historical
from data_pipeline.parse_finra_eod import parse_finra_eod
from data_pipeline.helpers import get_sort_columns
import settings

def create_trades():
    trades = pa.concat_tables([
        parse_finra_historical(f"{settings.get('$.raw_data_path')}finra/historical/trace_cbc.zip", 'trace_cbc_historical'),
        parse_finra_historical(f"{settings.get('$.raw_data_path')}finra/historical/trace_cb_144ac.zip", 'trace_cb_144ac_historical'),
        parse_finra_eod(f"{settings.get('$.raw_data_path')}finra/trace_cbc/", 'trace_cbc'),
        parse_finra_eod(f"{settings.get('$.raw_data_path')}finra/trace_cb_144ac/", 'trace_cb_144ac')
    ])
    # sort
    sort_columns = get_sort_columns([   'report_date',
                                        'execution_date',
                                        'figi',
                                        'cusip',
                                        'price',
                                        'quantity'], trades.column_names)
    return trades.sort_by([(column, 'ascending') for column in sort_columns])
