import pyarrow as pa
import pyarrow.compute as pc

from data_pipeline.csv_parser import parse_csv_files
from data_pipeline.helpers import get_file_paths_of_cached_files
from performance_helper import create_profiler
import s3
import settings

_column_types: dict[str, pa.DataType] = {
    # '22': pa.uint32(),    # SecurityIDSource: Currently BondCliQ is only using CUSIP code. This will be extended in the future to other types like SEDOL, or ISIN (1 = CUSIP, 2 = SEDOL)
    '48': pa.string(),    # SecurityID: Security Identifier
    # '107': pa.string(),   # Concatenated string: "Ticker" + "Coupon Rate" + "Maturity"
    # '126': pa.string(),   # ExpireTime
    # '218': pa.float64(),  # Spread: Spread to benchmark in basis points.
    # '223': pa.float64(),  # Coupon Rate
    '269': pa.uint32(),   # MDEntryType: Market Data entry type (0 = Bid, 1 = Offer)
    '270': pa.float64(),  # MDEntryPx: Price per unit of quantity. Supported: Par, Spread(bps), Yield (whole percentage, i.e. 3.25% is sent as 3.25)
    '271': pa.float64(),  # MDEntrySize: Actual notional quantity.
    '272': pa.string(),   # MDEntryDate: Date of market data entry.
    '273': pa.string(),   # MDEntryTime: Time of market data entry. (the trade)
    # '278': pa.string(),   # MDEntryID: Unique Market Data Entry identifier
    # '279': pa.uint32(),   # MDUpdateAction: Type of market data update (0 = New, 1 = Change, 2 = Delete)
    '448': pa.string(),    # PartyID: Market data broker/dealer code.
    # '541': pa.string()   # ExpireDate
}

# To get the entry date, we combine 2 columns: MDEntryDate and MDEntryTime.
def _parse_bondcliq_entry_datetime(table: pa.table, columns: tuple[str,str]) -> pa.array:
    return pc.assume_timezone(
                pc.strptime(
                    pc.binary_join_element_wise(
                        table.column(columns[0]), table.column(columns[1]), 'T'
                    ),
                    format='%Y-%m-%dT%H:%M:%S',
                    unit='ns'
                ),
            settings.get('$.finra_timezone')).cast(pa.int64())

# Convert the raw table to parsed table.
# The SecurityID is now cusip.
# The MDEntryPx is now price.
# The MDEntrySize is now quantity.
def _process(table: pa.table) -> pa.table:
    return pa.table({
            'entry_type': table.column('269'),
            'cusip': table.column('48'),
            'price': table.column('270'),
            'quantity': table.column('271'),
            'entry_date': _parse_bondcliq_entry_datetime(table, ('272', '273')),
            'party_id': table.column('448')
            })

def parse_bondcliq() -> pa.table:
    with create_profiler('parse bondcliq')() as p:
        mapping = s3.get_all_with_prefix(f"{settings.get('$.raw_data_path')}bondcliq/pretrade_")
        file_paths = get_file_paths_of_cached_files(mapping)
        file_paths = [f for f in file_paths if f.lower().endswith('.csv')]
        result = parse_csv_files(file_paths, _column_types, quote_char=False, process=_process)
        p.print(f'rows: {result.num_rows}')
        return result
