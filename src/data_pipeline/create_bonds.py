from data_pipeline.helpers import get_sort_columns
from data_pipeline.parse_cbonds import parse_cbonds

def create_bonds():
    # parse and combine all bond sources
    bonds = parse_cbonds()
    # sort
    sort_columns = get_sort_columns([   'ticker',
                                        'maturity'], bonds.column_names)
    return bonds.sort_by([(column, 'ascending') for column in sort_columns])
