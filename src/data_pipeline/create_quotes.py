from data_pipeline.parse_bondcliq import parse_bondcliq
from data_pipeline.helpers import get_sort_columns

def create_quotes():
    # parse and combine all quote sources
    quotes = parse_bondcliq()
    # sort
    sort_columns = get_sort_columns([   'entry_date',
                                        'cusip',
                                        'party_id',
                                        'entry_type',
                                        'price'], quotes.column_names)
    return quotes.sort_by([(column, 'ascending') for column in sort_columns])
