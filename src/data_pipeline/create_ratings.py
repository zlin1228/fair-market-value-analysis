from data_pipeline.parse_moodys import parse_moodys
from data_pipeline.helpers import get_sort_columns

def create_ratings():
    # parse and combine all rating sources
    ratings = parse_moodys()
    # sort
    sort_columns = get_sort_columns([   'cusip',
                                        'rating_date',
                                        'rating'], ratings.column_names)
    return ratings.sort_by([(column, 'ascending') for column in sort_columns])
