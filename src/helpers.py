import datetime
import numpy as np
import pyarrow as pa


def _to_date_time(timestamp):
    if timestamp is datetime.datetime:
        return timestamp
    dt64 = np.datetime64(int(timestamp), 'ns')
    ts = (dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    return datetime.datetime.utcfromtimestamp(ts)


def epoch_ns_to_iso_str(epoch_ns, include_ms=False):
    if include_ms:
        return f"{datetime.datetime.fromtimestamp(epoch_ns / 1e9).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]}Z"
    else:
        return datetime.datetime.fromtimestamp(epoch_ns / 1e9).strftime('%Y-%m-%dT%H:%M:%SZ')


def from_single_date_time_to_timestamp(d):
    return int(d.timestamp()) * 1e9


def reverse_ordinals(ordinals):
    reverse = {}
    for k in ordinals:
        reverse[k] = {v: k for k, v in ordinals[k].items()}
    return reverse


def get_isin_from_cusip(cusip_str, country_code='US'):
    """
    >>> get_isin_from_cusip('037833100', 'US')
    'US0378331005'
    """
    isin_to_digest = country_code + cusip_str.upper()

    get_numerical_code = lambda c: str(ord(c) - 55)
    encode_letters = lambda c: c if c.isdigit() else get_numerical_code(c)
    to_digest = ''.join(map(encode_letters, isin_to_digest))

    ints = [int(s) for s in to_digest[::-1]]
    every_second_doubled = [x * 2 for x in ints[::2]] + ints[1::2]

    sum_digits = lambda i: sum(divmod(i, 10))
    digit_sum = sum([sum_digits(i) for i in every_second_doubled])

    check_digit = (10 - digit_sum % 10) % 10
    return isin_to_digest + str(check_digit)


def get_table_from_array(array, schema):
    column_names = [f.name for f in schema]
    table = pa.table({column_name: pa.array(array[:, i].flatten()) for i, column_name in enumerate(column_names)}, schema = schema)
    return table