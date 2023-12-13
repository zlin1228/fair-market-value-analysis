import csv
import io
from typing import Dict, Literal
import xml.etree.ElementTree as ET

import pyarrow as pa
import pyarrow.compute as pc

from data_pipeline.csv_parser import parse_csv_file
from data_pipeline.helpers import create_update_status
import s3
import settings

# XSD: https://xbrl.sec.gov/rocr/2015/ratings-2015-03-31.xsd
namespaces = {'ns': 'http://xbrl.sec.gov/ratings/2015-03-31'}

_delimiter: Literal[','] = ','
_quotechar: Literal['"'] = '"'
_column_types: Dict[str, pa.DataType] = {
    'cusip': pa.string(),
    'date': pa.timestamp('ns'),
    'rating': pa.string()
}

def _process(table: pa.table) -> pa.table:
    return pa.table({
        'cusip': table.column('cusip'),
        'rating_date': pc.assume_timezone(table.column('date'), settings.get('$.finra_timezone')).cast(pa.int64()),
        'rating': table.column('rating')})

def parse_moodys():
    paths = s3.get_all_paths_with_prefix(f"{settings.get('$.raw_data_path')}moodys/xbrl100-")
    paths = [p for p in paths if p.lower().endswith('.zip')]
    paths.sort()
    file_list = s3.get_object(paths[-1])
    with io.BytesIO() as binary_buffer:
        # don't use TextIOWrapper as a context manager since it is just wrapping binary_buffer
        text_buffer = io.TextIOWrapper(binary_buffer, encoding="utf-8", newline='')
        writer = csv.DictWriter(text_buffer, fieldnames=['cusip', 'date', 'rating'], delimiter=_delimiter, quotechar=_quotechar)
        writer.writeheader()
        xml_files = [file for file in file_list if file.lower().endswith('.xml')]
        update_status = create_update_status(len(xml_files))
        for i, input_file in enumerate(xml_files):
            # load the XML and get the root
            tree = ET.parse(input_file)
            root = tree.getroot()
            # find all ROCRA -> ISD -> IND elements
            for ind in root.iterfind('./ns:ROCRA/ns:ISD/ns:IND', namespaces):
                # iterate all CUSIP elements
                for cusip in ind.iterfind('ns:CUSIP', namespaces):
                    # make sure the CUSIP element contains a value (it should)
                    if cusip.text:
                        # iterate all INRD within the IND
                        for inrd in ind.iterfind('ns:INRD', namespaces):
                            # find the R (rating) and RAD (date)
                            r = inrd.find('ns:R', namespaces)
                            rad = inrd.find('ns:RAD', namespaces)
                            # make sure we found a rating and date
                            if r is not None and rad is not None and r.text and rad.text:
                                # write the row
                                writer.writerow({'cusip': cusip.text, 'date': rad.text, 'rating': r.text})
            update_status(i)
        binary_buffer.seek(0)
        return parse_csv_file(binary_buffer, _column_types, delimiter=_delimiter, quote_char=_quotechar, process=_process)
