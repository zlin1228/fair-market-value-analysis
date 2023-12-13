# HEAVILY MODIFIED DERIVATIVE OF THE CODE FOUND HERE:
# https://github.com/OpenFIGI/api-examples/blob/master/python/example-with-requests.py

# Copyright 2017 Bloomberg Finance L.P.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Iterable
import time

import requests

_MAX_JOBS_PER_REQUEST = 100
_MIN_REQUEST_INTERVAL = 6 / 25  # 25 per 6 seconds

_last_request_time = 0

def _map_jobs(jobs: Iterable[dict]):
    '''
    Generator that yields (job, result) tuples.  Takes care of batching and throttling.

    Parameters
    ----------
    jobs : iter(dict)
        An iterable of dicts that conform to the OpenFIGI API request structure. See
        https://www.openfigi.com/api#request-format for more information.

    Yields
    -------
    (dict, dict)
        First dict is the job, second dict is the result conforming to the OpenFIGI API
        response structure.  See https://www.openfigi.com/api#response-fomats
        for more information.
    '''
    url = 'https://api.openfigi.com/v3/mapping'
    headers = {'Content-Type': 'text/json', 'X-OPENFIGI-APIKEY': _API_KEY}
    batch = []
    def process_batch():
        global _last_request_time
        interval = time.time() - _last_request_time
        if interval < _MIN_REQUEST_INTERVAL:
            time.sleep(_MIN_REQUEST_INTERVAL - interval)
        _last_request_time = time.time()
        response = requests.post(url=url, headers=headers, json=batch)
        if response.status_code != 200:
            raise Exception('Bad response code {}'.format(str(response.status_code)))
        results = response.json()
        for job, result in zip(batch, results):
            yield (job, result)
        batch.clear()
    for job in jobs:
        batch.append(job)
        if len(batch) >= _MAX_JOBS_PER_REQUEST:
            for pair in process_batch():
                yield pair
    for pair in process_batch():
        yield pair

# ---------- END DERIVATIVE CODE ----------

import os
import tempfile
import zipfile

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as csv

from data_pipeline.csv_parser import parse_csv_file
from data_pipeline.helpers import create_update_status, get_sort_columns
import s3
import settings

_openfigi_cache_column_types: dict[str, pa.DataType] = {
                    'cusip': pa.string(),
                    'figi': pa.string(),
                    'name': pa.string(),
                    'ticker': pa.string(),
                    'exchCode': pa.string(),
                    'compositeFIGI': pa.string(),
                    'securityType': pa.string(),
                    'marketSector': pa.string(),
                    'shareClassFIGI': pa.string(),
                    'securityType2': pa.string(),
                    'securityDescription': pa.string()}
_openfigi_schema = pa.schema([pa.field(c, _openfigi_cache_column_types[c]) for c in _openfigi_cache_column_types])

def update_and_load_openfigi_cache(cusips: pa.array=None):
    s3_path = f"{settings.get('$.raw_data_path')}openfigi/openfigi_cache.zip"
    if s3.object_size(s3_path):
        openfigi_cache = parse_csv_file(s3.get_object(s3_path)[0], _openfigi_cache_column_types)
    else:
        openfigi_cache = pa.table({c: [] for c in _openfigi_cache_column_types}, schema=_openfigi_schema)
    if cusips and len(cusips):
        updated = False
        def update(id_type):
            nonlocal openfigi_cache
            nonlocal updated
            cusips_missing_from_cache = cusips.filter(pc.invert(pc.is_in(cusips, openfigi_cache['cusip'])))
            if len(cusips_missing_from_cache):
                update_status = create_update_status(len(cusips_missing_from_cache))
                additions = {c: [] for c in _openfigi_cache_column_types}
                for i, (job, result) in enumerate(_map_jobs({"idType": id_type, "idValue": c.as_py()} for c in cusips_missing_from_cache)):
                    if 'warning' in result:
                        print(f'''OpenFigi warning for request "{job}": "{result['warning']}"''')
                    if 'data' in result:
                        if len(result['data']) != 1:
                            print(f'''OpenFigi unexpected response for request "{job}": "{result['data']}"''')
                        else:
                            for c in (c for c in _openfigi_cache_column_types if c != 'cusip'):
                                additions[c].append(result['data'][0][c] if c in result['data'][0] else '')
                            additions['cusip'].append(job['idValue'])
                    update_status(i)
                if len(additions['cusip']):
                    updated = True
                    openfigi_cache = pa.concat_tables([openfigi_cache, pa.table(additions, schema=_openfigi_schema)])
        # see https://www.openfigi.com/api#v3-idType-values
        update('ID_CUSIP')  # CUSIP - Committee on Uniform Securities Identification Procedures.
        update('ID_CINS')  # CINS - CUSIP International Numbering System.
        if updated:
            openfigi_cache = openfigi_cache.sort_by([(c, 'ascending') for c in get_sort_columns(['cusip', 'figi'], openfigi_cache.column_names)])
            with tempfile.TemporaryDirectory() as temp_dir:
                def zip_and_put_table(table, name):
                    local_file = os.path.join(temp_dir, 'temp.csv')
                    csv.write_csv(table, local_file)
                    local_zip = os.path.join(temp_dir, 'temp.zip')
                    with zipfile.ZipFile(local_zip, 'w', compression=zipfile.ZIP_DEFLATED,
                                                compresslevel=settings.get('$.data_compression_level')) as zip:
                            zip.write(local_file, arcname=f'{name}.csv')
                    s3.upload_file(local_zip, f"{settings.get('$.raw_data_path')}openfigi/{name}.zip")
                zip_and_put_table(openfigi_cache, 'openfigi_cache')
                # TODO: this will just overwrite whatever was there each time, not ideal but good enough for a quick reference on what might be missing
                cusips_missing_from_openfigi = pa.table({'cusip': cusips.filter(pc.invert(pc.is_in(cusips, openfigi_cache['cusip'])))})
                cusips_missing_from_openfigi = cusips_missing_from_openfigi.sort_by([('cusip', 'ascending')])
                zip_and_put_table(cusips_missing_from_openfigi, 'cusips_missing_from_openfigi')
    return openfigi_cache
