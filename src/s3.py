import json
import os
import re
import sys
from typing import Dict, List, Union

import boto3
import lz4.frame
import zipfile

from performance_helper import create_profiler
import settings

# resource...will be phased out of AWS
s3 = boto3.resource('s3')
# client...will have long-term support
s3_client = boto3.client('s3')

quoted_re = re.compile('"([^"]*)"')
s3url_re = re.compile('s3://([^/]*)/(.*)')

_get_etag = lambda obj: quoted_re.search(obj.e_tag).group(1)


def get_object(s3url, etag=None):
    """Gets an S3 object

    Extracts .zip files and returns a list of paths to the extracted file(s)
    Decompresses .lz4 files and returns the path to the decompressed file
    Saves all other files and returns the path to the file

    Caches objects and only downloads/extracts if not already in cache

    Keyword arguments:
    s3url -- an S3 URL in the format s3://bucket-name/key-name
    etag -- optional, can be provided if already retrieved from S3
    """
    profile = create_profiler(f's3.get_object {s3url}')
    with profile() as p:
        match = s3url_re.search(s3url)
        bucket = match.group(1)
        key = match.group(2)
        obj = s3.Object(bucket, key)
        s3_cache_path = settings.get('$.s3_cache_path')
        if not os.path.isdir(s3_cache_path):
            os.mkdir(s3_cache_path)
            p.print('created s3 cache directory')
        etag = etag or _get_etag(obj)
        d_path = os.path.join(s3_cache_path, etag)
        if not os.path.isdir(d_path):
            p.print(f'making etag directory: {etag}')
            os.mkdir(d_path)
        f_path = os.path.join(d_path, os.path.basename(key))
        m_path = os.path.join(d_path, '_manifest.json')
        if not os.path.isfile(m_path):
            with profile('download'):
                obj.download_file(f_path)
            root, ext = os.path.splitext(f_path)
            file_list = [f_path]
            if ext.lower() == '.zip':
                with profile('extract'):
                    e_path = os.path.join(d_path, 'expanded')
                    with zipfile.ZipFile(f_path, 'r') as zf:
                        namelist = zf.namelist()
                        zf.extractall(e_path)
                os.remove(f_path)
                file_list = [os.path.join(e_path, file) for file in namelist if not file.endswith('/')]
            elif ext.lower() == '.lz4':
                with profile('decompress'):
                    with lz4.frame.open(f_path, mode='rb') as lz4_file:
                        with open(root, 'wb') as output_file:
                            output_file.write(lz4_file.read())
                os.remove(f_path)
                file_list = [root]
            with open(m_path, 'wt') as f:
                f.write(json.dumps(file_list))
        with open(m_path, 'rt') as f:
            manifest = json.loads(f.read())
            p.print(f'{s3url} cached in {d_path}')
            return manifest

def get_all_with_prefix(s3url) -> Dict[str,Union[List[str],str]]:
    """Gets all zip files with the S3 prefix

    Finds all .zip files with the given prefix
    Extracts the zips and returns a dict mapping S3 path to extracted file(s)
    Caches objects and only downloads/extracts if not already in cache

    Keyword arguments:
    s3url -- an S3 URL in the format s3://bucket-name(/prefix)?
    """
    with create_profiler(f's3.get_all_with_prefix {s3url}')():
        match = s3url_re.search(s3url) or s3url_re.search(s3url + '/')
        bucket = match.group(1)
        key = match.group(2)
        b = s3.Bucket(bucket)
        obj_list = b.objects.filter(Prefix=key)
        # note: creating a "folder" in the AWS S3 UI creates an item of size 0 with a trailing delimiter
        result = {obj_url: get_object(obj_url, _get_etag(obj))
                  for obj in obj_list
                  if not (obj_url := f's3://{obj.bucket_name}/{obj.key}').endswith('/')}
        return result

def get_all_paths_with_prefix(s3url) -> List[str]:
    """Gets all paths with the S3 prefix

    Finds all files with the given prefix
    Returns a list of S3 paths

    Keyword arguments:
    s3url -- an S3 URL in the format s3://bucket-name(/prefix)?
    """
    with create_profiler(f's3.get_all_paths_with_prefix {s3url}')():
        match = s3url_re.search(s3url) or s3url_re.search(s3url + '/')
        bucket = match.group(1)
        key = match.group(2)
        b = s3.Bucket(bucket)
        obj_list = b.objects.filter(Prefix=key)
        # note: creating a "folder" in the AWS S3 UI creates an item of size 0 with a trailing delimiter
        result = [f's3://{obj.bucket_name}/{obj.key}' for obj in obj_list if not obj.key.endswith('/')]
        return result

def upload_file(path: str, s3url: str, extra_args={}):
    """Uploads a local file to S3

    Keyword arguments:
    path -- path to local file
    s3url -- an S3 URL in the format s3://bucket-name/key-name
    """
    profile = create_profiler(f's3.upload_file "{path}" to "{s3url}"')
    with profile() as p:
        match = s3url_re.search(s3url)
        bucket = match.group(1)
        key = match.group(2)
        s3_client.upload_file(path, bucket, key, extra_args)

def upload_fileobj(fileobj, s3url: str, extra_args={}):
    """Uploads a readable file-like object to S3.  File object must be opened in binary mode, not text mode.

    Keyword arguments:
    fileobj -- file-like object opened in binary mode
    s3url -- an S3 URL in the format s3://bucket-name/key-name
    """
    profile = create_profiler(f's3.upload_fileobj to "{s3url}"')
    with profile() as p:
        match = s3url_re.search(s3url)
        bucket = match.group(1)
        key = match.group(2)
        s3_client.upload_fileobj(fileobj, bucket, key, extra_args)

def delete_object(s3url: str):
    """Deletes an object from S3.  Deletes are only allowed on objects in the deepmm.temp bucket.

    Keyword arguments:
    s3url -- an S3 URL in the format s3://bucket-name/key-name
    """
    profile = create_profiler(f's3.delete_object "{s3url}"')
    with profile():
        match = s3url_re.search(s3url)
        bucket = match.group(1)
        key = match.group(2)
        if bucket == 'deepmm.temp':
            obj = s3.Object(bucket, key)
            obj.delete()
        else:
            raise ValueError('Deletes are only allowed on deepmm.temp')

def copy_object(s3fromurl, s3tourl):
    """Copy an S3 object

    Copy an S3 obj to another location

    Keyword arguments:
    s3fromurl -- an S3 URL in the format s3://bucket-name/key-name
    s3tourl -- an S3 URL in the format s3://bucket-name/key-name
    """
    profile = create_profiler(f's3.copy_object {s3fromurl} {s3tourl}')
    with profile() as p:
        fmatch = s3url_re.search(s3fromurl)
        if fmatch:
            copy_source = {'Bucket': fmatch.group(1), 'Key': fmatch.group(2)}
            tmatch = s3url_re.search(s3tourl)
            if tmatch:
                s3.meta.client.copy(copy_source, Bucket=tmatch.group(1), Key=tmatch.group(2))
            else:
                print('Usage: s3tourl <s3://bucket-name/key-name>')
        else:
            print('Usage: s3fromurl <s3://bucket-name/key-name>')

def object_size(s3url):
    """Return size if the S3 object exists, or None if not

    Keyword arguments:
    s3url -- an S3 URL in the format s3://bucket-name/key-name
    """
    profile = create_profiler(f's3.object_size {s3url}')
    with profile() as p:
        match = s3url_re.search(s3url)
        if match:
            bucket = match.group(1)
            objkey = match.group(2)
            response = s3.meta.client.list_objects_v2(
                Bucket=bucket,
                Prefix=objkey
            )
            for obj in response.get('Contents', []):
                if obj['Key'] == objkey:
                    return obj['Size']
    return None

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: <s3://bucket-name/key-name>')
        exit()
    p = get_all_with_prefix(sys.argv[1])
    print(p)
