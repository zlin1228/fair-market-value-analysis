import asyncio
import base64
import json
import os
import threading
from typing import Dict

from pubsub import pub
import requests

from finra_multicast import FINRA_BTDS_MULTICAST_TOPIC, FINRA_144A_MULTICAST_TOPIC
from finra import parse_data
from log import create_logger
import settings

logger = create_logger('memory_db')

API_BASE = settings.get(f'$.environment_settings.{settings.get("$.environment")}.memorydb.api')
ENVIRONMENT = settings.get('$.environment')
FINRA_CACHE_KEY = settings.get(f'$.environment_settings.{ENVIRONMENT}.memorydb.finra_cache_key')
MEMORYDB_BATCH_SIZE = settings.get('$.server.fmv.memorydb.batch_size')
MEMORYDB_INTERVAL = settings.get('$.server.fmv.interval.memorydb')

def zadd(key: str, mapping: Dict[str, int]):
    try:
        logger.debug(f'zadd called with "{key}", {mapping}')
        response = requests.post(
            f'{API_BASE}/zadd',
            headers={'X-API-key': os.environ['DEEPMM_MEMORYDB_API_KEY'],
                        'Content-Type': 'application/json'},
            json={
                'key': key,
                'mapping': mapping
            })
        if response.ok:
            logger.info(f'zadd added {response.text} values to "{key}"')
        else:
            logger.error(f'zadd failed: {response}')
    except:
        logger.exception('zadd error')
        raise

def zrange(key, start, stop):
    try:
        logger.debug(f'zrange called with {key}, {start}, {stop}')
        response = requests.get(
            f'{API_BASE}/zrange',
            headers={'X-API-key': os.environ['DEEPMM_MEMORYDB_API_KEY']},
            params={
                'key': key,
                'start': start,
                'stop': stop
            }
        )
        if response.ok:
            logger.debug(response.text)
            data = json.loads(response.text)
            logger.info(f'zrange called with {key}, {start}, {stop} returned {len(data)} values')
            return data
        else:
            logger.error(f'zrange failed: {response}')
            return []
    except:
        logger.exception('zrange error')
        raise

def create_send_to_memorydb_task(data_logger):
    memorydb_outbound_buffer = set()
    def send_to_memorydb():
        try:
            nonlocal memorydb_outbound_buffer
            batch = memorydb_outbound_buffer.copy()
            if batch:
                memorydb_outbound_buffer -= batch # note: use -= since adding happens on a different thread
                data_logger.info(f'sending batch of {len(batch)} to MemoryDB')
                data_logger.debug(batch)
                # spawn a thread since this will block
                # how to handle logging if we want to spawn a process instead and move this completely off the event loop:
                # https://docs.python.org/3/howto/logging-cookbook.html#logging-to-a-single-file-from-multiple-processes
                threading.Thread(target=zadd, args=(FINRA_CACHE_KEY,
                                { m : int(parse_data(base64.b64decode(m), True)['DATE_TIME'].timestamp()) for m in batch})).start()
        except:
            data_logger.exception('error sending data to memorydb')
    # note: this gets called on a different thread
    def memorydb_handler(message):
        memorydb_outbound_buffer.add(message)
        if len(memorydb_outbound_buffer) >= MEMORYDB_BATCH_SIZE:
            send_to_memorydb() # send if our buffer is full
    async def periodically_send_to_memorydb():
        while True:
            await asyncio.sleep(MEMORYDB_INTERVAL)
            send_to_memorydb() # send if we haven't sent recently
    memorydb_task = asyncio.create_task(periodically_send_to_memorydb())
    pub.subscribe(memorydb_handler, FINRA_BTDS_MULTICAST_TOPIC)
    pub.subscribe(memorydb_handler, FINRA_144A_MULTICAST_TOPIC)
    def cleanup():
        try:
            pub.unsubscribe(memorydb_handler, FINRA_BTDS_MULTICAST_TOPIC)
            pub.unsubscribe(memorydb_handler, FINRA_144A_MULTICAST_TOPIC)
        except:
            data_logger.exception('error unsubscribing memorydb_handler')
        try:
            memorydb_task.cancel()
        except:
            data_logger.exception('error cancelling memorydb_task')
    return cleanup
