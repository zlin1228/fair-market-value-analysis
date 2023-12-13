import settings
settings.override_if_main(__name__, 2)

import asyncio
import json
import os
import sys
import threading
import time

from pubsub import pub
from websockets.client import connect

from finra_multicast import FINRA_BTDS_MULTICAST_TOPIC, listen_btds, \
                            FINRA_144A_MULTICAST_TOPIC, listen_144a
from helpers import epoch_ns_to_iso_str
from database.memory_db import create_send_to_memorydb_task

from log import create_logger
memorydb_logger = create_logger('health_checker__memorydb')
logger = create_logger('health_checker')

async def check(ports):
    health_checker_start = epoch_ns_to_iso_str(time.time_ns(), include_ms=True)
    logger.info(f'port list: {", ".join((str(p) for p in ports))}')

    last_btds_message = 0.0
    # note: this gets called on a different thread
    def handler_btds(_):
        nonlocal last_btds_message
        last_btds_message = time.time_ns()
        logger.debug(f'setting last_btds_message to {last_btds_message}')
    pub.subscribe(handler_btds, FINRA_BTDS_MULTICAST_TOPIC)

    last_144a_message = 0.0
    # note: this gets called on a different thread
    def handler_144a(_):
        nonlocal last_144a_message
        last_144a_message = time.time_ns()
        logger.debug(f'setting last_144a_message to {last_144a_message}')
    pub.subscribe(handler_144a, FINRA_144A_MULTICAST_TOPIC)

    create_send_to_memorydb_task(memorydb_logger)

    def run_thread(thread):
        thread.daemon = True
        thread.start()

    listen_btds_thread = threading.Thread(target=listen_btds)
    run_thread(listen_btds_thread)
    listen_144a_thread = threading.Thread(target=listen_144a)
    run_thread(listen_144a_thread)

    if not os.path.isdir('../www'):
        logger.info('making ../www directory')
        os.mkdir('../www')

    try:
        while True:
            await asyncio.sleep(settings.get('$.health_checker.fmv.interval'))
            try:
                results = {
                    'health_checker_start': health_checker_start,
                    'start': epoch_ns_to_iso_str(time.time_ns(), include_ms=True),
                    'ports': ports,
                    'port_status': {}
                }
                for port in ports:
                    try:
                        async with connect(f'ws://localhost:{port}') as websocket:
                            await websocket.send(json.dumps({ 'health_check': True }))
                            resp = json.loads(await websocket.recv())
                            results['port_status'][port] = {k: epoch_ns_to_iso_str(v, include_ms=True) for k, v in resp.items()}
                    except BaseException as e:
                        logger.exception(f'error accessing port {port}')
                        results['port_status'][port] = { 'error': str(e) }
                results['end'] = epoch_ns_to_iso_str(time.time_ns(), include_ms=True)
                results['last_btds_message'] = epoch_ns_to_iso_str(last_btds_message, include_ms=True) if last_btds_message else None
                results['last_144a_message'] = epoch_ns_to_iso_str(last_144a_message, include_ms=True) if last_144a_message else None
                logger.debug(json.dumps(results))
                with open('../www/health.json', 'w') as output:
                    output.write(json.dumps(results))
            except BaseException as e:
                logger.exception('error in check')
                with open('../www/health.json', 'w') as output:
                    output.write(json.dumps({ 'error': str(e) }))
    except:
        logger.exception('error causing health_checker exit')
    # keep listening and sending to memorydb even if health checking failed
    listen_btds_thread.join()
    listen_144a_thread.join()


def main():
    try:
        ports = [int(p) for p in sys.argv[1].split(',') if p]
    except:
        print('Usage: $ python health_checker.py <comma-separated port list ex: 8810,8811,8812 > <optional settings overrides>')
        return
    asyncio.run(check(ports))

if __name__ == "__main__":
    main()
