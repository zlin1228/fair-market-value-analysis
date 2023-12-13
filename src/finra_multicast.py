# BTDS specification: https://www.finra.org/sites/default/files/BTDS-specs-v4.6A.pdf
# 144A specification: https://www.finra.org/sites/default/files/2022-02/BTDS-144A-Specs-v2.8.pdf
import settings

import base64
from pubsub import pub
import socket
import struct
import sys
import threading

from log import create_logger

FINRA_BTDS_MULTICAST_TOPIC = '--- FINRA BTDS MULTICAST TOPIC ---'
FINRA_144A_MULTICAST_TOPIC = '--- FINRA 144A MULTICAST TOPIC ---'

_listening_btds_semaphore = threading.BoundedSemaphore()
_listening_btds = False

_listening_144a_semaphore = threading.BoundedSemaphore()
_listening_144a = False

def listen(finra_multicast_topic, logger, local_address, group_address):
    """Starts the finra btds or 144a multicast listener

    Creates a socket that is added to the multicast group and enters an infinite while loop
    that uses a blocking recvfrom call to wait for data to arrive on the socket and then
    publishes the data it receives to the FINRA_BTDS_MULTICAST_TOPIC or FINRA_144A_MULTICAST_TOPIC.

    The listener should be started as a daemon thread:
        listen_thread = threading.Thread(target=listen)
        listen_thread.daemon = True # note: this must be set before start()
        listen_thread.start()

    Subscribers to FINRA_BTDS_MULTICAST_TOPIC or FINRA_144A_MULTICAST_TOPIC should note that 
    their handler will get called from the listen thread so they should consider thread safety 
    when processing the data they receive.
    """

    # create the socket
    logger.info('creating socket')
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP) as sock:
        # set SO_REUSEADDR because linux requires that all processes belong to the same
        # effective userid when listening to the same multicast using SO_REUSEPORT
        # see  https://stackoverflow.com/a/14388707/10149510
        logger.info(f'setting SO_REUSEADDR on socket')
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # add socket to multicast group
        logger.info(f'adding socket to group {group_address}')
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, group_address)
        # bind socket
        logger.info(f'binding socket to {local_address}')
        sock.bind(local_address)
        # listen forever
        while True:
            # BTDS Version 4.6A spec and 144A Version 2.8 spec: Each block contains a 
            # maximum of 1000 data characters. Messages may not span blocks.
            data, _ = sock.recvfrom(1024)
            base64data = base64.b64encode(data).decode('ascii')
            logger.debug(base64data)
            # publish data we receive to the topic
            pub.sendMessage(finra_multicast_topic, message=base64data)

def listen_btds():
    global _listening_btds_semaphore
    global _listening_btds
    with _listening_btds_semaphore:
        if _listening_btds:
            raise RuntimeError('Already listening')
        _listening_btds = True

    logger = create_logger(f'finra_btds_multicast')
    local_address = '0.0.0.0', settings.get(f'$.finra_btds_multicast.port')
    group_address = struct.pack('4sL', socket.inet_aton(settings.get(f'$.finra_btds_multicast.ip')), socket.INADDR_ANY)
    listen(FINRA_BTDS_MULTICAST_TOPIC, logger, local_address, group_address)

def listen_144a():
    global _listening_144a_semaphore
    global _listening_144a
    with _listening_144a_semaphore:
        if _listening_144a:
            raise RuntimeError('Already listening')
        _listening_144a = True

    logger = create_logger(f'finra_144a_multicast')
    local_address = '0.0.0.0', settings.get(f'$.finra_144a_multicast.port')
    group_address = struct.pack('4sL', socket.inet_aton(settings.get(f'$.finra_144a_multicast.ip')), socket.INADDR_ANY)
    listen(FINRA_144A_MULTICAST_TOPIC, logger, local_address, group_address)

def create_listen_thread(listen):
    listen_thread = threading.Thread(target=listen)
    listen_thread.daemon = True
    listen_thread.start()

def run_listen_thread(finra_multicast_topic, listen):
    def handler(message):
        print(f'received: {message}')
    print('subscribing to multicast topic')
    pub.subscribe(handler, finra_multicast_topic)
    print('starting listener')
    listen_thread = threading.Thread(target=listen)
    listen_thread.daemon = True
    listen_thread.start()
    print('listening forever')
    listen_thread.join() # join to listen forever
    # unreachable since we're listening forever, but this is how to clean up
    pub.unsubscribe(handler, finra_multicast_topic)

if __name__ == "__main__":
    if sys.argv[1] == 'btds':
        run_listen_thread(FINRA_BTDS_MULTICAST_TOPIC, listen_btds)
    elif sys.argv[1] == '144a':
        run_listen_thread(FINRA_144A_MULTICAST_TOPIC, listen_144a)
    else:
        raise Exception(f"Multicast type is not recognized")