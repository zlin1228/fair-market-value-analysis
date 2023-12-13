import sys
sys.path.insert(0, '../')

# ensure tensorflow does not initialize with GPU support (we load tensorflow before our code in this test)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# IANA 224.0.0.69-224.0.0.100 Unassigned (Multicast - Local Network Control Block) https://www.iana.org/assignments/multicast-addresses/multicast-addresses.xhtml
# IANA 8751-8762 Unassigned https://www.iana.org/assignments/service-names-port-numbers/service-names-port-numbers.xhtml?search=8751-8762
MEMORYDB_API_KEY = 'memorydb_api_key'
standard_settings = {
    "$.server.fmv.port": 8751,
    "$.data_path": "s3://deepmm.test.data/deepmm.parquet/v0.7.zip",
    "$.keras_models.load": True,
    "$.keras_models.use_gpu": False, # not all of our test environments have gpus
    "$.environment_settings.development.model_bucket": "deepmm.test.data/integration_test_models/test_residual_nn_date_range",
    "$.server.fmv.model": "residual_nn_integration_test",
    "$.enable_bondcliq": False,
    "$.enable_inference": True,
    "$.server.fmv.memorydb.batch_size": 100,
    "$.server.fmv.memorydb.propogation_wait_time": 0.1, # this is enough for the tests
    "$.multicast_data_split": 0.5,
    "$.finra_btds_multicast.ip": "224.0.0.80",
    "$.finra_btds_multicast.port": 8752,
    "$.finra_144a_multicast.ip": "224.0.0.81",
    "$.finra_144a_multicast.port": 8753}
import settings
settings.override(standard_settings)

import asyncio
from datetime import datetime
import json
import time
from typing import Callable
import unittest
from unittest.mock import Mock, patch
from websockets.client import connect

from tests.helpers import CleanEnvironment, create_token, FinraBroadcastMock

class MemoryDBMock():
    """MemoryDB mock implementation."""
    def __init__(self):
        self.data = {}
    def zadd(self, key, mapping):
        if key not in self.data:
            self.data[key] = {}
        prev = len(self.data[key])
        self.data[key] = {**self.data[key], **mapping}
        return len(self.data[key]) - prev
    def zrange(self, key, start, stop):
        if key not in self.data:
            return []
        snapshot = [(k,v) for k,v in self.data[key].items()]
        snapshot.sort(key=lambda e: e[1])
        return [e[0] for e in snapshot[start:(stop+1 if stop >= 0 else stop)]]

class ClientConnection:
    """Context helper wrapping a WebSocket client connection.

        Provides 'send' and 'expect' helper functions and remembers all messages it receives.

        Closes the socket when exiting."""
    def __init__(self, url: str):
        self.__url = url
        self.messages = []
        self.__expect_start_index = 0
    async def __aenter__(self):
        self.__connection = await connect(self.__url)
        self.__listener_task = asyncio.create_task(self.__listener())
        return self
    async def __aexit__(self, exc, value, tb):
        self.__listener_task.cancel()
        await self.__connection.close()
    async def __listener(self):
        while True:
            self.messages.append(json.loads(await self.__connection.recv()))
    def send(self, _json):
        return self.__connection.send(json.dumps(_json))
    async def expect(self, validator: Callable, timeout_message: str='timeout', timeout: int=60):
        start = time.time()
        while time.time() - start < timeout:
            for i, message in enumerate(self.messages, self.__expect_start_index):
                if validator(message):
                    self.__expect_start_index = i + 1
                    return True
            await asyncio.sleep(0.01)
        raise TimeoutError(timeout_message)

class CoroutineBatch:
    """Context helper that collects coroutines and awaits them as it exits."""
    def __init__(self):
        self.__coroutines = []
    async def __aenter__(self):
        return self.__add
    async def __aexit__(self, exc, value, tb):
        await asyncio.gather(*self.__coroutines)
    def __add(self, coroutine):
        self.__coroutines.append(coroutine)

def create_expected_trades_validator(expected_trades):
    def validator(m):
        # if the message does not contain 'trade' or the length is different
        if 'trade' not in m or len(m['trade']) != len(expected_trades):
            return False
        for i in range(len(expected_trades)):
            # get the keys in this trade except record_num
            keys = {k for k in m['trade'][i] if k != 'record_num'}
            # check if the keys match
            if keys != {k for k in expected_trades[i]}:
                return False
            # check if all the values match
            for k in keys:
                if m['trade'][i][k] != expected_trades[i][k]:
                    return False
        # if we get here then m matches expected_trades
        return True
    return validator

class TestFmvWebSocketServer(unittest.IsolatedAsyncioTestCase):

    async def test_server(self):
        # set up clean environment, mock environment variable(s), requests.get, and requests.post
        with CleanEnvironment(lambda _: standard_settings), \
                        patch('os.environ', { 'DEEPMM_MEMORYDB_API_KEY': MEMORYDB_API_KEY }), \
                        patch('requests.get') as get_mock, \
                        patch('requests.post') as post_mock:
            # now import the server code so it loads into the clean environment and picks up the mocks
            import fmv_websocket_server
            # create MemoryDB mock implementation
            self.memorydb_mock = MemoryDBMock()
            # create FINRA broadcast mock implementation
            self.finra_mock = FinraBroadcastMock(
                            settings.get('$.finra_btds_multicast.ip'),
                            settings.get('$.finra_btds_multicast.port'),
                            settings.get('$.finra_144a_multicast.ip'),
                            settings.get('$.finra_144a_multicast.port'))
            # create set that represents tokens we treat as authorized
            self.authorized_tokens = set()
            # create set that represents tokens we treat as active
            self.active_tokens = set()
            # ---------- create requests.get and requests.post mock implementations ----------
            MEMORYDB_API_BASE = settings.get(f'$.environment_settings.{settings.get("$.environment")}.memorydb.api')
            ENVIRONMENT = settings.get('$.environment')
            FINRA_CACHE_KEY = settings.get(f'$.environment_settings.{ENVIRONMENT}.memorydb.finra_cache_key')
            AUTHORIZEWEBSOCKET_URL = f"{settings.get(f'$.environment_settings.{ENVIRONMENT}.authentication.api')}/authorizewebsocket"
            CHECKWEBSOCKET_URL = f"{settings.get(f'$.environment_settings.{ENVIRONMENT}.authentication.api')}/checkwebsocket"
            # implement get mock
            def get_side_effect(*args, **kwargs):
                time.sleep(0.01) # simulate network latency
                resp = Mock()
                if len(args) and args[0] == f'{MEMORYDB_API_BASE}/zrange':
                    resp.text = json.dumps(self.memorydb_mock.zrange(**kwargs['params']))
                    resp.ok = True
                else:
                    raise ValueError(f'unmocked get request: {args[0]}')
                return resp
            get_mock.side_effect = get_side_effect
            # implement post mock
            def post_side_effect(*args, **kwargs):
                time.sleep(0.01) # simulate network latency
                resp = Mock()
                if len(args) and args[0] == f'{MEMORYDB_API_BASE}/zadd':
                    resp.text = f"{self.memorydb_mock.zadd(**kwargs['json'])}"
                    resp.ok = True
                elif len(args) and args[0] == AUTHORIZEWEBSOCKET_URL:
                    resp.json = lambda: { 'result': 'success' if kwargs['headers']['Authorization'] in self.authorized_tokens else 'access denied' }
                elif len(args) and args[0] == CHECKWEBSOCKET_URL:
                    resp.json = lambda: { 'result': 'active' if kwargs['headers']['Authorization'] in self.active_tokens else 'deactivated' }
                else:
                    raise ValueError(f'unmocked post request: {args[0]}')
                return resp
            post_mock.side_effect = post_side_effect
            # ---------- MemoryDB ----------
            # add old data to MemoryDB (overlaps date range of parquet file)
            # use a weekend day so there is no real data on that day making it easier to check that it is excluded
            old_finra_broadcast_mock = FinraBroadcastMock(
                            settings.get('$.finra_btds_multicast.ip'),
                            settings.get('$.finra_btds_multicast.port'),
                            settings.get('$.finra_144a_multicast.ip'),
                            settings.get('$.finra_144a_multicast.port'),
                            timestamp=datetime(2022, 2, 25, 8, 0, 0).timestamp())
            self.memorydb_mock.zadd(FINRA_CACHE_KEY, {m: r for r,m in await old_finra_broadcast_mock.create_trades(count=100)})
            # add current day MemoryDB data (make sure this isn't a multiple of the memorydb batch size to test partial batch)
            self.memorydb_mock.zadd(FINRA_CACHE_KEY, {m: r for r,m in await self.finra_mock.create_trades(count=1025)})
            # ---------- start the server ----------
            self.server_task = asyncio.create_task(fmv_websocket_server.main())
            # broadcast trades while we wait for a health check response indicating the server is ready
            trades_broadcasted_during_startup = 0
            while True:
                try:
                    async with connect(f"ws://localhost:{settings.get('$.server.fmv.port')}") as websocket:
                        await websocket.send(json.dumps({ 'health_check': True }))
                        print(f'successfully connected to server and received {json.loads(await websocket.recv())}')
                        break
                except:
                    self.memorydb_mock.zadd(FINRA_CACHE_KEY, {m: r for r,m in await self.finra_mock.broadcast_trades(count=100)})
                    trades_broadcasted_during_startup += 100
                    await asyncio.sleep(0.1)
            # make sure we've broadcast a fixed number of messages so the tests are deterministic
            if trades_broadcasted_during_startup < 500:
                await self.finra_mock.broadcast_trades(count=500-trades_broadcasted_during_startup)
            elif trades_broadcasted_during_startup > 500:
                raise IndexError('broadcast more than 500 trades during startup')
            # give the server a chance to process the new messages
            await asyncio.sleep(1)

            # ---------- test ----------
            # url of the test server
            url = f"ws://localhost:{settings.get('$.server.fmv.port')}"
            # create multiple clients to make sure the server can service multiple connections simultaneously
            async with ClientConnection(url) as client1, ClientConnection(url) as client2, ClientConnection(url) as client3:
                # Create a coroutine batch, all coroutines added to the batch will be awaited when the context is exiting.
                # The client and server code is all asynchronous so this lets us add send and expect coroutines to a batch
                # and simply wait for them to either return successfully or raise an exception if they fail.
                async with CoroutineBatch() as add:
                    # no token
                    add(client1.send({'fmv_stream': {'figi': 'BBG00K85WG22', 'size': 10, 'trade_count': 20}}))
                    add(client1.expect(lambda m: m == {'message': 'unauthorized'}, 'client1 did not receive unauthorized'))
                    # invalid token
                    add(client1.send({'token': create_token('invalid', 'token'), 'fmv_stream': {'figi': 'BBG00K85WG22', 'size': 10, 'trade_count': 5}}))
                    add(client1.expect(lambda m: m == {'message': 'forbidden'}, 'client1 did not receive forbidden'))

                    # valid token and valid request
                    token2 = create_token('c2 ojti', 'c2 jti')
                    self.authorized_tokens.add(token2)
                    self.active_tokens.add(token2)
                    add(client2.send({'token': token2, 'fmv_stream': {'figi': 'BBG00K85WG22', 'size': 10, 'trade_count': 100}}))
                    add(client2.expect(create_expected_trades_validator(self.finra_mock.get_expected_trades('BBG00K85WG22', 100)), 'trade not in client2 response'))

                    token3 = create_token('c3 ojti', 'c3 jti')
                    self.authorized_tokens.add(token3)
                    self.active_tokens.add(token3)
                    # figi that is filtered from parquet (no rating as of Mar 18, 2023)
                    add(client3.send({'token': token3, 'fmv_stream': {'figi': 'BBG00KGGX0D1', 'size': 1, 'trade_count': 100}}))
                    add(client3.expect(lambda m: {'message': 'Unrecognized figi'}, 'should have received "Unrecognized figi" in client3 response'))
                    # valid token, valid request
                    add(client3.send({'token': token3, 'fmv_stream': {'figi': 'BBG00J5HRSQ6', 'size': 1, 'trade_count': 100}}))
                    add(client3.expect(create_expected_trades_validator(self.finra_mock.get_expected_trades('BBG00J5HRSQ6', 100)), 'trade not in client3 response'))

                    token1 = create_token('c1 ojti', 'c1 jti')
                    self.authorized_tokens.add(token1)
                    self.active_tokens.add(token1)
                    # valid token, valid request
                    add(client1.send({'token': token1, 'fmv_stream': {'figi': 'BBG00K85WG22', 'size': 5, 'trade_count': 500}}))
                    add(client1.expect(create_expected_trades_validator(self.finra_mock.get_expected_trades('BBG00K85WG22', 500)), 'trade not in client1 response'))
                    # deactivate token
                    self.active_tokens.remove(token1)
                    add(client1.expect(lambda m: m == {'message': 'deactivated'}, 'client1 did receive deactivated'))

                    # new invalid token
                    add(client2.send({'token': create_token('c2_new ojti', 'c2_new jti'), 'fmv_stream': {'figi': 'BBG00K85WG22', 'size': 7, 'trade_count': 100}}))
                    add(client2.expect(lambda m: m == {'message': 'forbidden'}, 'client2 did not receive forbidden'))

                    # new valid token
                    token3_new = create_token('c3_new ojti', 'c3_new jti')
                    self.authorized_tokens.add(token3_new)
                    self.active_tokens.add(token3_new)
                    # valid token, valid request
                    add(client3.send({'token': token3_new, 'fmv_stream': {'figi': 'BBG00FM535P8', 'size': 1, 'trade_count': 200}}))
                    add(client3.expect(create_expected_trades_validator(self.finra_mock.get_expected_trades('BBG00FM535P8', 200)), 'trade not in client3 response'))


if __name__ == '__main__':
    unittest.main()
