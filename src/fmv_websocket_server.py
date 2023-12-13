import settings
settings.override({"$.enable_inference": True})
settings.override_if_main(__name__, 1)

import asyncio
import base64
import itertools
import json
import re
import time
from typing import Callable
import uuid

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pubsub import pub
import requests
from websockets.server import serve
from websockets.exceptions import ConnectionClosedOK

from finra_multicast import FINRA_BTDS_MULTICAST_TOPIC, listen_btds, \
                            FINRA_144A_MULTICAST_TOPIC, listen_144a, \
                            create_listen_thread
from data_pipeline.helpers import map_ordinals
import database.memory_db as memory_db
from finra import parse_messages
from generator import TraceDataGenerator
from helpers import epoch_ns_to_iso_str
from load_data import create_process_trades, get_initial_trades, get_ordinals
from log import create_logger
from model import TraceModel
from performance_helper import create_profiler
from trade_history import TradeHistory
from train_evaluate import get_model_class
from typical_trade import create_fmv_json


logger = create_logger('fmv_websocket_server')
data_logger = create_logger('fmv_websocket_server__data')
websockets_logger = create_logger('fmv_websocket_server__websockets')

ENVIRONMENT = settings.get('$.environment')
PORT = settings.get('$.server.fmv.port')
FINRA_CACHE_KEY = settings.get(f'$.environment_settings.{ENVIRONMENT}.memorydb.finra_cache_key')
MEMORYDB_BATCH_SIZE = settings.get('$.server.fmv.memorydb.batch_size')
MEMORYDB_PROPOGATION_WAIT_TIME = settings.get('$.server.fmv.memorydb.propogation_wait_time')
AUTHORIZEWEBSOCKET_URL = f"{settings.get(f'$.environment_settings.{ENVIRONMENT}.authentication.api')}/authorizewebsocket"
CHECKWEBSOCKET_URL = f"{settings.get(f'$.environment_settings.{ENVIRONMENT}.authentication.api')}/checkwebsocket"
DATA_LOOP_INTERVAL = settings.get('$.server.fmv.interval.data_loop')
HEARTBEAT_INTERVAL = settings.get('$.server.fmv.interval.heartbeat')
MODEL_INTERVAL = settings.get('$.server.fmv.interval.model')

token_re = re.compile(r'"token"\s*:\s*"[^"]*"')
def scrub_token(msg): return token_re.sub('"token": "<token>"', msg)

_health_stats__server_started = 0.0
_health_stats__last_data_loop_run = 0.0
_health_stats__last_data_loop_batch = 0.0

class WebSocketWrapper:
    def __init__(self, websocket, recv_handler,
                    info: Callable[[str],None]=websockets_logger.info,
                    debug: Callable[[str],None]=websockets_logger.debug,
                    error: Callable[[str],None]=websockets_logger.error,
                    exception: Callable[[str],None]=websockets_logger.exception):
        self.__websocket = websocket
        self.__recv_handler = recv_handler
        self.__info = info
        self.__debug = debug
        self.__error = error
        self.__exception = exception
        self.__tasks = set()
        self.__listening = False
        self.__exiting = False
        self.exited = asyncio.Future()
    def start_listening(self):
        if not self.__listening:
            self.__info('starting to listen')
            self.__listening = True
            self.create_task(self.__listener())
        else:
            self.__error('start_listening called more than once')
    def send(self, msg):
        self.__debug(f'sending {msg}')
        self.create_task(self.__websocket.send(msg))
    def create_task(self, coroutine):
        name = getattr(coroutine, '__name__', '<unnamed coroutine>')
        if not self.__exiting:
            self.__debug(f'{name} - wrapping')
            async def wrapper():
                self.__debug(f'{name} - started')
                task = asyncio.create_task(coroutine)
                self.__tasks.add(task)
                try:
                    await task
                except asyncio.CancelledError:
                    self.__debug(f'{name} - cancelled')
                except ConnectionClosedOK:
                    self.__debug(f'{name} - connection closed ok')
                    self.exit()
                except:
                    self.__exception(f'{name} - exception')
                    self.exit()
                self.__tasks.remove(task)
                self.__debug(f'{name} - finished')
            return asyncio.create_task(wrapper())
        else:
            self.__debug(f'exiting so not creating task for {name}')
    def exit(self):
        if not self.__exiting:
            self.__debug('exiting')
            self.__exiting = True
            asyncio.create_task(self.__exit_handler())
    async def __listener(self):
        while True:
            msg = await self.__websocket.recv()
            self.__debug(f'received {scrub_token(msg)}')
            self.__recv_handler(msg)
    async def __exit_handler(self):
        self.__debug('exit - starting')
        for task in self.__tasks:
            try:
                self.__debug('exit - cancelling task')
                task.cancel()
            except BaseException as e:
                self.__debug(f'exit - exception cancelling task: {e}')
        # sleep twice to let the CancelledError exceptions propogate
        # through the tasks and then the task wrappers
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        if len(self.__tasks):
            self.__error(f'exit - tasks not empty, {len(self.__tasks)} remaining')
        self.exited.set_result(None)
        self.__info('exited')

class FMV_Handler:
    def __init__(self, websocket, websocketid, generator: TraceDataGenerator, model):
        self.__websocketid = websocketid
        self.__wrapped_websocket = WebSocketWrapper(websocket, self.__recv_handler, self.__info, self.__debug, self.__error, self.__exception)
        self.__generator = generator
        self.__model = model
        # Get a reverse dictionary of the ordinals for side which is a pyarrow StringArray:)
        side = get_ordinals()['side'].to_pylist()
        self.__side_num_to_side = {ordinal: string for string, ordinal in zip(side, range(len(side)))}
        buy_sell = get_ordinals()['buy_sell'].to_pylist()
        self.__buy_sell_num_to_buy_sell = {ordinal: string for string, ordinal in zip(buy_sell, range(len(buy_sell)))}
        self.__record_num = itertools.count() # <- TODO: FIX THIS, this is a workaround
        self.__token = None
        self.__reset = lambda: None
    def __info(self, msg):
        websockets_logger.info(f'{self.__websocketid}: {msg}')
    def __debug(self, msg):
        websockets_logger.debug(f'{self.__websocketid}: {msg}')
    def __error(self, msg):
        websockets_logger.error(f'{self.__websocketid}: {msg}')
    def __exception(self, msg):
        websockets_logger.exception(msg)
    def __send_msg(self, msg):
        self.__wrapped_websocket.send(json.dumps({ 'message': msg}))
    def __send_json(self, the_json):
        self.__wrapped_websocket.send(json.dumps(the_json))

    async def serve(self):
        self.__wrapped_websocket.start_listening()
        await self.__wrapped_websocket.exited
        self.__reset()

    ###################################################
    ## Start of __recv_handler related methods.
    ###################################################
    def __authorize_websocket(self, atts):
        # authorize
        if not self.__token:
            self.__send_msg('unauthorized')
            return False
        self.__info('authorizing websocket')
        authRes = requests.post(AUTHORIZEWEBSOCKET_URL, headers={'Authorization': self.__token},
                                json={
                                    'websocketid': self.__websocketid,
                                    'log': f"{atts['figi']},{atts['size']},{atts['trade_count']}"
                                }).json()
        self.__info(f'authorization response: {authRes}')
        if 'result' not in authRes or authRes['result'] != 'success':
            self.__send_msg('forbidden')
            return False

        return True
    def __recv_token(self, msg):
        self.__token = msg['token']
        origin_jti = None
        jti = None
        try:
            # extract the claims, we don't need to validate the token since that is done in AWS
            claims = json.loads(base64.b64decode(f"{self.__token.split('.')[1]}==").decode('utf-8'))
            origin_jti = claims['origin_jti']
            jti = claims['jti']
        except:
            self.__exception('error extracting token claims')
        self.__info(f'updated token: {origin_jti or ""}, {jti or ""}')

    def __recv_health_check(self):
        self.__info('health check')
        self.__send_json({"server_started": _health_stats__server_started,
                          "last_data_loop_run": _health_stats__last_data_loop_run,
                          "last_data_loop_batch": _health_stats__last_data_loop_batch})
        return

    def __check_for_required_attributes(self, atts):
        for att in ['figi', 'trade_count', 'size']:
            if att not in atts:
                self.__send_msg(f'missing {att}')
                return False
        return True

    def __parse_size(self, atts):
        try:
            size = float(atts['size']) * 1_000_000.0
            assert size > 0.0
        except:
            self.__send_msg('Invalid size')
            return None
        return size

    def __parse_trade_count(self, atts):
        try:
            trade_count = int(atts['trade_count'])
            assert trade_count > 0
        except:
            self.__send_msg('Invalid trade_count')
            return None
        return trade_count

    def __parse_figi(self, atts):
        try:
            figi = get_ordinals()['figi'].index(atts['figi']).as_py()
            if figi < 0:
                raise KeyError()
        except:
            self.__send_msg('Unrecognized figi')
            return None
        return figi

    def __parse_figi_trace(self, figi, trade_count):
        figi_trace = self.__generator.get_figi_trace(figi, trade_count)
        if figi_trace is None or figi_trace.shape[0] <= 0:
            self.__send_msg('Unrecognized figi')
            return None
        return figi_trace

    def start_tasks(self, figi_row, size):
        topic = str(figi_row["figi"][0].as_py())
        serve_model_task = self.__wrapped_websocket.create_task(self.__serve_model(figi_row, size))
        heartbeat_task = self.__wrapped_websocket.create_task(self.__heartbeat())
        pub.subscribe(self.__new_trades_listener, topic)
        self.set_fmv_stream_reset(topic, serve_model_task, heartbeat_task)

    def send_initial_trades(self, figi_row, figi_trace, size, trade_count):
        trades = figi_trace.slice(max(0, figi_trace.num_rows - trade_count))
        first_execution_date = trades['execution_date'][0].as_py()
        last_execution_date = time.time_ns()
        initial_model_execution_dates = pa.array(np.linspace(first_execution_date, last_execution_date,
                                                             settings.get('$.server.fmv.initial_model_price_count')))
        model_prices = create_fmv_json(figi_row, self.__generator, self.__model, get_ordinals(), size,
                                       initial_model_execution_dates, initial_model_execution_dates)
        trade_list = self.__create_trade_list_json(trades)
        self.__wrapped_websocket.send(json.dumps({
            "trade": trade_list,
            "model_price": model_prices
        }))

    def __recv_handler(self, msg_text):
        msg = json.loads(msg_text)
        if 'token' in msg:
            self.__recv_token(msg)
        if 'health_check' in msg:
            return self.__recv_health_check()
        elif 'fmv_stream' in msg:
            # reset state
            self.__reset()
            atts = msg['fmv_stream']
            # parse input
            if not self.__check_for_required_attributes(atts):
                return
            size = self.__parse_size(atts)
            if size is None:
                return
            trade_count = self.__parse_trade_count(atts)
            if trade_count is None:
                return
            figi = self.__parse_figi(atts)
            if figi is None:
                return
            figi_trace = self.__parse_figi_trace(figi, trade_count)
            if figi_trace is None:
                return

            if settings.get("$.server.fmv.authorize_websocket") and not self.__authorize_websocket(atts):
                return

            figi_row = figi_trace.slice(figi_trace.num_rows - 1)
            self.start_tasks(figi_row, size)
            self.send_initial_trades(figi_row, figi_trace, size, trade_count)

    #####################################
    # End of recv_handler-related methods
    #####################################

    def set_fmv_stream_reset(self, topic, serve_model_task, heartbeat_task):
        def fmv_stream_reset():
            try:
                pub.unsubscribe(self.__new_trades_listener, topic)
            except:
                self.__exception('fmv_stream_reset: error unsubscribing')
            try:
                serve_model_task.cancel()
            except:
                self.__exception('fmv_stream_reset: error cancelling serve_model_task')
            try:
                heartbeat_task.cancel()
            except:
                self.__exception('fmv_stream_reset: error cancelling heartbeat_task')

        self.__reset = fmv_stream_reset

    async def __serve_model(self, figi_row, quantity):
        while True:
            await asyncio.sleep(MODEL_INTERVAL)
            report_date_series = pa.array([time.time_ns()])
            model_prices = create_fmv_json(figi_row, self.__generator, self.__model,
                                        get_ordinals(), quantity,
                                        report_date_series, report_date_series)
            self.__send_json({"model_price": model_prices})

    def __new_trades_listener(self, new_trades):
        self.__send_json({"trade": self.__create_trade_list_json(new_trades)})

    async def __heartbeat(self):
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            response = requests.post(
                                CHECKWEBSOCKET_URL,
                                headers={'Authorization': self.__token or ''},
                                json={'websocketid': self.__websocketid}).json()
            self.__debug(f'heartbeat response: {response}')
            if "result" not in response or response["result"] != "active":
                self.__send_msg('deactivated')
                self.__reset()

    def __create_trade_list_json(self, trades: pa.Table):
        X_b, _ = self.__generator.generate_batch(trades)
        Y_b_hat = self.__model.evaluate_batch(X_b)
        model_prices = Y_b_hat[:, self.__generator.get_rfq_label_index('price')]
        result = []
        for i in range(trades.shape[0]):
            side = self.__side_num_to_side[trades['side'][i].as_py()]
            buy_sell = self.__buy_sell_num_to_buy_sell[trades['buy_sell'][i].as_py()]
            if side != 'D':
                side = buy_sell
            result.append({
                "record_num": next(self.__record_num), # <- TODO: FIX THIS, this is a workaround
                "report_date": epoch_ns_to_iso_str(trades['report_date'][i].as_py()),
                "execution_date": epoch_ns_to_iso_str(trades['execution_date'][i].as_py()),
                "side": side,
                "quantity": int(trades['quantity'][i].as_py()),
                "price": float(trades['price'][i].as_py()),
                "yield": trades['yield'][i].as_py(),
                "model_price": model_prices[i]
            })
        return result

#####################################
# Begin main function related functions
#####################################
def _process_new_trade_messages(base64_messages):
    new_trades = parse_messages(base64_messages, 'multicast')
    if new_trades and len(new_trades):
        return create_process_trades()(map_ordinals(new_trades, get_ordinals()))

async def initialize_server_data_and_model():
    # the server will await this future so setting it shuts everything down
    exit_if_result_set = asyncio.Future()
    logger.info(f'Environment: {ENVIRONMENT}')
    profile = create_profiler('server')
    with profile('get initial trades'):
        trades = get_initial_trades()
    min_report_date = pc.min(trades['report_date']).as_py()
    max_report_date = pc.max(trades['report_date']).as_py()
    logger.info(
        f"trades date range: {epoch_ns_to_iso_str(min_report_date)} - {epoch_ns_to_iso_str(max_report_date)}")
    split = TraceModel.get_split_index(trades, 'multicast_data_split')
    trades = trades[split:]
    logger.info(
        f"reduced trades date range: {epoch_ns_to_iso_str(pc.min(trades['report_date']).as_py())} - {epoch_ns_to_iso_str(pc.max(trades['report_date']).as_py())}")
    with profile('create TradeHistory'):
        trade_history = TradeHistory()
        trade_history.append(trades)
    with profile('create generator'):
        generator = TraceDataGenerator(trade_history, should_shuffle=False)
    with profile('load_model'):
        model_name = settings.get('$.server.fmv.model')
        model_class, model_settings = get_model_class(model_name)
        if 'overrides' in model_settings:
            settings.override(model_settings['overrides'])
        model = model_class(model_name, model_settings, generator, generator, generator)
        model.create()
    return generator, trade_history, model, exit_if_result_set, trades

def setup_primary_buffer_and_handler():
    primary_buffer = set()

    # note: this gets called on a different thread
    def primary_handler(message):
        data_logger.debug(message)
        primary_buffer.add(message)

    startup_buffer = set()

    # note: this gets called on a different thread
    def startup_handler(message):
        data_logger.debug('adding to startup_buffer')
        startup_buffer.add(message)

    pub.subscribe(primary_handler, FINRA_BTDS_MULTICAST_TOPIC)
    pub.subscribe(startup_handler, FINRA_BTDS_MULTICAST_TOPIC)

    pub.subscribe(primary_handler, FINRA_144A_MULTICAST_TOPIC)
    pub.subscribe(startup_handler, FINRA_144A_MULTICAST_TOPIC)

    return primary_buffer, startup_buffer, primary_handler, startup_handler

async def data_loop(primary_buffer, trade_history):
    while True:
        await asyncio.sleep(DATA_LOOP_INTERVAL)
        global _health_stats__last_data_loop_run
        _health_stats__last_data_loop_run = time.time_ns()
        try:
            if primary_buffer:
                global _health_stats__last_data_loop_batch
                _health_stats__last_data_loop_batch = time.time_ns()
                batch = primary_buffer.copy()
                primary_buffer -= batch # note: use -= since adding happens on a different thread
                data_logger.info(f'data_loop batch of {len(batch)}')
                data_logger.debug(batch)
                new_trades = _process_new_trade_messages(batch)
                if new_trades and len(new_trades):
                    data_logger.debug(f"""Table\n{new_trades.select(
                        ['figi', 'cusip', 'report_date', 'execution_date', 'side', 'price', 'yield', 'quantity'])}""")
                    trade_history.append(new_trades)
                    unique_figis = new_trades['figi'].unique()
                    # TODO: these are the individual new trades but they might have been combined with
                    # each other or with previous trades.  We need to broadcast the updated combined trades.
                    for figi in unique_figis:
                        figi_subset = pc.filter(new_trades, pc.equal(new_trades['figi'],figi))
                        data_logger.debug(f"""publishing to {figi}\n {figi_subset.select(
                            ['figi', 'report_date', 'execution_date', 'side', 'price', 'yield', 'quantity'])}""")
                        pub.sendMessage(str(figi), new_trades=figi_subset)
        except:
            data_logger.exception('error in data_loop')
            logger.exception('error in data_loop')


async def get_memorydb_contents_synchronously():
    logger.info("start reading from memorydb")
    data_logger.info("start reading from memorydb")
    start_index = 0
    memorydb_inbound = set()
    while True:
        data = memory_db.zrange(FINRA_CACHE_KEY, start_index, start_index + MEMORYDB_BATCH_SIZE - 1)
        memorydb_inbound |= set(data)
        data_logger.info(f'retrieved memorydb batch, memorydb_inbound len: {len(memorydb_inbound)}')
        start_index += len(data)
        # give up control momentarily to move any messages out of the socket buffer
        await asyncio.sleep(0)
        if len(data) < MEMORYDB_BATCH_SIZE:
            last_few = parse_messages(data[-100:], 'multicast')
            if last_few and len(last_few):
                data_logger.debug(f"""last few MemoryDB items\n{last_few}""")
            break
    logger.info("finish reading from memorydb")
    data_logger.info("finish reading from memorydb")
    return memorydb_inbound

async def recover_from_memory_db(trade_history, startup_buffer, startup_handler, trades):
    if memory_db.zrange(FINRA_CACHE_KEY, 0, 0):
        # MemoryDB contains data so we are starting after trading has started for the day
        # sleep to allow any in-flight messages to propogate to MemoryDB
        logger.info(f"sleeping {MEMORYDB_PROPOGATION_WAIT_TIME}s for in-flight MemoryDB propogation")
        await asyncio.sleep(MEMORYDB_PROPOGATION_WAIT_TIME)
        memorydb_inbound = await get_memorydb_contents_synchronously()
        # give up control momentarily to move any messages out of the socket buffer
        await asyncio.sleep(0)
        logger.info(f'memorydb_inbound len: {len(memorydb_inbound)}')
        # remove any messages that we've already seen come in on the multicast
        memorydb_inbound -= startup_buffer
        logger.info(f'memorydb_inbound len after removing startup_buffer: {len(memorydb_inbound)}')
        new_trades = _process_new_trade_messages(memorydb_inbound)
        if new_trades and len(new_trades):
            # only append messages reported after the end of trades
            logger.info(f'new trades from memorydb_inbound: {len(new_trades)}')
            logger.info(
                f"memorydb_inbound date range: {epoch_ns_to_iso_str(pc.min(new_trades['report_date']).as_py())} - {epoch_ns_to_iso_str(pc.max(new_trades['report_date']).as_py())}")
            new_trades = new_trades.filter(pc.greater(new_trades['report_date'], pc.max(trades['report_date'])))
            if new_trades and len(new_trades):
                logger.info(f'memorydb_inbound len after filtering: {len(new_trades)}')
                logger.info(
                    f"memorydb_inbound date range after filtering: {epoch_ns_to_iso_str(pc.min(new_trades['report_date']).as_py())} - {epoch_ns_to_iso_str(pc.max(new_trades['report_date']).as_py())}")
                trade_history.append(new_trades)
                logger.info(f'total memorydb_inbound items added to trade_history: {len(new_trades)}')
            else:
                logger.info('memorydb_inbound empty after filtering')
        else:
            logger.info('memorydb_inbound empty')
    else:
        logger.info(f'not sleeping: MemoryDB empty')
    pub.unsubscribe(startup_handler, FINRA_BTDS_MULTICAST_TOPIC)
    pub.unsubscribe(startup_handler, FINRA_144A_MULTICAST_TOPIC)
    startup_buffer.clear()


async def launch_websocket_listener(generator, model,
                                    data_loop_task, exit_if_result_set, memorydb_task_cleanup):
    async def handler(websocket, _):
        try:
            websocketid = str(uuid.uuid4())
            logger.info(f'{websocketid}: starting handler')
            fmv_handler = FMV_Handler(websocket, websocketid, generator, model)
            await fmv_handler.serve()
            logger.info(f'{websocketid}: exiting handler')
            logger.debug(f'active tasks: {len(asyncio.all_tasks())}')
        except:
            logger.exception('error in handler')

    logger.info(f"About to start server listening to localhost:{PORT}")
    global _health_stats__server_started
    _health_stats__server_started = time.time_ns()
    # use server as a context manager, server shuts down when context exits
    async with serve(handler, 'localhost', PORT):
        await exit_if_result_set
    try:
        data_loop_task.cancel()
        memorydb_task_cleanup()
    except:
        logger.exception('error during cleanup')


async def main():
    generator, trade_history, model, exit_if_result_set, trades = await initialize_server_data_and_model()
    primary_buffer, startup_buffer, primary_handler, startup_handler = setup_primary_buffer_and_handler()
    memorydb_task_cleanup = memory_db.create_send_to_memorydb_task(data_logger)
    # start btds multicast listener
    create_listen_thread(listen_btds)
    # start 144a multicast listener
    create_listen_thread(listen_144a)
    data_loop_task = asyncio.create_task(data_loop(primary_buffer, trade_history))
    await recover_from_memory_db(trade_history, startup_buffer, startup_handler, trades)
    await launch_websocket_listener(generator, model, data_loop_task,
                                    exit_if_result_set, memorydb_task_cleanup)
    logger.info('exiting main')


if __name__ == '__main__':
    asyncio.run(main())
