import asyncio
import base64
from datetime import datetime
import importlib
from itertools import zip_longest
import json
from math import floor
import os
import re
import socket
import struct
import tempfile
from typing import Callable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import tensorflow as tf
import zipfile

from data_pipeline.helpers import map_ordinals
import finra
from generator import TraceDataGenerator
from grouped_history import combine_trace
from helpers import epoch_ns_to_iso_str
import load_data
from model import TraceModel
import s3
import settings
from trade_history import TradeHistory
from train_evaluate import get_model_class

# list of bonds used for the integration test data and integration tests
BONDS = [
    # Top 20 In Investment Grade Spreadsheet from James
    # CUSIP        FIGI              ISIN          TICKER
    ('126650CX6', 'BBG00K85WG22'), # US126650CX62, CVS 4.3 03/25/28
    ('06051GFH7', 'BBG00717KNY7'), # US06051GFH74, BAC 4.2 08/26/24 MTN
    ('20030NCT6', 'BBG00M53S4K8'), # US20030NCT63, CMCSA 4.15 10/15/28
    ('958102AM7', 'BBG00JXGN2L3'), # US958102AM75, WDC 4.75 02/15/26
    ('38141GVM3', 'BBG0062KHLH6'), # US38141GVM31, GS 4 03/03/24
    ('00287YAQ2', 'BBG008MXGMD5'), # US00287YAQ26, ABBV 3.6 05/14/25
    ('92826CAD4', 'BBG00BMBZ6V0'), # US92826CAD48, V 3.15 12/14/25
    ('172967KA8', 'BBG00B2XQKX1'), # US172967KA87, C 4.45 09/29/27
    ('46625HRV4', 'BBG00DBJW2Y0'), # US46625HRV41, JPM 2.95 10/01/26
    ('126650CZ1', 'BBG00K85WH93'), # US126650CZ11, CVS 5.05 03/25/48
    ('06051GGA1', 'BBG00F0TZ3D3'), # US06051GGA13, BAC 3.248 10/21/27 MTN
    ('94974BFY1', 'BBG006KDBLF0'), # US94974BFY11, WFC 4.1 06/03/26 MTN
    ('594918BR4', 'BBG00DJ1B6J2'), # US594918BR43, MSFT 2.4 08/08/26
    ('68389XBN4', 'BBG00J5HRSQ6'), # US68389XBN49, ORCL 3.25 11/15/27
    ('61746BDQ6', 'BBG006DDT5Z2'), # US61746BDQ68, MS 3.875 04/29/24 F
    ('38141GVR2', 'BBG00B6VJFW1'), # US38141GVR28, GS 4.25 10/21/25
    ('10922NAC7', 'BBG00KH690L0'), # US10922NAC74, BHF 3.7 06/22/27
    ('92343VCR3', 'BBG007DM0XF2'), # US92343VCR33, VZ 3.5 11/01/24
    ('10922NAF0', 'BBG00KGGX0D1'), # US10922NAF06, BHF 4.7 06/22/47 (no rating so being filtered - Mar 18, 2023)
    ('94974BGH7', 'BBG0083CSQF8'), # US94974BGH78, WFC 3 02/19/25 MTN

    # Top 20 In High Yield Spreadsheet from James
    # CUSIP        FIGI              ISIN          TICKER
    ('345370CR9', 'BBG00FGQXY82'), # US345370CR99, F 4.346 12/08/26
    ('382550BF7', 'BBG00CW3HJ74'), # US382550BF73, GT 5 05/31/26
    ('075896AB6', 'BBG006T3THG2'), # US075896AB63, BBBY 4.915 08/01/34
    ('345370CS7', 'BBG00FGQZFW4'), # US345370CS72, F 5.291 12/08/46
    ('345370CA6', 'BBG00004J1S7'), # US345370CA64, F 7.45 07/16/31
    ('345370CQ1', 'BBG003S95DC6'), # US345370CQ17, F 4.75 01/15/43
    ('343412AC6', 'BBG007K8PLT9'), # US343412AC69, FLR 3.5 12/15/24
    ('81180WAN1', 'BBG007LT1TB0'), # US81180WAN11, STX 5.75 12/01/34
    ('29078EAA3', 'BBG0000CFMW2'), # US29078EAA38, LUMN 7.995 06/01/36
    ('075896AC4', 'BBG006T3TKF6'), # US075896AC47, BBBY 5.165 08/01/44
    ('724479AJ9', 'BBG0063HM4T1'), # US724479AJ97, PBI 4.625 03/15/24
    ('343412AF9', 'BBG00LSDJQ97'), # US343412AF90, FLR 4.25 09/15/28
    ('655664AR1', 'BBG006G2HB38'), # US655664AR15, JWN 5 01/15/44
    ('382550BG5', 'BBG00G4X4X91'), # US382550BG56, GT 4.875 03/15/27
    ('156700AM8', 'BBG0000JXSP2'), # US156700AM80, LUMN 7.6 09/15/39 P
    ('651229AW6', 'BBG00CGXVTM1'), # US651229AW64, NWL 4.2 04/01/26
    ('075896AA8', 'BBG006T3TB02'), # US075896AA80, BBBY 3.749 08/01/24
    ('25470XAY1', 'BBG00DR3Y7G5'), # US25470XAY13, DISH 7.75 07/01/26
    ('25470XAW5', 'BBG007TJ51K1'), # US25470XAW56, DISH 5.875 11/15/24
    ('984121CB7', 'BBG0000MMJZ6'), # US984121CB79, XRXCRP 6.75 12/15/39

    # Top 20 from Finra 144a
    # CUSIP        FIGI              ISIN          TICKER
    ('893647BE6', 'BBG00N6WRN48'), # US893647BE67, TDG 6.25 03/15/26 REGS
    ('37045XDB9', 'BBG00X7K7WR8'), # US37045XDB91, GM V5.7 PERP C
    ('097751BT7', 'BBG00NHJQM02'), # US097751BT78, BBDBCN 7.875 04/15/27 REGS
    ('451102BZ9', 'BBG00RR4Z6Q2'), # US451102BZ91, IEP 5.25 05/15/27
    ('25277LAA4', 'BBG00PNVYH69'), # US25277LAA44, DSPORT 5.375 08/15/26 REGS
    ('855030AN2', 'BBG00NS7D7Y6'), # US855030AN20, SPLS 7.5 04/15/26 REGS       no trades in parquet as of Mar 22, 2023
    ('1248EPCD3', 'BBG00Q9KZKL2'), # US1248EPCD32, CHTR 4.75 03/01/30 REGS
    ('69867DAC2', 'BBG00NJLY6L8'), # US69867DAC20, POWSOL 8.5 05/15/27 REGS
    ('25277LAC0', 'BBG00PNVYRF7'), # US25277LAC00, DSPORT 6.625 08/15/27 REGS
    ('1248EPCE1', 'BBG00RMHNL42'), # US1248EPCE15, CHTR 4.5 08/15/30 REGS
    ('023771S58', 'BBG00VNJPZS9'), # US023771S586, AAL 11.75 07/15/25 REGS
    ('00165CAP9', 'BBG00WFDN400'), # US00165CAP95, AMC 10 06/15/26 REGS
    ('143658BN1', 'BBG00Z716PP3'), # US143658BN13, CCL 5.75 03/01/27 REGS
    ('126307BA4', 'BBG00PM8GY00'), # US126307BA42, CSCHLD 5.75 01/15/30 REGS
    ('1248EPBX0', 'BBG00HCXW509'), # US1248EPBX05, CHTR 5 02/01/28 REGS
    ('1248EPBT9', 'BBG00FRYKHD8'), # US1248EPBT92, CHTR 5.125 05/01/27 REGS
    ('55903VAQ6', 'BBG015XRZ2J8'), # US55903VAQ68, WBD 5.141 03/15/52 REGS
    ('20338QAA1', 'BBG00N70FTG4'), # US20338QAA13, COMM 8.25 03/01/27 REGS
    ('69888XAA7', 'BBG00NLCY4K1'), # US69888XAA72, ENDP 7.5 04/01/27 REGS
    ('67054KAA7', 'BBG00CMXXZW7'), # US67054KAA79, SFRFP 7.375 05/01/26 REGS

    # Top 7 Unknown (highly traded but not in spreadsheets from James)
    # CUSIP        FIGI              TICKER
    ('037833AK6', 'BBG004HST063'), # AAPL 2.4 05/03/23
    ('88167AAE1', 'BBG00DDM0BW8'), # TEVA 3.15 10/01/26
    ('06051GEU9', 'BBG003T485S3'), # BAC 3.3 01/11/23 GMTN (matured)
    ('912909AN8', 'BBG00K9VLX29'), # X 6.25 03/15/26
    ('71654QCC4', 'BBG00FM535P8'), # PEMEX 6.75 09/21/47
    ('71654QCG5', 'BBG00K5W8536'), # PEMEX 6.5 03/13/27
    ('912909AM0', 'BBG00HBW6JJ1'), # X 6.875 08/15/25
]

# message template for generating mock finra messages
MESSAGE_TEMPLATE = {
            'MESSAGE_CATEGORY': 'T',
            'MESSAGE_TYPE': 'M',
            'RESERVED': None,
            'RETRANSMISSION_REQUESTER': 'O',
            'MESSAGE_SEQUENCE_NUMBER': 1,
            'MARKET_CENTER_ORIGINATOR_ID': 'O',
            'DATE_TIME': datetime(2023, 3, 3, 8, 1, 1),
            'SYMBOL': 'CVS4608028',
            'CUSIP': '126650CX6',
            'BSYM': 'BBG00K85WG22',
            'SUB_PRODUCT_TYPE': 'CORP',
            'ORIGINAL_DISSEMINATION_DATE': None,
            'QUANTITY_INDICATOR': 'A',
            'QUANTITY': '00000100000.00',
            'PRICE': 97.066,
            'REMUNERATION': 'N',
            'SPECIAL_PRICE_INDICATOR': None,
            'SIDE': 'S',
            'AS_OF_INDICATOR': None,
            'EXECUTION_DATE_TIME': datetime(2023, 3, 3, 8, 1, 0),
            'FUTURE_USE': None,
            'SALE_CONDITION_3': None,
            'SALE_CONDITION_4': None,
            'SETTLEMENT_DATE': '20230303',
            'YIELD_DIRECTION': None,
            'YIELD': 4.967028,
            'WHEN_ISSUED_INDICATOR': None,
            'REPORTING_PARTY_TYPE': 'D',
            'CONTRA_PARTY_TYPE': 'C',
            'ATS_INDICATOR': None,
            'CHANGE_INDICATOR': '0'}
# list of possible quantities to use for generated messages
POSSIBLE_QUANTITIES = [
            ('00000001000.00', 'A'),
            ('00000010000.00', 'A'),
            ('00000025000.00', 'A'),
            ('00000050000.00', 'A'),
            ('00000075000.00', 'A'),
            ('00000100000.00', 'A'),
            ('00000250000.00', 'A'),
            ('00000500000.00', 'A'),
            ('00000750000.00', 'A'),
            ('1MM+', 'E'),
            ('5MM+', 'E')]
# list of possible side/contra party pairs to use for generated message pairs
POSSIBLE_SIDE_CONTRA_PARTY_TYPE = [
            (('B','C'), ('S','A')),
            (('B','A'), ('S','C')),
            (('B','A'), ('S','C')),
            (('B','A'), ('S','D')),
            (('B','A'), ('S','A'))]

class CleanEnvironment:
    """Creates a clean execution environment.

    Clears the Keras backend session, seeds all RNG, and enables Keras determinism.

    Creates temp directory and sets 's3-cache', 'data', and 'models' to save to the temp directory (can be changed with create_overrides).

    Calls create_overrides with the temp directory to get settings to override, reloads settings, and applies overrides.

    Can be used as a context manager:
        with CleanEnvironment():
            ...

    or by manually calling cleanup:
        env = CleanEnvironment()
        ...
        env.cleanup()

    Keyword arguments:
    create_overrides -- function that accepts the temporary directory and returns the settings to override
    seed -- seed value for random number generation (default = 1)
    """
    def __init__(self, create_overrides: Callable[[str], dict]=lambda temp_dir: {}, seed: int=1):
        # create a temporary directory
        self.tempdir = tempfile.TemporaryDirectory()
        print(f'creating clean_environment with temporary directory: {self.tempdir.name}')
        # Clear the Keras backend session.
        # Keras assigns a globally unique identifier to every layer as it is created.
        # This resets Keras so layers created in the same order will get the same unique identifier.
        tf.keras.backend.clear_session()
        # This sets the seed of random, np.random, tf.random, and the Keras backend seed generator to the same seed value
        tf.keras.utils.set_random_seed(seed)
        # tell Tensorflow to be deterministic
        tf.config.experimental.enable_op_determinism()
        # reload the settings module
        importlib.reload(settings)
        # anything saved in self.tempdir will get cleared when self.tempdir gets deleted
        settings.override({
            "$.s3_cache_path": os.path.join(self.tempdir.name, 's3-cache'),
            "$.local_data_path": os.path.join(self.tempdir.name, 'data'),
            "$.keras_models.local_model_dir": os.path.join(self.tempdir.name, 'models'),
            # add the results of calling create_overrides last so the above settings can be overridden
            **create_overrides(self.tempdir.name)})
        # reload the load_data module to clear any cached data
        importlib.reload(load_data)
    def __repr__(self):
        return "<{} {!r}>".format(self.__class__.__name__, self.tempdir)
    def __enter__(self):
        pass
    def __exit__(self, exc, value, tb):
        self.cleanup()
    def cleanup(self):
        # clean up the tempdir
        self.tempdir.cleanup()

# regex and utility function to map extracted file names to their full paths
expanded_re = re.compile(r'^.*/expanded/(.*)$')
def _create_extracted_file_to_full_path_map(input):
    return {expanded_re.match(f).group(1): f for f in (input if isinstance(input, list) else [input])}

def compare_local_files(file_map_1: dict, file_map_2: dict):
    """Accepts two dictionaries mapping filename(s) to their full path on disk returns whether all files are equivalent
    
    example:
    result = compare_local_files(
        { 'my_file': '/full/path/to/__first__/my_file',
          'subdir/another': '/full/path/to/__first__/can/be/anywhere/another' },
        { 'my_file': '/full/path/to/__second__/my_file',
          'subdir/another': 'full/path/to/__second__/somewhere/another' })
    print(result) # True if both versions of the files pointed to by 'my_file' and 'subdir/another' are equivalent, otherwise False

    example:
    result = compare_local_files(
        { 'my_file': '/full/path/to/__first__/my_file',
          'another': '/full/path/to/__first__/another' },
        { 'my_file': '/full/path/to/__second__/my_file')
    print(result) # False (filename sets are different)
    """
    # create a set containing the filename(s) from each dictionary and compare the sets
    if {k for k in file_map_1.keys()} != {k for k in file_map_2.keys()}:
        return False
    # filename sets are equivalent so compare the binary of each file
    for k in file_map_1.keys():
        print(f'comparing {k}')
        # open each file in read binary mode
        with open(file_map_1[k], 'rb') as f1, open(file_map_2[k], 'rb') as f2:
            # compare each "line" in the binary, use fillvalue=None in case one has more lines than the other
            for line1, line2 in zip_longest(f1, f2, fillvalue=None):
                if line1 != line2:
                    print('  different')
                    return False
        print('  equivalent')
    return True

def compare_s3_files(s3_url_1, s3_url_2):
    """Compares two zip files stored in S3, returns True if they are equivalent, otherwise False.
    """
    print(f'comparing {s3_url_1} and {s3_url_2}')
    # create a map of the filename(s) within the zip to its full extracted path for both extracted zip files
    file_map_1 = _create_extracted_file_to_full_path_map(s3.get_object(s3_url_1))
    file_map_2 = _create_extracted_file_to_full_path_map(s3.get_object(s3_url_2))
    # compare
    result = compare_local_files(file_map_1, file_map_2)
    print(f"{s3_url_1} and {s3_url_2} are {'' if result else 'NOT '}equivalent")
    return result

def compare_local_to_s3(local_path, s3_url):
    """Compares two zip files, one local and one stored in S3.  Returns True if they are equivalent, otherwise False.
    """
    print(f'comparing {local_path} and {s3_url}')
    # use a temporary directory
    with tempfile.TemporaryDirectory() as temp_directory:
        with zipfile.ZipFile(local_path, 'r') as zf:
            # retrieve the list of files in the local zip file
            namelist = zf.namelist()
            # extract into the temporary directory
            zf.extractall(temp_directory)
        # create a map of the filename(s) within the zip to its full extracted path for local zip file
        file_map_1 = {file: os.path.join(temp_directory, file) for file in namelist if not file.endswith('/')}
        # create a map of the filename(s) within the zip to its full extracted path for S3 zip file
        file_map_2 = _create_extracted_file_to_full_path_map(s3.get_object(s3_url))
        # compare
        result = compare_local_files(file_map_1, file_map_2)
        print(f"{local_path} and {s3_url} are {'' if result else 'NOT '}equivalent")
        return result

def _serialize_finra_field(field, data):
    """Serialize a parsed FINRA data field back to a string version of its original representation"""
    dtype = finra.field_sizes[field]
    if isinstance(dtype, int):
        length = dtype
    else:
        length = finra.finra_dtypes[dtype]
    if data is None:
        return ' ' * length
    if isinstance(dtype, int):
        return data.ljust(length, ' ')
    if dtype == 'SHORT_INT' or dtype == 'LONG':
        return f'{data}'[0:length].rjust(length, '0')
    if dtype == 'FLOAT' or dtype == 'BIG_FLOAT':
        return f'{data:.6f}'.rjust(length, '0')
    if dtype == 'DATE':
        return data
    if dtype == 'DATE_TIME':
        return datetime.strftime(data, '%Y%m%d%H%M%S')
    raise NotImplementedError(f'Unrecognized dtype \"{dtype}\"')

def serialize_finra_message(message):
    """Accepts an object representing a parsed FINRA message and returns the original serialized FINRA binary as a base64 encoded ascii string.
    """
    result = bytes.fromhex('01')
    for field in finra.header_fields:
        result += _serialize_finra_field(field, message[field]).encode('ascii')
    message_spec = finra.message_categories[message['MESSAGE_CATEGORY']][message['MESSAGE_TYPE']]
    for message_type_field_group in message_spec:
        for field in finra.field_groups[message_type_field_group]:
            result += _serialize_finra_field(field, message[field]).encode('ascii')
    result += bytes.fromhex('03')
    return base64.b64encode(result).decode('ascii')

class FinraBroadcastMock():

    def __init__(self, multicast_group_btds, multicast_port_btds, multicast_group_144a, multicast_port_144a, \
                 seed: int=1, sequence_number=1, timestamp: float=datetime(2023, 3, 3, 8, 1, 0).timestamp()):
        self.__multicast_group_btds = multicast_group_btds
        self.__multicast_port_btds = multicast_port_btds
        self.__multicast_group_144a = multicast_group_144a
        self.__multicast_port_144a = multicast_port_144a
        # use a seeded numpy random generator so we are deterministically random and isolated from all other rng
        self.__np_rng = np.random.default_rng(seed)
        self.__sequence_number = sequence_number
        self.__timestamp = timestamp
        # load initial trades from parquet
        self.__trades = load_data.get_initial_trades()
        # use the same split and columns as the server
        split = TraceModel.get_split_index(self.__trades, 'multicast_data_split')
        self.__trades = self.__trades[split:]
        self.__trades = self.__trades.select(sorted(self.__trades.column_names))
        self.__trades = self.__trades.sort_by('report_date')
        # create a generator and model in the same way as the server
        self.__trade_history = TradeHistory()
        self.__trade_history.append(self.__trades)
        self.__generator = TraceDataGenerator(self.__trade_history, should_shuffle=False)
        model_name = settings.get('$.server.fmv.model')
        model_class, model_settings = get_model_class(model_name)
        self.__model = model_class(model_name, model_settings, self.__generator, self.__generator, self.__generator)
        self.__model.create()
        # create cache used to improve performance while generating messages and initialize it with integration test bonds
        self.__cache = {}
        for bond in BONDS:
            self.__add_figi_to_cache_if_needed(bond[1])
        self.__message_buffer = set()

    def __add_figi_to_cache_if_needed(self, figi: str):
        # add the figi to the cache if not already there
        if figi not in self.__cache:
            self.__cache[figi] = {}
            # filter trades by the figi
            # note: the cache exists so we only have to do this once per figi
            figi_trades = self.__trades.filter(pc.equal(self.__trades['figi'], pc.index(load_data.get_ordinals()['figi'], figi)))
            if figi_trades:
                # if trades exist for this figi then initialize the cache with the actual trade values
                self.__cache[figi]['first_price'] = figi_trades['price'][0].as_py()
                self.__cache[figi]['last_price'] = figi_trades['price'][len(figi_trades) - 1].as_py()
                self.__cache[figi]['yield'] = figi_trades['yield'][len(figi_trades) - 1].as_py() + self.__np_rng.normal(0, 0.5)
            else:
                # if no trades exist for this figi then initialize the cache with auto-generated values
                self.__cache[figi]['first_price'] = 100 + self.__np_rng.normal(0, 5)
                self.__cache[figi]['last_price'] = 100 + self.__np_rng.normal(0, 5)
                self.__cache[figi]['yield'] = 5 + self.__np_rng.normal(0, 1)

    async def __create_messages(self, figi: str=''):
        if figi:
            # if we are passed a figi then retrieve the associated cusip
            cusip = [c for c,f in BONDS if f == figi][0]
        else:
            # otherwise pick a random bond
            cusip, figi = BONDS[self.__np_rng.integers(0, len(BONDS))]
        # add the figi to the cache if needed
        self.__add_figi_to_cache_if_needed(figi)
        # select the side/contra party to be used for this message pair
        side_contra_party_type = POSSIBLE_SIDE_CONTRA_PARTY_TYPE[self.__np_rng.integers(0, len(POSSIBLE_SIDE_CONTRA_PARTY_TYPE))]
        # copy the message template into our message
        message = {**MESSAGE_TEMPLATE}
        # set the time the message was sent from FINRA
        message['DATE_TIME'] = datetime.fromtimestamp(self.__timestamp)
        # trades must be reported to FINRA within 15 min, this represents the delay between when the trade happened and when it was reported to FINRA
        report_delay = 1 + self.__np_rng.random() * 15 * 60
        message['EXECUTION_DATE_TIME'] = datetime.fromtimestamp(self.__timestamp - report_delay)
        # we don't currently use this so just mock it for now
        message['SYMBOL'] = 'mocked'
        # set the cusip and figi
        message['CUSIP'] = cusip
        message['BSYM'] = figi
        # select and set the quantity to be used for this message pair
        quantity, quantity_indicator = POSSIBLE_QUANTITIES[self.__np_rng.integers(0, len(POSSIBLE_QUANTITIES))]
        message['QUANTITY_INDICATOR'] = quantity_indicator
        message['QUANTITY'] = quantity
        # set the price using normal distribution noise that tends toward the original price, floor of 10
        message['PRICE'] = max(10.0, self.__cache[figi]['last_price'] + \
                        self.__np_rng.normal(0, 0.5) + \
                        self.__np_rng.normal(1 if self.__cache[figi]['last_price'] < self.__cache[figi]['first_price'] else -1, 0.5))
        # update the cache with the price
        self.__cache[figi]['last_price'] = message['PRICE']
        # for now just set the yield using normal distribution noise that tends toward 5, floor of 1...improve this to be more representative if needed
        message['YIELD'] = max(1.0, self.__cache[figi]['yield'] + \
                        self.__np_rng.normal(0, 0.5) + \
                        self.__np_rng.normal(1 if self.__cache[figi]['yield'] < 5 else -1, 0.5))
        # update the cache with the yield
        self.__cache[figi]['yield'] = message['YIELD']
        # create the result array
        result = []
        # helper function to add a message to the result
        def add_message(m):
            # set the sequence number and increment        
            m['MESSAGE_SEQUENCE_NUMBER'] = self.__sequence_number
            self.__sequence_number += 1 + self.__np_rng.integers(0, 5)
            result.append((
                            # time the message was received over the multicast, add a random amount of time to simulate network latency
                            floor((m['DATE_TIME'].timestamp() + self.__np_rng.random()*10) * 1_000_000_000),
                            # serialized message received
                            serialize_finra_message(m)))
        # number of messages representing the same trade that will need to be deduplicated, usually 1 but sometimes up to 5
        duplicate_count = max(1, self.__np_rng.integers(0, 20) - 14)
        # set the side/contra party for the first message(s)
        message['SIDE'] = side_contra_party_type[0][0]
        message['CONTRA_PARTY_TYPE'] = side_contra_party_type[0][1]
        for _ in range(duplicate_count):
            add_message(message)
        # set the side/contra party for the second message(s)
        message['SIDE'] = side_contra_party_type[1][0]
        message['CONTRA_PARTY_TYPE'] = side_contra_party_type[1][1]
        for _ in range(duplicate_count):
            add_message(message)
        # increment the timestamp
        self.__timestamp += max(1, self.__np_rng.normal(1, 1)) + max(0, self.__np_rng.normal(-30, 20))
        return result

    async def __create_trades(self, count: int=1, figi: str='', broadcast: bool=True):
        # create the messages
        messages = [t for _ in range(count) for t in await self.__create_messages(figi=figi)]
        # add the new messages to a buffer...we don't need to process them until we need to create the expected trades
        for m in [p[1] for p in messages]:
            self.__message_buffer.add(m)
        if broadcast:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP) as sock:
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, struct.pack('b', 1))
                for i, (_, m) in enumerate(messages):
                    # broadcast the message either over the btds or 144a multicast at random
                    if self.__np_rng.choice([True, False]):
                        sock.sendto(base64.b64decode(m), (self.__multicast_group_btds, self.__multicast_port_btds))
                    else:
                        sock.sendto(base64.b64decode(m), (self.__multicast_group_144a, self.__multicast_port_144a))
                    # throttle sends to not flood the network with UDP packets
                    # only remove when specifically (and temporarily) testing burst capacity
                    if i % 10 == 9:
                        await asyncio.sleep(0.1)
        return messages

    async def create_trades(self, count: int=1, figi: str=''):
        return await self.__create_trades(count=count, figi=figi, broadcast=False)

    async def broadcast_trades(self, count: int=1, figi: str=''):
        return await self.__create_trades(count=count, figi=figi, broadcast=True)

    def get_expected_trades(self, figi: str, count: int):
        # process any messages in the buffer
        buffer = self.__message_buffer.copy()
        self.__message_buffer -= buffer
        if buffer:
            # add to trades
            new_trades = finra.parse_messages([m for m in buffer], 'multicast')
            if new_trades and len(new_trades):
                new_trades = load_data.create_process_trades()(map_ordinals(new_trades, load_data.get_ordinals()))
            if new_trades and len(new_trades):
                # append to the trade_history for model pricing...
                self.__trade_history.append(new_trades)
                # ...but do everything else manually for our self.__trades to ensure expected behavior
                new_trades = new_trades.select(self.__trades.column_names)
                self.__trades = pa.concat_tables([self.__trades, new_trades])
                self.__trades = combine_trace(self.__trades)
                columns = sorted(self.__trades.column_names)
                columns.remove('report_date')
                columns.insert(0, 'report_date')
                self.__trades = self.__trades.sort_by([(col, 'ascending') for col in columns])
        # filter by figi
        figi_trades = self.__trades.filter(pc.equal(self.__trades['figi'], pc.index(load_data.get_ordinals()['figi'], figi)))
        # slice the last 'count' trades...if count is greater than the number of figi_trades then get them all
        trades = figi_trades.slice(max(0, len(figi_trades) - count))
        # generate a batch for the trades
        X_b, _ = self.__generator.generate_batch(trades)
        # evaluate the batch
        Y_b_hat = self.__model.evaluate_batch(X_b)
        # retrieve the model prices
        model_prices = Y_b_hat[:, self.__generator.get_rfq_label_index('price')]
        # format the data, add the model price, and return the result
        result = []
        for i in range(len(trades)):
            side = load_data.get_ordinals()['side'][trades['side'][i].as_py()].as_py()
            buy_sell = load_data.get_ordinals()['buy_sell'][trades['buy_sell'][i].as_py()].as_py()
            if side != 'D':
                side = buy_sell
            result.append({
                "report_date": epoch_ns_to_iso_str(trades['report_date'][i].as_py()),
                "execution_date": epoch_ns_to_iso_str(trades['execution_date'][i].as_py()),
                "side": side,
                "quantity": int(trades['quantity'][i].as_py()),
                "price": float(trades['price'][i].as_py()),
                "yield": trades['yield'][i].as_py(),
                "model_price": model_prices[i]})
        return result

def create_token(origin_jti, jti):
    """Accept an origin_jti and jti and create a mock token"""
    return f".{base64.b64encode(json.dumps({'origin_jti': origin_jti, 'jti': jti}).encode('utf-8')).decode('ascii')}"
