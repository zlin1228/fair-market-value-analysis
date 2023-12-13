import asyncio
import websockets
import json
import pathlib
import ssl
import settings

########################################
# Used only for testing purposes.
########################################

async def start_fmv_stream():
    ssl_context=None
    uri = 'ws://localhost'
    uri += ":" + str(int(settings.get('$.server.fmv.port')))
    async with websockets.connect(uri, ssl=ssl_context) as websocket:
        fmv_start_json = {
            "fmv_stream": {
                #"figi":"BBG00JXGN2L3",
                #"figi":"BBG00FGQXY82",
                "figi":"BBG00JRWS7T1",
                "size":"10", "trade_count": 20
            }
        }

        await websocket.send(json.dumps(fmv_start_json))

        while True:
            r = await websocket.recv()
            print(r)

asyncio.run(start_fmv_stream())