import asyncio
from bleak import BleakScanner
import yaml
from importlib import resources

async def run():
    info = {}
    devices = await BleakScanner.discover()
    for d in devices:
        if 'H10' in d.name:
            print(d)
            info['address'] = d.address
            info['name'] = d.name
            # info['metadata'] = d.metadata
            with resources.path('ActWalkEst.resources','polar.yml') as p:
                path=str(p)
            with open(p, 'w') as outfile:
                yaml.dump(info, outfile)
if __name__=="__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())
