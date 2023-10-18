import os
import asyncio
import yaml
from importlib import resources

from ActWalkEst.polar.collect_polar import start_polar
from ActWalkEst.polar.search_polar import run as search

def pol_search():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(search())

def start():
    pol_search()
    with resources.path('ActWalkEst.resources','polar.yml') as p:
        path=str(p)
    file=open(path,'r')
    polar_info=yaml.safe_load(file)
    address=polar_info['address']
    print(address)
    start_polar(address)



