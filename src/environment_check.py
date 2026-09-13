#!/usr/bin/env python3
from __future__ import annotations
import importlib, sys
REQ=['numpy','pandas','scipy','requests','rasterio','geopandas','shapely','pyproj','networkx','sklearn','joblib','xarray','cdsapi']
BAD=[]
if not ((3,11) <= sys.version_info[:2] <= (3,13)):
    BAD.append(f"Python {sys.version.split()[0]} outside supported 3.11-3.13 (recommended 3.13)")
for name in REQ:
    try: importlib.import_module(name)
    except Exception as e: BAD.append(f"{name}: {e}")
print(f"Python: {sys.version.split()[0]}")
if BAD:
    print("Environment check: FAIL")
    for x in BAD: print(f"- {x}")
    raise SystemExit(1)
print("Environment check: PASS")
