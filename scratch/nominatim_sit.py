import urllib.request
import urllib.parse
import json
import ssl
import time

ctx = ssl._create_unverified_context()
queries = [
    'SIT Tumkur',
    'SIT College Tumkur',
    'Siddaganga Institute of Technology',
    'SIT Library Tumkur',
    'SIT Auditorium Tumkur',
    'SIT Stadium Tumkur'
]

for q in queries:
    url = 'https://nominatim.openstreetmap.org/search?q=' + urllib.parse.quote(q) + '&format=json&limit=3'
    req = urllib.request.Request(url, headers={'User-Agent': 'DrustiX-Cadastral-App/1.0'})
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            print(f"Query: {q} -> Found {len(data)}")
            for r in data:
                print(f"   Name: {r.get('display_name')} | Lat/Lon: {r.get('lat')}, {r.get('lon')}")
    except Exception as e:
        print(f"Query: {q} -> Error {e}")
    time.sleep(1)
