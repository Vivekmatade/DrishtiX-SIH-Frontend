import urllib.request
import urllib.parse
import json
import ssl

ctx = ssl._create_unverified_context()
query = """[out:json][timeout:25];
(
  node(13.3230,77.1230,13.3310,77.1310);
  way(13.3230,77.1230,13.3310,77.1310);
);
out body;
"""
url = 'https://overpass-api.de/api/interpreter?data=' + urllib.parse.quote(query)
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req, context=ctx, timeout=20) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        print('Total elements:', len(data.get('elements', [])))
        for el in data.get('elements', []):
            tags = el.get('tags', {})
            name = tags.get('name') or tags.get('amenity') or tags.get('building')
            if 'name' in tags:
                print(f"[{el.get('type')}] ID: {el.get('id')} Name: {tags.get('name')} | Tags: {tags}")
                if 'lat' in el and 'lon' in el:
                    print(f"  Coord: lat={el['lat']}, lon={el['lon']}")
        with open('data/sit_tumkur/osm_elements.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
except Exception as e:
    print('Error:', e)
