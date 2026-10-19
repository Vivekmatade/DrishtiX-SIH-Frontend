import urllib.request
import ssl
import xml.etree.ElementTree as ET

ctx = ssl._create_unverified_context()
url = 'https://api.openstreetmap.org/api/0.6/map?bbox=77.1235,13.3240,77.1295,13.3295'
req = urllib.request.Request(url, headers={'User-Agent': 'DrustiX-Cadastral-App/1.0'})

with urllib.request.urlopen(req, context=ctx, timeout=20) as resp:
    content = resp.read()

with open('data/sit_tumkur/osm_map.xml', 'wb') as f:
    f.write(content)

root = ET.fromstring(content)
nodes = {}
for node in root.findall('node'):
    nid = node.attrib['id']
    nodes[nid] = (float(node.attrib['lat']), float(node.attrib['lon']))

with open('data/sit_tumkur/osm_parsed_buildings.txt', 'w', encoding='utf-8') as out:
    for node in root.findall('node'):
        tags = {t.attrib['k']: t.attrib['v'] for t in node.findall('tag')}
        if 'name' in tags or 'amenity' in tags:
            out.write(f"Node: {tags.get('name', tags.get('amenity'))} | {tags} at {nodes[node.attrib['id']]}\n")

    for way in root.findall('way'):
        tags = {t.attrib['k']: t.attrib['v'] for t in way.findall('tag')}
        nd_refs = [nd.attrib['ref'] for nd in way.findall('nd') if nd.attrib['ref'] in nodes]
        if nd_refs and ('name' in tags or 'building' in tags or 'amenity' in tags or 'highway' in tags):
            lats = [nodes[ref][0] for ref in nd_refs]
            lons = [nodes[ref][1] for ref in nd_refs]
            avg_lat = sum(lats)/len(lats)
            avg_lon = sum(lons)/len(lons)
            out.write(f"Way: {tags.get('name', tags.get('building', 'unnamed'))} | {tags} at lat={avg_lat:.6f}, lon={avg_lon:.6f}\n")

print("Finished parsing successfully!")
