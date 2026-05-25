import gzip
import csv
import xml.etree.ElementTree as ET

with gzip.open("osm.net.xml.gz", "rt") as f:
    tree = ET.parse(f)

root = tree.getroot()

with open("utm_nodes.csv", "w", newline="") as out:
    writer = csv.writer(out)
    writer.writerow(["node_id", "utm_easting", "utm_northing"])

    for node in root.findall(".//node"):
        writer.writerow([
            node.get("id"),
            node.get("x"),
            node.get("y")
        ])

