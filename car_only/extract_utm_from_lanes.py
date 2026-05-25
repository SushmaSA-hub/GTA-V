import xml.etree.ElementTree as ET
import csv

tree = ET.parse("osm.net.xml")
root = tree.getroot()

with open("utm_from_lanes.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["lane_id", "utm_easting", "utm_northing"])

    for lane in root.findall(".//lane"):
        shape = lane.get("shape")
        if shape:
            for point in shape.split():
                x, y = point.split(",")
                writer.writerow([lane.get("id"), x, y])
