import csv
from pyproj import Transformer

# Define transformer: UTM Zone 32N -> WGS84
transformer = Transformer.from_crs(
    "EPSG:32632",   # UTM Zone 32N
    "EPSG:4326",    # WGS84 (lat, lon)
    always_xy=True  # ensures (x, y) -> (lon, lat)
)

with open("utm_from_lanes.csv", "r") as infile, \
     open("lanes_wgs84.csv", "w", newline="") as outfile:

    reader = csv.DictReader(infile)
    fieldnames = ["lane_id", "longitude", "latitude"]
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        x = float(row["utm_easting"])
        y = float(row["utm_northing"])

        lon, lat = transformer.transform(x, y)

        writer.writerow({
            "lane_id": row["lane_id"],
            "longitude": lon,
            "latitude": lat
        })
