import xml.etree.ElementTree as ET
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
import math

def load_sumo_trace(filepath: str, delta_t: float):
    """
    Parses a SUMO FCD XML trace and extracts vehicle positions at discrete
    time intervals separated by delta_t.
    
    Returns:
        dict: mapping { time_slot_index : { vehicle_id : (x, y, speed) } }
    """
    trace_data = defaultdict(dict)
    
    # We only care about timesteps that align with our slots
    # e.g., slot 0 is t=0, slot 1 is t=30, etc.
    # To handle floating point precision and exact matches, we track the next target
    current_slot = 0
    target_time = current_slot * delta_t
    
    # Use iterparse for memory efficiency on large FCD XML files
    context = ET.iterparse(filepath, events=("start", "end"))
    
    current_time = 0.0
    capture_current_step = False
    
    try:
        for event, elem in context:
            if event == "start" and elem.tag == "timestep":
                current_time = float(elem.get("time"))
                # If we crossed the target time, capture this timestep for the slot
                if current_time >= target_time - 0.001:
                    capture_current_step = True
                    
            elif event == "end" and elem.tag == "vehicle":
                if capture_current_step:
                    vid = elem.get("id")
                    x = float(elem.get("x"))
                    y = float(elem.get("y"))
                    speed = float(elem.get("speed"))
                    trace_data[current_slot][vid] = (x, y, speed)
                    
            elif event == "end" and elem.tag == "timestep":
                if capture_current_step:
                    # Move to next target slot
                    current_slot += 1
                    target_time = current_slot * delta_t
                    capture_current_step = False
                elem.clear() # clear memory
    except ET.ParseError:
        print("Warning: Reached end of incomplete XML trace. Returning parsed data so far.")
            
    return dict(trace_data)

def get_edge_hotspots(trace_data: dict, num_edges: int):
    """
    Finds optimal locations (x, y) for Edge Servers by clustering all vehicle
    positions recorded in the trace.
    
    Returns:
        list of tuples: [(x1, y1), (x2, y2), ..., (x_N, y_N)]
    """
    # Collect a sample of points to find hotspots
    points = []
    # To prevent massive arrays on huge traces, we take a subset of slots
    slots = list(trace_data.keys())
    # Subsample every 5th slot to speed up KMeans
    sample_slots = slots[::5] if len(slots) > 10 else slots
    
    for slot in sample_slots:
        for vid, (x, y, speed) in trace_data[slot].items():
            points.append([x, y])
            
    if len(points) == 0:
        # Fallback if trace is empty
        return [(0.0, 0.0) for _ in range(num_edges)]
        
    points_arr = np.array(points)
    
    # If we have fewer active vehicles than edge servers
    n_clusters = min(num_edges, len(points_arr))
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(points_arr)
    
    centers = kmeans.cluster_centers_
    
    hotspots = [(float(c[0]), float(c[1])) for c in centers]
    
    # Pad if we didn't have enough points for n_clusters
    while len(hotspots) < num_edges:
        hotspots.append(hotspots[0])
        
    return hotspots
