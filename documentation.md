# Project Documentation: Joint Service Caching and Computation Offloading in VEC

This document provides a detailed overview of the work completed so far, covering the problem statement, dataset collection, mobility prediction, system integration, and the detailed working of the MA-DDPG framework.

## Problem Statement

In Vehicular Edge Computing environments, optimizing task processing delay and energy consumption relies heavily on the availability of necessary service programs at the edge. Existing methods employing heuristics (like LRU) or single-agent reinforcement learning (like DDPG) fail to perform optimally in dense, dynamic, multi-vehicle scenarios. Heuristics cannot anticipate future mobility or task demands, while single-agent DRL suffers from environment non-stationarity and leads to uncoordinated "herding behavior," causing RSU queue overloads and redundant cache evictions. Consequently, there is a lack of a cooperative, multi-agent framework capable of intelligently and jointly making continuous service caching and computation offloading decisions in real-time while explicitly resolving inter-vehicle resource conflicts.

## 1. Dataset Collection and Preprocessing

The foundation of our vehicular edge computing (VEC) simulation relies on realistic vehicular mobility data. We extracted and processed this data through the following pipeline:

1. **Road Network Extraction**: We used OpenStreetMap (OSM) Web Wizard to extract the road network of the Surathkal region, including junctions, lane information, and initial traffic movement data.
2. **SUMO Traffic Simulation**: The downloaded network was fed into the Simulation of Urban Mobility (SUMO) simulator (`osm.sumocfg`). This simulation generated realistic vehicular mobility behavior (movement, waiting times, congestion delays).
3. **Data Conversion**: The XML outputs (`tripinfos.xml`, `edgeData.xml`) were converted to CSV format using SUMO utility scripts (`xml2csv.py`).
4. **Coordinate Transformation**: Extracted UTM Zone 32N coordinates were converted to standard GPS WGS84 latitude/longitude using the `pyproj` library for broader compatibility.
5. **Data Cleaning**: Missing values and incomplete rows were removed, and the cleaned dataset was finalized as `tripinfos_clean.csv`.

## 2. Mobility Prediction

The objective of this phase was to accurately predict vehicle travel time (mobility prediction), which directly impacts resource allocation at the edge. 

### Selected Features
- `tripinfo_routeLength` (Total Travel Distance)
- `tripinfo_waitingTime` (Total Waiting Duration)
- `tripinfo_timeLoss` (Delay Due to Traffic/Congestion)
- `tripinfo_stopTime` (Total Stopped Duration)
- `tripinfo_speedFactor` (Relative Vehicle Speed Factor)
- `tripinfo_departDelay` (Initial Departure Delay)

**Target Variable**: `tripinfo_duration` (Travel time in seconds)

### Model Evaluation
We trained and evaluated several machine learning models. **CatBoost** achieved the best predictive performance, as it efficiently handles feature interactions.

| Model | RMSE | MAE | R² Score |
|-------|------|-----|----------|
| **CatBoost** | **70.07** | **48.99** | **0.9903** |
| LightGBM | 73.41 | 51.81 | 0.9894 |
| Random Forest | 75.44 | 52.42 | 0.9888 |
| XGBoost | 78.70 | 55.14 | 0.9878 |
| Decision Tree | 105.99 | 73.44 | 0.9779 |
| GRU | 872.82 | 685.99 | 0.0351 |

> Note: The GRU model performed poorly because our dataset consisted of independent trip-level samples rather than strictly sequential time-series data.

## 3. Simulation Parameters & Infrastructure

To simulate the Vehicular Edge Computing environment accurately, we established a robust infrastructure with defined capacities for vehicles, edge servers (RSUs), and the remote cloud.

### RSU Placement & Network Layout
- **Number of RSUs**: There are **3 Edge Servers (RSUs)** deployed along the road.
- **Coverage**: Each RSU has a communication range of **500 meters**.
- **Road Length**: This covers a total road length of **1500 meters** in the simulated environment.

### Vehicle Density (Cars in Dataset)
- **Vehicle Count**: The number of cars in the simulation is defined dynamically by a vehicle density parameter ($\rho$). 
- By default, $\rho = 4$ vehicles per edge server.
- This results in **12 total vehicles** interacting with the 3 RSUs simultaneously. The density can be scaled to test system robustness under different traffic loads (typically ranging from 2 to 5 cars per RSU).

### Cloud Integration
- **Role of the Cloud**: The Remote Core Cloud acts as the ultimate fallback for computation tasks. It possesses **unlimited computational power** and **infinite storage capacity** (meaning all 5 system services are permanently cached there).
- **Network Link**: The edge servers are connected to the cloud via a WAN link with a fixed bandwidth of **10 Mbps**.
- **Impact**: Offloading to the cloud (Case 6) guarantees successful execution but incurs significant transmission delay due to the long distance and lower bandwidth compared to local edge execution.

## 4. The 6-Cases Offloading and Caching Strategy

The VEC simulator determines execution routing based on where a requested service program is cached. This yields **6 distinct cases** depending on the vehicle’s cache, the nearest RSU's cache, and the neighboring edge pool's cache.

1. **Case 1 (Local Execution):** 
   - *Condition*: The required service program is cached locally on the vehicle itself. 
   - *Action*: The entire task is executed on the vehicle's local CPU. No offloading delay is incurred.
2. **Case 2 (Partial Offloading to Nearest Edge):** 
   - *Condition*: The service is cached *both* locally on the vehicle and at the nearest edge server. 
   - *Action*: The task is partitioned. A ratio $(1 - \alpha)$ is executed locally, and a ratio $\alpha$ is offloaded to the nearest edge server, allowing parallel execution.
3. **Case 3 (Complete Offloading to Nearest Edge):** 
   - *Condition*: The service is *not* cached locally, but it is available at the nearest edge server. 
   - *Action*: The entire task is transmitted over the wireless uplink and executed by the nearest edge server.
4. **Case 4 (Partial Offloading to Edge Pool):** 
   - *Condition*: The task is completely offloaded to the nearest edge server, but the nearest edge server is under heavy load. The neighboring edge pool also has the service cached.
   - *Action*: To balance the load, the nearest edge server processes a portion of the task and forwards the remaining portion to a neighboring edge server in the edge pool over the wired edge-to-edge link.
5. **Case 5 (Complete Offloading to Edge Pool):** 
   - *Condition*: The required service is *not* cached at the nearest edge server, but it *is* available in the neighboring edge pool. 
   - *Action*: The nearest edge server acts as a relay, forwarding the entire task to the neighboring edge server for execution.
6. **Case 6 (Remote Cloud Execution):** 
   - *Condition*: The required service is not cached locally, at the nearest edge server, nor anywhere in the edge pool. 
   - *Action*: The task must be offloaded to the remote Core Cloud via the WAN link. This incurs the maximum possible transmission delay and is heavily penalized by the reinforcement learning agent.

## 5. System Integration: Incorporating the Dataset

The cleaned dataset and the CatBoost mobility predictions act as the backbone for the VEC simulator. 

By anticipating how long a vehicle will remain in the coverage area of a specific Roadside Unit (RSU) and mapping its trajectory, the edge network can make **proactive caching** decisions.

### Workflow Integration

1. **Mobility Trace & Prediction**: SUMO traces and CatBoost predictions determine vehicle positions and expected travel durations.
2. **State Extraction**: The system monitors real-time RSU cache status, queue lengths, and uplink bandwidth.
3. **Action Execution**: This environment state is fed into our Deep Learning agent, which continuously decides where to offload tasks and which services to cache or evict across the network.

## 6. Multi-Agent Deep Deterministic Policy Gradient (MA-DDPG) Working

To optimize joint service caching and computation offloading, we designed a communication-enhanced Multi-Agent Reinforcement Learning (MARL) framework based on MA-DDPG. 

### Why MA-DDPG?
Single-agent DDPG assumes a stationary environment. In multi-vehicle scenarios, independent agents operating simultaneously often exhibit "herding behavior"—where multiple vehicles offload to the same RSU, causing sudden cache evictions, network congestion, and queue overloads. MA-DDPG solves this through a cooperative multi-agent architecture.

### Centralized Training with Decentralized Execution (CTDE)
- **Centralized Training**: During training, a centralized *critic* network observes the global state and joint actions of all agents. This stabilizes the learning process by overcoming the non-stationarity introduced by multiple agents interacting in the same shared environment.
- **Decentralized Execution**: During execution, local *actor* networks make decentralized decisions based solely on their local observations (task sizes, local queue length, bandwidth to RSU).

### Explicit Intent-Sharing Protocol
To further prevent herding behavior and maintain cache diversity, we integrated an explicit inter-agent communication protocol:
1. Vehicles broadcast continuous *intent messages* to neighboring agents.
2. These messages share the vehicle's intended caching and offloading targets.
3. Agents dynamically negotiate and adjust their caching requests based on the intents of others, resolving conflicts over the same RSU resources proactively.

### The Decision Process (MDP Formulation)
- **State**: Local task size, required service, RSU cache status, computational load of the RSU, vehicle CPU queue, and wireless channel state.
- **Action**: 
  - *Offloading ratio* (proportion of task offloaded to the edge).
  - *Caching decision vector* (priority for caching specific services, dictating cache evictions at the RSU).
- **Reward**: A global system cost function that penalizes high execution latency and energy consumption. 

By jointly optimizing continuous caching placements and offloading ratios, and facilitating explicit multi-agent coordination, the proposed MA-DDPG framework effectively minimizes latency and energy consumption while ensuring scalable, dynamic resource management in IoV networks.
