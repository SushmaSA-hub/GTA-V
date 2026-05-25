import sys

new_content = r"""%------------------------------------------------
\chapter{Introduction}

The rapid development of Internet of Things (IoT) and artificial intelligence (AI) has recently led to the emergence of diverse computation-intensive and latency-sensitive vehicular applications, such as augmented reality (AR) navigation and autonomous driving. However, offloading all the computing tasks to the remote cloud results in a heavy backhaul load and unacceptable latency. Through vehicular edge computing (VEC) \cite{vec_concept}, vehicular users can offload their tasks via vehicle-to-infrastructure (V2I) communications to edge nodes to reduce response latency, which caters for unprecedentedly exploding data traffic and increasingly stringent requirements of vehicular applications.\\

Task computation requires both input task data from users and program data installed on edges, which are defined as content caching and service caching, respectively. Content caching refers to caching of the input data needed and output data generated (e.g., in computational or infotainment applications) at vehicles and edge nodes \cite{mobility_caching}. Since these data dominate mobile data traffic, content caching at edge servers can effectively alleviate mobile traffic on backhaul links and reduce content delivery latency. On the other hand, service caching refers to caching the specific programs for task execution \cite{joint_caching_offloading}. As a motivating example, in object detection, the input data consists of videos and radar sensor data, and task execution requires the corresponding object detection service program to be cached in the vehicle or the edge server. The input data of object detection service is typically unique and hardly reusable for other executions. In comparison, service program data in the cache is evidently reusable by future executions of the same type of tasks. Because edge servers have limited caching space, how to selectively cache service program over space (e.g., at multiple edge servers) and time resources for achieving optimum transmission and computing performance is crucial for efficient task computation.\\

The design of optimal computation offloading and service caching faces many challenges in vehicular networks. First, vehicles and edge servers can only cache a small number of service programs at a time due to limited storage space. Thus, which service programs should be cached needs to be decided judiciously. Second, the computing resources on different edge servers may be unevenly distributed. It is critical to balance the computation load by cooperative offloading. Third, the computation offloading decisions and the service caching decisions are closely correlated. Intuitively, we tend to offload a task if the required program is already cached at the edge. Besides, the network status and available resources of edge servers change dynamically during the movement of vehicles. Therefore, it becomes significant and yet very challenging to design an appropriate service caching and computation offloading strategy in the VEC systems.\\

In this thesis, we investigate a joint optimization of service caching and computation offloading for a VEC system with limited storage and computing capacities, taking account of time-varying task requests and dynamic network topology. In order to make full use of the limited caching and computing resources of each node (i.e., vehicles and edge servers) as well as the cooperative offloading between edge servers, we propose a multi-agent deep reinforcement learning (MARL)-based service caching and computation offloading scheme \cite{maddpg_paper}. By introducing continuous intent-sharing communication between agents \cite{marl_comm}, the proposed framework provides low-complexity decision making, prevents cache thrashing, and enables adaptive, conflict-free resource management across the vehicular edge.

"""

with open('/home/vamsi/Desktop/leaf/examples/vec_caching/thesis_final.tex', 'r') as f:
    lines = f.readlines()

start_idx = -1
end_idx = -1

for i, line in enumerate(lines):
    if line.strip() == r'\chapter{Introduction}':
        start_idx = i - 1 # Include the %------ line above it
        break

for i in range(start_idx+1, len(lines)):
    if line.strip() == r'\section{Scope of Project}':
        end_idx = i
        break

if start_idx != -1 and end_idx != -1:
    new_lines = lines[:start_idx] + [new_content] + lines[end_idx:]
    with open('/home/vamsi/Desktop/leaf/examples/vec_caching/thesis_final.tex', 'w') as f:
        f.writelines(new_lines)
    print(f"Successfully replaced intro content")
else:
    print(f"Failed to find indices. Start: {start_idx}, End: {end_idx}")

