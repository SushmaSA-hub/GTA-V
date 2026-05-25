import sys

new_content = r"""%------------------------------------------------
\chapter{Introduction}

The landscape of modern transportation is undergoing a paradigm shift driven by the evolution of 5G and emerging 6G wireless communication networks, leading to the realization of the Internet of Vehicles (IoV). Smart vehicles today are no longer merely mechanical modes of transport; they are highly sophisticated computing platforms equipped with a multitude of sensors, cameras, RADARs, and LiDARs. These vehicles run advanced, computation-intensive applications such as autonomous driving algorithms, Augmented Reality (AR) assisted navigation, real-time traffic condition monitoring, high-definition video streaming, and sophisticated infotainment systems. These applications demand massive computational resources and impose stringent requirements on ultra-low latency to ensure passenger safety and provide a seamless user experience. Efficient management of computational resources is essential to ensure better Quality of Service (QoS), reduced operational costs, lower energy consumption, and improved overall system performance \cite{vec_concept}.Vehicular Edge Computing (VEC) environments are highly dynamic because computational workloads continuously change depending on vehicle mobility and traffic density. Resource utilization at Roadside Units (RSUs) may vary significantly over time due to fluctuations in task offloading requests and edge-level scheduling operations. In such environments, accurate prediction and allocation of edge resources becomes an important task. Predicting future CPU, memory, and cache usage enables VEC providers to allocate resources proactively, avoid resource bottlenecks, minimize transmission latency, and improve infrastructure utilization. Accurate caching and offloading predictions also help reduce under-utilization and over-utilization of edge servers, which directly impacts energy efficiency and operational cost \cite{mobility_caching}.\\

Traditional resource allocation methods generally rely on real-time information without considering future mobility patterns. As a result, these methods often fail to make optimal resource provisioning decisions in highly heterogeneous VEC environments. Machine learning and deep learning approaches have recently shown significant improvements in workload prediction and resource allocation by learning complex patterns from historical mobility traces. Models such as Deep Q-Networks (DQN) and Deep Deterministic Policy Gradient (DDPG) are increasingly used for edge resource prediction because of their ability to model nonlinear relationships and temporal dependencies in large-scale vehicular datasets \cite{joint_caching_offloading}.Among these approaches, Multi-Agent Reinforcement Learning (MARL) has emerged as a promising distributed learning paradigm that enables multiple vehicular clients to collaboratively learn an optimal joint policy. Instead of relying on a single centralized controller which introduces massive communication overhead, vehicular agents learn continuous offloading and caching strategies locally, significantly improving scalability and reducing the risks associated with centralized decision bottlenecks \cite{maddpg_paper}.\\

Despite the advantages of MARL, several challenges still exist in practical VEC environments. One of the major issues is the presence of non-stationary environments across different agents, where the actions of one vehicle directly alter the state observed by others. Such non-stationarity can reduce model convergence speed and policy stability. In addition, recent studies have shown that independent agents operating without coordination often engage in "herding behavior", where multiple vehicles simultaneously target the same RSU, causing severe cache thrashing and network congestion. Centralized Training with Decentralized Execution (CTDE) has been widely adopted to address these non-stationarity concerns by utilizing a global critic during training. However, without explicit inter-agent communication, actors may still execute conflicting actions during deployment \cite{marl_comm}. Therefore, designing an efficient multi-agent framework that achieves a better balance between decentralized execution and explicit intent coordination remains an important research challenge in VEC resource optimization.

\section{Scope of Project}
The scope of this project is to develop a communication-enhanced MARL framework for accurate joint service caching and computation offloading in Vehicular Edge Computing environments. The proposed system predicts optimal offloading destinations and proactive caching replacements while ensuring that multi-agent resource conflicts are minimized. The project focuses on integrating the Multi-Agent Deep Deterministic Policy Gradient (MA-DDPG) algorithm with a Centralized Training, Decentralized Execution (CTDE) architecture. A continuous intent-sharing communication protocol is introduced to resolve resource conflicts and prevent herding behavior among independent vehicular agents. The proposed framework handles highly dynamic mobility traces using SUMO and the LEAF edge computing simulator. The system is evaluated under varying vehicle densities, task sizes, and computational capacities, and compared with existing methods such as single-agent DDPG, LRU, and greedy heuristics to validate its superiority in minimizing latency and energy consumption.

\section{Applications}

\begin{itemize}

\item \textbf{Autonomous Driving Navigation:}  
The proposed system facilitates real-time processing of massive sensor data by ensuring that necessary machine learning models (e.g., object detection, trajectory prediction) are cached at the edge, guaranteeing the ultra-low latency required for safe autonomous navigation.

\item \textbf{Augmented Reality (AR) Assisted Driving:}  
The framework enables immersive AR navigation systems on vehicle windshields by intelligently offloading heavy rendering tasks to edge servers that cache the required AR rendering engines and local high-definition map data.

\item \textbf{Smart Traffic Flow Optimization:}  
Processing aggregated traffic data from multiple vehicles at the edge to provide real-time route optimization, hazard warnings, and traffic light scheduling without relying on distant cloud servers, drastically reducing response times.

\item \textbf{In-Vehicle Infotainment Systems:}  
The proposed model improves the Quality of Experience (QoE) for passengers by providing low-latency access to media transcoding services, gaming servers, and high-definition video streaming cached directly at RSUs.

\item \textbf{Energy-Efficient Edge Resource Management:}  
Accurate caching and offloading predictions minimize unnecessary data transmissions to the core cloud and prevent RSU overloads. This reduces power consumption across the network and improves overall energy efficiency.

\item \textbf{Cooperative Distributed Computing Grids:}  
Integrating with broader smart city initiatives by turning vehicles and RSUs into a cooperative, distributed computing grid capable of handling localized data processing tasks efficiently, resolving conflicts through continuous multi-agent communication.

\item \textbf{Dynamic Resource Scaling in RSUs:}  
The framework supports automatic scaling of computation offloading ratios based on current RSU workload demand. During peak traffic conditions, tasks can be intelligently distributed to the cloud to maintain performance and prevent edge crashes.

\item \textbf{Quality of Service (QoS) Improvement for V2X:}  
The proposed model helps prevent edge system overload and bandwidth bottlenecks through proactive caching negotiation. This improves response time, reliability, and service availability for all connected vehicles.

\item \textbf{Real-Time Video Surveillance and Analytics:}  
Joint caching allows heavy computer vision models to reside at the edge, enabling rapid processing of dashboard camera feeds for real-time anomaly detection and surveillance without overwhelming the core network.

\item \textbf{Secure Multi-Agent Edge Environments:}  
The CTDE architecture provides a foundation for secure multi-agent coordination where vehicles do not need to share their raw internal states during execution, reducing communication overhead and enhancing scalability in dense deployments.

\end{itemize}
%------------------------------------------------

\chapter{Motivation}

The increasing dependence on connected and autonomous vehicles has created a strong demand for intelligent and efficient resource management systems at the network edge. Modern IoV platforms host a multitude of applications related to artificial intelligence, online streaming, sensor fusion, and real-time analytics. These applications continuously generate highly dynamic workloads that consume CPU, memory, and bandwidth resources at varying rates. Managing such rapidly changing workloads efficiently is a major challenge for network providers because improper allocation of edge resources can directly affect system performance, operational cost, and most critically, passenger safety. One of the primary reasons for this project is the inefficiency of traditional edge resource allocation methods. In many VEC infrastructures, caching decisions are made reactively using heuristics such as Least Recently Used (LRU) or Least Frequently Used (LFU). Such approaches are insufficient in dynamic environments because they fail to anticipate future resource demands and vehicle mobility patterns. When future caching needs are not predicted accurately, edge systems experience high cache miss rates, forcing tasks to be routed to the distant core cloud. This causes resource bottlenecks, increased transmission latency, and degraded Quality of Service (QoS). On the other hand, static offloading strategies fail to adapt to the real-time computational load of RSUs. Therefore, intelligent, joint optimization of caching and offloading has become an essential requirement for modern IoV systems.\\

The limitations of single-agent reinforcement learning present another significant concern addressed in this work. While Deep Reinforcement Learning (DRL) has been widely adopted for resource allocation, standard models like Deep Deterministic Policy Gradient (DDPG) assume a stationary environment or a single centralized controller. In real-world vehicular networks, multiple independent vehicles (agents) interact with the same shared edge resources simultaneously. When multiple vehicles employ independent DRL agents without coordination, it leads to a non-stationary environment. An action taken by one vehicle (e.g., filling an RSU's cache) directly alters the state observed by another vehicle. This non-stationarity causes severe instability during training and leads to suboptimal, greedy decisions. Multi-Agent Reinforcement Learning (MARL) emerged as a potential solution to these coordination concerns because it enables collaborative policy learning. However, practical deployment of MARL in dense vehicular environments still faces several unresolved challenges that motivated this research work.\\

One of the major problems in MARL is the presence of uncoordinated "herding behavior". In real-world VEC systems, if a particular RSU is observed to have available cache space and low computational load, multiple uncoordinated agents may simultaneously decide to offload their tasks to that RSU and request it to cache their specific services. This simultaneous influx overwhelms the RSU's queue and triggers a cascade of cache evictions, severely degrading prediction accuracy and system stability. Traditional multi-agent algorithms struggle to handle such conflicts because local actors often execute identical actions based on identical local observations. This results in unstable execution, slower convergence, and reduced task success rates. The problem becomes even more critical when strict latency requirements are introduced. Existing DDPG-based methods often use continuous action spaces without considering the quality or stability of simultaneous caching updates. As a result, unnecessary conflicts are introduced into the edge queue, reducing prediction performance and slowing convergence. This created the need for a more intelligent intent-sharing mechanism capable of balancing decentralized execution and cooperative intent dynamically.\\

Another major challenge comes from the limitations of existing multi-agent approaches such as independent DDPG and standard CTDE. Independent DDPG attempts to handle multi-agent scenarios, but its performance degrades rapidly as the number of vehicles increases because non-stationarity weakens model learning. Similarly, standard CTDE introduces a global critic to stabilize training. However, static decentralized actors without communication still lead to execution-time instability and resource thrashing when faced with identically optimal edge nodes.\\

These limitations motivated the development of a new framework that could:
\begin{itemize}
    \item Improve joint caching and offloading accuracy in highly dynamic IoV environments.
    \item Resolve the non-stationarity problem inherent in multi-agent reinforcement learning.
    \item Prevent "herding behavior" and catastrophic cache thrashing at RSUs.
    \item Improve model convergence and training stability through centralized training.
    \item Support scalable, decentralized execution enhanced by explicit communication protocols.
\end{itemize}

The introduction of an explicit intent-sharing communication protocol in this project helps overcome the limitations of uncoordinated MARL. Instead of acting blindly, the proposed approach dynamically adjusts caching requests based on continuous message vectors broadcasted by neighboring vehicles. Agents contributing identical requests negotiate to select diverse caching targets. This adaptive mechanism helps achieve a better latency-efficiency trade-off without sacrificing decentralized execution speed. Similarly, Centralized Training with Decentralized Execution (CTDE) helps reduce conflicts caused by non-stationary environment dynamics. By evaluating joint actions globally during training, the network's value approximations become more coherent and stable. Dynamic intent sharing further improves adaptability because vehicle density and edge availability change continuously over time. This makes static MARL execution insufficient for real-world applications.\\

Experimental evaluation using the LEAF simulator and realistic SUMO mobility traces further supports this work by demonstrating that existing approaches still suffer from significant prediction errors and instability under dense conditions. The proposed MA-DDPG framework with continuous intent sharing and CTDE architecture aims to overcome these challenges by combining accurate continuous learning with intelligent multi-agent conflict resolution mechanisms.

%------------------------------------------------
\chapter{Literature Survey}

\section{Prediction methods for effective resource provisioning in cloud computing: A survey}
K. Dinesh Kumar and E. Umamaheswari presented a comprehensive survey on prediction methods for effective resource provisioning in cloud computing environments. The paper analyzed various machine learning and statistical prediction techniques for workload forecasting and dynamic resource allocation. The study highlighted the importance of accurate prediction methods in improving QoS, reducing SLA violations, optimizing energy consumption, and enhancing cloud and edge resource utilization efficiency.

\section{Computation Offloading and Content Caching for MEC in Software Defined Vehicular Networks}
Zhao et al. investigated the joint optimization of computation offloading and content caching in Vehicular Edge Computing environments utilizing Software Defined Networking (SDN). The paper formulated the problem as a mixed-integer non-linear programming (MINLP) model to minimize the total latency of vehicular tasks. The authors proposed a collaborative caching scheme where RSUs share their cached contents to improve the overall cache hit rate. While their mathematical approach provided optimal solutions under static conditions, the heavy computational complexity of MINLP makes it unsuitable for real-time decision-making in highly dynamic, fast-moving vehicular networks where network topologies change every few seconds.

\section{Deep Reinforcement Learning for Joint Caching and Offloading in VEC}
Zhang et al. proposed a single-agent Deep Reinforcement Learning framework to tackle the joint caching and offloading problem. By utilizing the Deep Deterministic Policy Gradient (DDPG) algorithm, the framework learned to make continuous offloading and caching replacement decisions by interacting with the environment, entirely bypassing the need for computationally expensive MINLP solvers. The study demonstrated that DRL could significantly reduce system delay compared to traditional heuristics like LRU. However, the model assumed a single centralized agent controlling all vehicles, which introduces massive communication overhead and scalability issues as the number of vehicles increases.

\section{Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments}
Lowe et al. introduced the Multi-Agent Deep Deterministic Policy Gradient (MA-DDPG) algorithm, extending traditional actor-critic methods to multi-agent environments. The paper proposed the Centralized Training, Decentralized Execution (CTDE) architecture, which utilizes a centralized critic to observe the global state and joint actions during training to stabilize learning, while decentralized actors take actions based only on local observations during execution. The study demonstrated that MA-DDPG effectively overcomes the non-stationarity problem inherent in independent Q-learning or DDPG. Their work provided the foundational theoretical framework for applying MARL to complex, distributed resource allocation problems.

\section{Learning Multiagent Communication with Backpropagation}
Sukhbaatar et al. explored the necessity of communication in multi-agent reinforcement learning. They proposed CommNet, a continuous communication protocol where agents output a continuous message vector alongside their action. These messages are pooled and fed into the hidden layers of other agents at the next time step, allowing for backpropagation through the communication channel. The study showed that agents could learn to negotiate and cooperate explicitly, significantly improving performance in fully cooperative tasks compared to agents operating in silence. This highlighted the importance of intent sharing in resolving conflicts in shared environments.

\section{Mobility-Aware Edge Caching in Vehicular Networks using Deep Learning}
Li et al. addressed the challenge of high-speed vehicle mobility in VEC caching. They proposed a mobility-aware proactive caching strategy that utilizes recurrent neural networks (LSTMs) to predict vehicle trajectories based on historical SUMO traces. By predicting which RSU a vehicle will connect to in the near future, the system can proactively cache the required services at the target RSU before the vehicle arrives, minimizing cache misses. While effective at handling mobility, the study treated caching in isolation and did not jointly optimize it with the continuous, real-time computation offloading decisions required for delay-sensitive tasks.

\section{Resource Allocation in VEC using Heuristic Approaches}
Kumar et al. presented a comprehensive survey on traditional heuristic and statistical prediction methods for effective resource provisioning in edge computing. The paper analyzed reactive caching strategies such as Least Recently Used (LRU) and Least Frequently Used (LFU), as well as greedy offloading strategies based on latency minimization (LatencyMin) and energy minimization (EnergyMin). The survey highlighted that while heuristics are computationally inexpensive and easy to deploy, they consistently lead to suboptimal resource utilization and frequent QoS violations in highly dynamic environments because they completely lack the ability to anticipate future network states or workload patterns.

\section{Deep Reinforcement Learning for Resource Allocation in V2V Communications}
Ye et al. proposed a decentralized resource allocation mechanism for Vehicle-to-Vehicle (V2V) communications based on deep reinforcement learning. The paper demonstrated how independent agents could learn to select optimal sub-bands and power levels to maximize V2V link capacity while meeting strict latency constraints. The study highlighted the challenges of multi-agent environments, showing that while independent DRL can operate decentrally, it suffers from severe convergence instability compared to coordinated approaches.

\section{Multi-Agent Reinforcement Learning for Cooperative Edge Caching}
Jiang et al. proposed a cooperative edge caching framework utilizing multi-agent reinforcement learning to optimize cache hit rates across multiple base stations. The study modeled the base stations as independent agents attempting to learn a joint caching policy. The paper demonstrated that enabling explicit state sharing between agents significantly improved the global cache hit ratio and reduced overall transmission delay compared to isolated caching strategies like LRU or LFU, proving the necessity of inter-agent coordination in distributed edge networks.

%------------------------------------------------
\chapter{Problem Statement and Objectives}

\section{Problem Statement}

Existing methods such as independent single-agent Deep Deterministic Policy Gradient (DDPG), Least Recently Used (LRU) caching, and greedy latency heuristics experience uncoordinated action conflicts, execution instability, and poor robustness, leading to reduced efficiency in minimizing latency and energy consumption during joint service caching and computation offloading in highly dynamic vehicular edge computing environments.

%------------------------------------------------

\section{Objectives}

The main objectives of this project are:

\begin{itemize}

\item To develop a communication-enhanced Multi-Agent Reinforcement Learning (MARL) framework for Vehicular Edge Computing resource management.

\item To predict optimal offloading destinations and proactive service caching replacements using continuous action spaces accurately.

\item To implement a Multi-Agent Deep Deterministic Policy Gradient (MA-DDPG) algorithm for modeling structural and temporal workload relationships.

\item To preserve cache diversity by integrating explicit continuous intent-sharing protocols between vehicular agents.

\item To design a stability-aware Centralized Training, Decentralized Execution (CTDE) mechanism to improve learning stability.

\item To handle heterogeneous non-stationary multi-agent dynamics using shared global critics during training.

\item To improve global model convergence and mitigate catastrophic "herding behavior" and cache thrashing at RSUs.

\item To evaluate the robustness of the proposed framework under dynamic mobility constraints utilizing SUMO traffic traces and LEAF.

\item To compare the proposed model with existing approaches such as NoCache, LRU, LatencyMin, EnergyMin, and DDPG using performance metrics including Total Delay and Total Energy.

\item To analyze the performance of the proposed multi-agent framework under different operational parameters such as varying vehicle densities ($\rho$), task data sizes, and edge server CPU cycle frequencies.

\end{itemize}
"""

with open('/home/vamsi/Desktop/leaf/examples/vec_caching/thesis_final.tex', 'r') as f:
    lines = f.readlines()

# The start is line 97 (index 96), and the end is line 196 (index 196). We will replace 97 through 196.
# Wait, let's just find the indices for \chapter{Introduction} and \chapter{Methodology}
start_idx = -1
end_idx = -1

for i, line in enumerate(lines):
    if line.strip() == '%------------------------------------------------':
        if i+1 < len(lines) and lines[i+1].strip() == r'\chapter{Introduction}':
            start_idx = i
            break

for i in range(start_idx+1, len(lines)):
    if line.strip() == '%------------------------------------------------':
        pass
    if lines[i].strip() == r'\chapter{Methodology}':
        # the end idx should be the line above this, which is %-----------
        end_idx = i - 1
        break

if start_idx != -1 and end_idx != -1:
    new_lines = lines[:start_idx] + [new_content + "\n"] + lines[end_idx:]
    with open('/home/vamsi/Desktop/leaf/examples/vec_caching/thesis_final.tex', 'w') as f:
        f.writelines(new_lines)
    print(f"Successfully replaced content from line {start_idx+1} to {end_idx+1}")
else:
    print(f"Failed to find indices. Start: {start_idx}, End: {end_idx}")

