# GTA-V: Distributed Task Offloading for Autonomous Vehicles: A Potential Game Theory Driven Approach
## Execution Environment

- Operating System: Microsoft Windows 10, 64-bit.

- Physical Memory (RAM) 16.0 GB.

### Prerequisites

- Python 3.8
- PyCharm Community Edition 2021.2. 
- SimPy simulator for experimental setup and execution.

## Installation of Simpy Simulator in Windows Environment

SimPy is a process-based discrete-event simulation framework based on standard Python. Processes in SimPy are defined by Python generator functions and 
can, for example, be used to model active components like customers, vehicles, or agents[2]. 

Install simpy-4.1.1.tar.gz from the URL:https://pypi.org/project/simpy/#files

### Repository Usage Guidelines:

Download the GTA-V repository and keep it in the drive where the Simpy simulator is present. The  GTA-V repository contains all executable files related to the proposed and baseline approaches.
- game_theory.py -> The main file is related to the proposed GTA-V approach, where we use non-cooperative game theory to optimise task offloading decisions.
  
- greedy_offloader.py -> The main file is related to a greedy-based strategy used to perform task offloading to improve task delay and energy, improving the satisfaction ratio[1].
  
- local_baseline.py -> The  main file is related to the local computation of all tasks inside the On-Board Unit(OBU) in autonomous vehicles.
  
- random_baseline.py -> The main file is related to the random offloading of tasks in local, RSUs, and cloud based on the resource availability[3].

- deco_offloader.py> The main file related to DISCO baseline approach, where deadline-based task scheduling is done for allocation of tasks[4].

### References

[1] P. Kumar, S. Sushma, K. Chandrasekaran, S. K. Addya, Collaborative deadline-sensitive multi-task offloading in vehicular-cloud networks, in: 2025 17th International Conference on Communication Systems
and NETworks (COMSNETS), IEEE, 2025, pp. 979–983.

[2] R. I. Tinini, M. R. P. dos Santos, G. B. Figueiredo, D. M. Batista, 5gpy: A Simpy-based simulator for performance evaluations in 5 G hybrid cloud-fog ran architectures, Simul. Model. Pract. Theory 101 (2020)102030.

[3] F. Zeng, Q. Chen, L. Meng and J. Wu, "Volunteer Assisted Collaborative Offloading and Resource Allocation in Vehicular Edge Computing," in IEEE Transactions on Intelligent Transportation Systems, vol. 22, no. 6, pp. 3247-3257, June 2021, doi: 10.1109/TITS.2020.2980422. keywords: {Servers;Task analysis;Edge computing;Resource management;Quality of service;Games;Energy consumption;Vehicular edge computing;volunteer assisted;stackelberg game;resource allocation;offloading}, 

[4] S. Azizi, M. Othman, H. Khamfroush, Deco: A deadline-aware and energy-efficient algorithm for task offloading in mobile edge computing, IEEE Systems Journal 17 (1) (2022) 952–963.

### Contributors

- Mrs. Sushma S A

https://scholar.google.co.in/citations?user=nW4F_3wAAAAJ&hl=en

- Mr. Prasanna Kumar 

prasanna-kumar-26@github.io 

-Dr. K Chandrashekaran

kch@nitk.edu.in

- Dr. Sourav Kanti Addya

https://souravkaddya.in/


### Citation & Acknowledgements
If any difficulties kindly send the messages to sushmasa.sit@gmail.com
