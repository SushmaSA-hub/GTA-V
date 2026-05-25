with open('/home/vamsi/Desktop/leaf/examples/vec_caching/thesis_final.tex', 'r') as f:
    lines = f.readlines()

new_methodology = r"""\chapter{Methodology}
\label{ch:methodology}

\section{Road Network Data Collection}
The study started with collecting the real-world road network from the Surathkal region using OpenStreetMap (OSM).
The OSM Web Wizard available in SUMO was used to download:
\begin{itemize}
    \item Road network
    \item Junctions
    \item Lane information
    \item Vehicle routes
    \item Traffic movement data
\end{itemize}
The downloaded network was converted into SUMO-compatible files.
Generated files include:
\begin{itemize}
    \item osm.net.xml.gz
    \item osm.passenger.trips.xml
    \item osm.sumocfg
    \item osm.view.xml
\end{itemize}

\section{SUMO Traffic Simulation}
After generating the road network, traffic simulation was performed using SUMO (Simulation of Urban Mobility).
The simulation was executed using the following command:
\begin{verbatim}
sumo -c osm.sumocfg
\end{verbatim}
The simulator generated realistic vehicular mobility behavior including:
\begin{itemize}
    \item Vehicle movement
    \item Waiting time
    \item Congestion delay
    \item Travel duration
    \item Stop time
    \item Route information
\end{itemize}
Generated simulation outputs:
\begin{itemize}
    \item tripinfos.xml
    \item edgeData.xml
    \item stats.xml
\end{itemize}

\section{XML to CSV Conversion}
The SUMO simulation outputs were generated in XML format.
To enable machine learning analysis, the XML files were converted into CSV format using the SUMO utility script:
\begin{verbatim}
xml2csv.py
\end{verbatim}
Generated CSV files include:
\begin{itemize}
    \item tripinfos.csv
    \item edgeData.csv
    \item stats.csv
\end{itemize}

\section{UTM Coordinate Extraction}
The SUMO road network internally used UTM Zone 32N coordinates.
Coordinates were extracted from:
\begin{itemize}
    \item Lane files
    \item Network files
\end{itemize}
Extracted attributes included:
\begin{itemize}
    \item UTM Easting
    \item UTM Northing
    \item Lane ID
\end{itemize}
Generated file:
\begin{verbatim}
utm_from_lanes.csv
\end{verbatim}

\section{UTM to WGS84 Coordinate Conversion}
To make the coordinates compatible with standard GPS-based geographic systems, UTM Zone 32N coordinates were converted into WGS84 latitude and longitude format.
The Python library used for conversion was:
\begin{verbatim}
pyproj
\end{verbatim}
Transformation performed:
\begin{center}
UTM Zone 32N $\rightarrow$ WGS84
\end{center}
Generated output file:
\begin{verbatim}
lanes_wgs84.csv
\end{verbatim}

\section{Data Cleaning and Preprocessing}
Before training machine learning models, the dataset was preprocessed to improve data quality.
The preprocessing operations included:
\begin{itemize}
    \item Removing missing values
    \item Removing incomplete rows
    \item Numeric datatype conversion
    \item Feature selection
\end{itemize}
Python libraries used:
\begin{itemize}
    \item Pandas
    \item NumPy
\end{itemize}
The cleaned dataset was stored as:
\begin{verbatim}
tripinfos_clean.csv
\end{verbatim}

\section{Feature Selection}
The following features were selected for mobility prediction.
\begin{table}[H]
\centering
\begin{tabular}{|c|c|}
\hline
\textbf{Feature} & \textbf{Description} \\
\hline
tripinfo\_routeLength & Total travel distance \\
tripinfo\_waitingTime & Vehicle waiting duration \\
tripinfo\_timeLoss & Delay due to congestion \\
tripinfo\_stopTime & Vehicle stop duration \\
tripinfo\_speedFactor & Relative speed factor \\
tripinfo\_departDelay & Vehicle departure delay \\
\hline
\end{tabular}
\caption{Selected Features}
\end{table}
Target variable:
\begin{verbatim}
tripinfo_duration
\end{verbatim}

\section{Train-Test Split}
The dataset was divided into training and testing subsets.
\begin{table}[H]
\centering
\begin{tabular}{|c|c|}
\hline
\textbf{Dataset} & \textbf{Percentage} \\
\hline
Training Data & 80\% \\
Testing Data & 20\% \\
\hline
\end{tabular}
\caption{Train-Test Split}
\end{table}
The training dataset was used for model learning while the testing dataset was used for evaluation.

\section{Machine Learning Models}
The following machine learning and deep learning models were implemented.
\subsection{Decision Tree}
Decision Tree was implemented as a baseline regression model.
\subsection{Random Forest}
Random Forest combined multiple decision trees using ensemble learning.
\subsection{XGBoost}
XGBoost was implemented for gradient boosting-based prediction.
\subsection{LightGBM}
LightGBM improved computational efficiency using leaf-wise tree growth.
\subsection{CatBoost}
CatBoost efficiently handled feature interactions and achieved the best prediction performance.
\subsection{GRU}
A GRU deep learning model was implemented for sequential learning.
However, GRU produced poor performance because the dataset consisted of independent trip-level samples rather than temporal sequential data.

\section{Model Training}
All machine learning models were trained using:
\begin{itemize}
    \item Structured trip-level vehicular data
    \item Regression-based prediction
    \item Travel duration as target variable
\end{itemize}
The GRU model additionally used:
\begin{itemize}
    \item Min-Max normalization
    \item Sequential input windows
\end{itemize}

\section{Performance Evaluation Metrics}
The trained models were evaluated using:
\begin{table}[H]
\centering
\begin{tabular}{|c|c|}
\hline
\textbf{Metric} & \textbf{Purpose} \\
\hline
RMSE & Prediction error magnitude \\
MAE & Average absolute error \\
R\textsuperscript{2} Score & Goodness of fit \\
\hline
\end{tabular}
\caption{Evaluation Metrics}
\end{table}

\section{Mathematical Formulation}
\subsection{RMSE}
\begin{equation}
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y_i})^2}
\end{equation}
\subsection{MAE}
\begin{equation}
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y_i}|
\end{equation}
\subsection{R\textsuperscript{2} Score}
\begin{equation}
R^2 = 1 - \frac{\sum(y_i-\hat{y_i})^2}{\sum(y_i-\bar{y})^2}
\end{equation}
Where:
\begin{itemize}
    \item $y_i$ = Actual value
    \item $\hat{y_i}$ = Predicted value
    \item $\bar{y}$ = Mean value
\end{itemize}

\section{Experimental Results}
The final model comparison results are shown below.
\begin{table}[H]
\centering
\begin{tabular}{|c|c|c|c|}
\hline
\textbf{Model} & \textbf{RMSE} & \textbf{MAE} & \textbf{R\textsuperscript{2}} \\
\hline
CatBoost & 70.07 & 48.99 & 0.9903 \\
LightGBM & 73.41 & 51.81 & 0.9894 \\
Random Forest & 75.44 & 52.42 & 0.9888 \\
XGBoost & 78.70 & 55.14 & 0.9878 \\
Decision Tree & 105.99 & 73.44 & 0.9779 \\
GRU & 872.82 & 685.99 & 0.0351 \\
\hline
\end{tabular}
\caption{Final Model Comparison Results}
\end{table}
CatBoost achieved the best predictive performance among all implemented models.

\section{Visualization}
\subsection{Scatter Plot}
The scatter plot shows the relationship between actual and predicted travel time values, demonstrating the prediction accuracy of the CatBoost model.
\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{Figure_2.png}
\caption{Actual vs Predicted Travel Time}
\label{fig:scatter}
\end{figure}

\subsection{Box Plot}
The box plot visualizes the distribution of dataset features and helps identify potential outliers.
\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{box_plot.png}
\caption{Box Plot of Dataset Features}
\label{fig:boxplot}
\end{figure}

\section{System Model}
Consider a VEC network comprising a set of vehicles $\mathcal{V} = \{1, 2, ..., V\}$, a set of Roadside Units (RSUs) $\mathcal{R} = \{1, 2, ..., R\}$ deployed along the road, and a centralized Core Cloud. Each vehicle $v$ generates a computation-intensive task at time slot $t$, denoted by $T_{v,t} = \{D_{v,t}, C_{v,t}, s_{v,t}\}$, where $D_{v,t}$ is the input data size (in bits), $C_{v,t}$ is the required CPU cycles to process one bit of data, and $s_{v,t} \in \mathcal{S}$ is the specific service program (from a library of $S$ total services) required to execute the task.
\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{infra.png}
\caption{System illustration.}
\label{fig:system_illustration}
\end{figure}

Each RSU $r$ is equipped with an edge server having a finite cache capacity $K_r$, meaning it can cache a maximum of $K_r$ services simultaneously ($K_r \ll S$). If a vehicle offloads task $T_{v,t}$ to RSU $r$, and service $s_{v,t}$ is cached at $r$, it is a \textit{cache hit}, and computation begins immediately. If it is a \textit{cache miss}, the RSU must download service $s_{v,t}$ from the Core Cloud, incurring a significant service fetching delay before computation can commence.

The decision variable for vehicle $v$ consists of the offloading ratio $\alpha_{v,t} \in [0, 1]$ (proportion of task processed locally vs. offloaded to the edge/cloud) and the caching replacement decision vector $\boldsymbol{c}_{v,t}$, which determines which service to evict and cache at the target RSU.

\section{Service Caching and Task Offloading Model}
In the considered VEC environment, the computation of a task requires both the input data from the vehicle and the corresponding service program. Due to limited caching capacity, not all service programs can be cached everywhere. Depending on the caching decisions at the vehicle, the nearest edge server (RSU), and the neighboring edge pool, the task offloading and execution can be classified into six distinct cases:
\begin{itemize}
    \item \textbf{Case 1 (Local Execution):} The required service program is cached locally on the vehicle. The entire task is executed on the vehicle's local CPU without any offloading.
    \item \textbf{Case 2 (Partial Offloading to Nearest Edge):} The service is cached both locally and at the nearest edge server. The task is partitioned, with a ratio $(1-o_v)$ executed locally and a ratio $o_v$ offloaded to the nearest edge.
    \item \textbf{Case 3 (Complete Offloading to Nearest Edge):} The service is not cached locally but is available at the nearest edge server. The entire task is transmitted to and executed by the nearest edge server.
    \item \textbf{Case 4 (Partial Offloading to Edge Pool):} The task is completely offloaded to the nearest edge server. However, to balance the load, the edge server partially offloads a ratio $o_e$ of the task to a neighboring edge server in the edge pool, executing the remaining $(1-o_e)$ locally.
    \item \textbf{Case 5 (Complete Offloading to Edge Pool):} The required service is not cached at the nearest edge server but is available in the neighboring edge pool. The task is forwarded by the nearest edge to the neighboring edge server for execution.
    \item \textbf{Case 6 (Cloud Execution):} The required service is neither cached at the nearest edge server nor within the edge pool. The task must be offloaded to the remote Core Cloud for execution, incurring maximum transmission delay.
\end{itemize}

\section{Final Workflow Summary}
The overall workflow of the proposed mobility prediction and joint action decision system is illustrated below. The framework integrates SUMO-generated mobility traces, mobility prediction metrics (via CatBoost), and state extraction to feed into the MA-DDPG agent. Based on these observations, the agent outputs continuous caching placements and offloading ratios which correspond directly to the 6 execution cases.

\begin{figure}[H]
\centering
\begin{tikzpicture}[
    node distance=0.8cm and 0.5cm,
    process/.style={rectangle, draw=black, thick, fill=blue!10, text width=6cm, align=center, rounded corners, minimum height=1cm},
    decision/.style={rectangle, draw=black, thick, fill=green!10, text width=6cm, align=center, rounded corners, minimum height=1cm},
    cases/.style={rectangle, draw=black, thick, fill=orange!10, text width=3.5cm, align=center, rounded corners, minimum height=1.2cm},
    arrow/.style={thick, ->, >=stealth}
]

% Top pipeline
\node (dataset) [process] {1. Dataset \& Environment\\(SUMO Traffic + LEAF)};
\node (mobility) [process, below=of dataset] {2. Mobility Prediction\\(CatBoost, LightGBM, etc.)};
\node (state) [process, below=of mobility] {3. State Extraction\\(Bandwidth, Cache, Queue)};
\node (agent) [decision, below=of state] {4. Deep Learning Agent\\(MA-DDPG CTDE)};
\node (action) [decision, below=of agent] {5. Joint Action Decisions\\(Caching Placement \& Offloading Ratio)};

% An intermediate node for routing
\node (dispatch) [below=0.6cm of action, inner sep=0, minimum size=0] {};

% Row 1 of cases
\node (case2) [cases, below=1.2cm of action] {Case 2:\\Partial to Nearest Edge};
\node (case1) [cases, left=of case2] {Case 1:\\Local Execution};
\node (case3) [cases, right=of case2] {Case 3:\\Complete to Nearest Edge};

% Row 2 of cases
\node (case5) [cases, below=of case2] {Case 5:\\Complete to Edge Pool};
\node (case4) [cases, left=of case5] {Case 4:\\Partial to Edge Pool};
\node (case6) [cases, right=of case5] {Case 6:\\Remote Cloud Execution};

% Connect pipeline
\draw [arrow] (dataset) -- (mobility);
\draw [arrow] (mobility) -- (state);
\draw [arrow] (state) -- (agent);
\draw [arrow] (agent) -- (action);

% Connect to cases
\draw [thick] (action.south) -- (dispatch);
\draw [arrow] (dispatch) -| (case1.north);
\draw [arrow] (dispatch) -- (case2.north);
\draw [arrow] (dispatch) -| (case3.north);

\draw [arrow] (case1.south) -- (case4.north);
\draw [arrow] (case2.south) -- (case5.north);
\draw [arrow] (case3.south) -- (case6.north);

\end{tikzpicture}
\caption{Overall Workflow: From Dataset and Mobility Prediction to DL Agent and Task Offloading Cases}
\label{fig:offloading_flowchart_tikz}
\end{figure}

\section{Markov Decision Process (MDP) Formulation}
The joint optimization problem is formulated as a Multi-Agent Markov Decision Process (MMDP), defined by the tuple $\langle \mathcal{N}, \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$:

\subsection{Agents ($\mathcal{N}$)}
The set of agents $\mathcal{N}$ corresponds to the active vehicles $\mathcal{V}$ in the network that are generating tasks and making joint offloading and caching decisions.

\subsection{State Space ($\mathcal{S}$)}
At each time step $t$, the global state $s_t \in \mathcal{S}$ encompasses the environment information for all agents. For a specific vehicle $v$, its local observation $o_{v,t}$ includes:
\begin{itemize}
    \item The current task size $D_{v,t}$ and required service $s_{v,t}$.
    \item The current cache status matrix of its associated RSU $r$.
    \item The current computational queue length (load) of RSU $r$.
    \item The vehicle's local CPU queue length.
    \item The wireless channel state information (uplink bandwidth) to RSU $r$.
\end{itemize}

\subsection{Action Space ($\mathcal{A}$)}
The action $a_{v,t} \in \mathcal{A}$ taken by vehicle $v$ is continuous and includes:
\begin{itemize}
    \item \textbf{Offloading Decision:} $\alpha_{v,t} \in [0, 1]$, representing the ratio of the task to be offloaded.
    \item \textbf{Caching Decision:} A continuous vector $\boldsymbol{c}_{v,t}$ representing the priority or probability of caching specific services. The environment discretizes this continuous vector to make the final discrete cache replacement choice (e.g., evicting the lowest priority service to make room for $s_{v,t}$).
\end{itemize}

\subsection{State Transition Probability ($\mathcal{P}$)}
The state transition function $\mathcal{P}(s_{t+1} | s_t, \boldsymbol{a}_t)$ defines the probability of transitioning to the next global state $s_{t+1}$ given the current state $s_t$ and the joint actions $\boldsymbol{a}_t$ of all agents. This is determined dynamically by the physical mobility (SUMO) and network environment (LEAF).

\subsection{Reward Function ($\mathcal{R}$)}
The objective is to minimize the total system cost. Therefore, the reward $r_{v,t}$ is defined as the negative of the weighted sum of delay and energy:
\begin{equation}
    Cost_{v,t} = \beta_d \cdot Delay_{v,t} + \beta_e \cdot Energy_{v,t} + \lambda \cdot Penalty_{conflict}
\end{equation}
\begin{equation}
    r_{v,t} = - Cost_{v,t}
\end{equation}
Where $\beta_d$ and $\beta_e$ are weighting factors for delay and energy. Crucially, we introduce $Penalty_{conflict}$, a penalty applied if multiple vehicles associated with the same RSU request identical caching actions, thereby explicitly discouraging redundant caching and herding behavior.

\subsection{Discount Factor ($\gamma$)}
The discount factor $\gamma \in [0, 1]$ determines the importance of future rewards. A value closer to 1 encourages agents to optimize for long-term system stability, while a value closer to 0 makes agents myopic, focusing only on immediate cost reductions.

\section{DRL (MA-DDPG Scheduling) and Communication}
To solve this continuous-action MMDP, we employ the Multi-Agent Deep Deterministic Policy Gradient (MA-DDPG) algorithm.

\subsection{Centralized Training, Decentralized Execution (CTDE)}
MA-DDPG utilizes the CTDE paradigm. 
\begin{itemize}
    \item \textbf{Critic Network (Centralized):} During the training phase in the simulator, a centralized Critic network is used. It takes as input the \textit{global state} $s_t$ (observations from all vehicles) and the \textit{joint action} $\boldsymbol{a}_t = \{a_{1,t}, ..., a_{V,t}\}$ of all vehicles to compute a global Q-value, $Q^{\pi}(s_t, a_{1,t}, ..., a_{V,t})$. This provides a holistic evaluation of how the combined actions impact the entire network, stabilizing training in the non-stationary multi-agent environment.
    \item \textbf{Actor Network (Decentralized):} During the execution phase (deployment), each vehicle $v$ utilizes its own local Actor network, $\mu_{\theta_v}(o_{v,t})$. The Actor takes only local observations $o_{v,t}$ to output its specific continuous action $a_{v,t}$. Thus, execution remains fully decentralized and scalable.
\end{itemize}

\subsection{Explicit Communication Protocol (MA-DDPG-Comm)}
To further enhance cooperation and prevent conflicts, we augment MA-DDPG with a CommNet-style protocol.
\begin{enumerate}
    \item \textbf{Message Encoding:} Each vehicle uses a neural network layer (`MessageEncoder`) to compress its current observation and intended action into a continuous, 16-dimensional message vector, $m_{v,t}$.
    \item \textbf{Message Broadcasting:} Vehicles broadcast their message vectors to neighboring vehicles within the same RSU coverage area.
    \item \textbf{Message Pooling and Concatenation:} Vehicle $v$ receives messages from its neighbors, aggregates them using mean-pooling to create a neighborhood context vector $\bar{m}_{v,t}$, and concatenates this with its own local observation $o_{v,t}$.
    \item \textbf{Decision Making:} The Actor network makes its final offloading and caching decision based on the concatenated input $[o_{v,t}, \bar{m}_{v,t}]$. This allows Vehicle A to know that Vehicle B intends to cache Service 1, prompting Vehicle A to cache Service 2 instead, maximizing cache diversity.
\end{enumerate}

\section{Baselines and Schedulers (Workflow)}
To rigorously benchmark the proposed MA-DDPG approach, the following schedulers are implemented in the simulation workflow:
\begin{itemize}
    \item \textbf{Proposed MA-DDPG:} The multi-agent CTDE framework with explicit communication described above.
    \item \textbf{Proposed DDPG:} A standard, independent single-agent DDPG model without communication or global critic, representing prior state-of-the-art learning approaches.
    \item \textbf{EnergyMin:} A greedy heuristic that calculates the required energy for local, edge, and cloud execution and selects the destination that strictly minimizes energy consumption.
    \item \textbf{LatencyMin:} A greedy heuristic that selects the execution destination strictly offering the lowest immediate processing delay.
    \item \textbf{LRU (Least Recently Used):} A traditional cache replacement strategy where the RSU evicts the service that has not been requested for the longest time to make room for new services.
    \item \textbf{NoCache:} A baseline representing traditional Mobile Cloud Computing. No services are cached at the edge. All non-local computations are forcefully offloaded to the remote core cloud, incurring maximum transmission delay.
\end{itemize}

The simulation workflow follows a repeating cycle of DP (Data Processing of states), MP (Message Passing between agents), and DRC (Decision execution and Reward Calculation) within the LEAF environment.

"""

start_idx = 237
end_idx = 395

new_lines = lines[:start_idx] + [new_methodology] + ["\n"] + lines[end_idx:]
with open('/home/vamsi/Desktop/leaf/examples/vec_caching/thesis_final.tex', 'w') as f:
    f.writelines(new_lines)
print("Replacement successful")
