import os, json
import matplotlib; matplotlib.use("Agg")

# Force PyTorch to use 1 thread to prevent catastrophic lock contention
os.environ["OMP_NUM_THREADS"] = "1"

from examples.vec_caching.main import (
    build_infrastructure, make_mobility
)
from examples.vec_caching.baselines import (
    run_latencymin_ddpg_episode, run_energymin_ddpg_episode, run_nocache_episode
)
from examples.vec_caching.ddpg_agent import DDPGAgent, STATE_DIM, ACTION_DIM
from examples.vec_caching.settings import REPLAY_SIZE, USE_SUMO
from examples.vec_caching.orchestrator import VECOrchestrator


def _save(name, delays, energies):
    os.makedirs("examples/vec_caching/results", exist_ok=True)
    with open(f"examples/vec_caching/results/metrics_{name}.json", "w") as f:
        json.dump({"delays": delays, "energies": energies}, f)

def run_all(episodes=600):
    print(f"--- RUNNING ALL 4 SCHEMES FOR {episodes} EPISODES ---\n")

    # 1. NO CACHE
    print("1. Running NoCache...")
    inf, veh, edg, cld = build_infrastructure()
    mob = make_mobility(veh, edg)
    delays, energies = [], []
    for ep in range(episodes):
        d, e = run_nocache_episode(veh, edg, cld, mob)
        delays.append(sum(d))
        energies.append(sum(e))
        if ep % 50 == 0: print(f"  Ep {ep}: Delay {delays[-1]:.2f}")
    _save("NoCache", delays, energies)

    # 2. PROPOSED DDPG (NO WARM START)
    print("\n2. Running Proposed DDPG (No Warm Start)...")
    inf, veh, edg, cld = build_infrastructure()
    mob = make_mobility(veh, edg)
    agent = DDPGAgent(STATE_DIM, ACTION_DIM, REPLAY_SIZE)
    orch = VECOrchestrator(inf, veh, edg, cld, mob, agent, training=True, warm_start_cache=False)
    delays, energies = [], []
    for ep in range(episodes):
        d, e = orch.run_episode()
        delays.append(sum(d))
        energies.append(sum(e))
        agent.decay_noise()
        if ep % 50 == 0: print(f"  Ep {ep}: Delay {delays[-1]:.2f}")
    _save("Proposed", delays, energies)

    # 3. LATENCY MIN
    print("\n3. Running LatencyMin...")
    inf, veh, edg, cld = build_infrastructure()
    mob = make_mobility(veh, edg)
    agent = DDPGAgent(STATE_DIM, ACTION_DIM, REPLAY_SIZE)
    delays, energies = [], []
    for ep in range(episodes):
        d, e = run_latencymin_ddpg_episode(veh, edg, cld, mob, agent)
        delays.append(sum(d))
        energies.append(sum(e))
        agent.decay_noise()
        if ep % 50 == 0: print(f"  Ep {ep}: Delay {delays[-1]:.2f}")
    _save("LatencyMin", delays, energies)

    # 4. ENERGY MIN
    print("\n4. Running EnergyMin...")
    inf, veh, edg, cld = build_infrastructure()
    mob = make_mobility(veh, edg)
    agent = DDPGAgent(STATE_DIM, ACTION_DIM, REPLAY_SIZE)
    delays, energies = [], []
    for ep in range(episodes):
        d, e = run_energymin_ddpg_episode(veh, edg, cld, mob, agent)
        delays.append(sum(d))
        energies.append(sum(e))
        agent.decay_noise()
        if ep % 50 == 0: print(f"  Ep {ep}: Delay {delays[-1]:.2f}")
    _save("EnergyMin", delays, energies)

    from examples.vec_caching.plot_results import plot_delay, plot_energy
    
    # Generate the 600-episode plots
    plot_delay("examples/vec_caching/results")
    plot_energy("examples/vec_caching/results")

if __name__ == "__main__":
    run_all()

