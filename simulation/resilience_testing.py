import time
import numpy as np
import matplotlib.pyplot as plt


def generate_base_logistics_data():
    """
    Generates the base logistics data for the transportation problem.
    """
    warehouses = [0, 1]
    markets = [0, 1, 2]
    supply = [50, 60]  # Base supply
    demand = [30, 40, 40]  # Base demand
    costs = [
        [2, 3, 1],
        [5, 4, 2]
    ]
    return warehouses, markets, supply, demand, costs


def heuristic_least_cost(warehouses, markets, supply, demand, costs):
    """
    Basic greedy heuristic allocation. (Used as a proxy for VAE‑GAN in the static baseline.)
    """
    supply_remaining = supply.copy()
    demand_remaining = demand.copy()
    num_warehouses = len(warehouses)
    num_markets = len(markets)
    allocation = np.zeros((num_warehouses, num_markets))

    routes = sorted([(i, j) for i in range(num_warehouses) for j in range(num_markets)],
                    key=lambda x: costs[x[0]][x[1]])

    for i, j in routes:
        amount = min(supply_remaining[i], demand_remaining[j])
        allocation[i, j] = amount
        supply_remaining[i] -= amount
        demand_remaining[j] -= amount

    total_cost = np.sum(allocation * np.array(costs))
    return total_cost


def simulate_vae_gan_dynamic(warehouses, markets, supply, demand, costs):
    """
    Simulated VAE‑GAN for dynamic conditions.
    In a real scenario, this function would receive live data via an API and continuously update predictions.
    Here, we use the heuristic solution as a baseline and add a small noise to simulate dynamic prediction uncertainty.
    """
    base_cost = heuristic_least_cost(warehouses, markets, supply, demand, costs)
    predicted_cost = base_cost + np.random.normal(0, 0.5)
    return predicted_cost


def dynamic_resilience_simulation_vae(num_iterations=50):
    """
    Simulates dynamic conditions by perturbing inputs.
    This simulation focuses solely on evaluating the VAE‑GAN's responsiveness.
    Returns cost and computation time lists for the simulated VAE‑GAN.
    """
    warehouses, markets, base_supply, base_demand, base_costs = generate_base_logistics_data()

    vae_gan_costs = []
    vae_gan_times = []

    for _ in range(num_iterations):
        # Perturbations to simulate dynamic changes:
        supply = [max(1, s + np.random.randint(-5, 6)) for s in base_supply]
        demand = [max(1, d + np.random.randint(-5, 6)) for d in base_demand]
        costs = [[max(0.1, c + np.random.uniform(-0.5, 0.5)) for c in row] for row in base_costs]

        # Time the simulated VAE‑GAN prediction.
        start = time.time()
        vae_gan_cost = simulate_vae_gan_dynamic(warehouses, markets, supply, demand, costs)
        vae_gan_time = time.time() - start

        vae_gan_costs.append(vae_gan_cost)
        vae_gan_times.append(vae_gan_time)

    return {"VAE-GAN": vae_gan_costs, "VAE-GAN_time": vae_gan_times}


if __name__ == "__main__":
    results = dynamic_resilience_simulation_vae(num_iterations=50)

    vae_gan_cost_mean = np.mean(results["VAE-GAN"])
    vae_gan_cost_std = np.std(results["VAE-GAN"])
    vae_gan_time_mean = np.mean(results["VAE-GAN_time"])
    vae_gan_time_std = np.std(results["VAE-GAN_time"])

    print("Dynamic Resilience Simulation (VAE‑GAN Only):")
    print("------------------------------------------------")
    print("VAE‑GAN Cost: {:.2f} ± {:.2f}".format(vae_gan_cost_mean, vae_gan_cost_std))
    print("VAE‑GAN Time: {:.6f}s ± {:.6f}s".format(vae_gan_time_mean, vae_gan_time_std))

    # Plot cost evolution.
    iterations = np.arange(1, 51)
    plt.figure(figsize=(12, 7))
    plt.plot(iterations, results["VAE-GAN"], label="Simulated VAE‑GAN", marker="^", linestyle='-')
    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Cost", fontsize=14)
    plt.title("Dynamic Resilience Simulation: VAE‑GAN Cost Evolution", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

    # Plot computation times.
    plt.figure(figsize=(12, 7))
    plt.plot(iterations, results["VAE-GAN_time"], label="VAE‑GAN Time (s)", marker="^", linestyle='-')
    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Computation Time (s)", fontsize=14)
    plt.title("Dynamic Resilience Simulation: VAE‑GAN Computation Time", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()
