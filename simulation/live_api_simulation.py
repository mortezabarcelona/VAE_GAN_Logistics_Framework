import time
import numpy as np


# For the purpose of this simulation, we re-use the base data generator.
def fetch_live_data():
    """
    Simulates fetching live data from an API.
    This function perturbs base logistics parameters (supply, demand, cost) to mimic real-time updates.
    """
    base_supply = [50, 60]
    base_demand = [30, 40, 40]
    base_costs = [
        [2, 3, 1],
        [5, 4, 2]
    ]

    # Introduce random perturbations to simulate live changes
    supply = [max(1, s + np.random.randint(-5, 6)) for s in base_supply]
    demand = [max(1, d + np.random.randint(-5, 6)) for d in base_demand]
    costs = [[max(0.1, c + np.random.uniform(-0.5, 0.5)) for c in row] for row in base_costs]

    return supply, demand, costs


def heuristic_least_cost(warehouses, markets, supply, demand, costs):
    """
    Basic heuristic allocation (used here as a proxy for VAE‑GAN static behavior).
    """
    supply_remaining = supply.copy()
    demand_remaining = demand.copy()
    num_warehouses = len(warehouses)
    num_markets = len(markets)
    allocation = np.zeros((num_warehouses, num_markets))

    # Sort routes based on base cost (lowest cost first)
    routes = sorted([(i, j) for i in range(num_warehouses) for j in range(num_markets)],
                    key=lambda x: costs[x[0]][x[1]])

    for (i, j) in routes:
        amount = min(supply_remaining[i], demand_remaining[j])
        allocation[i][j] = amount
        supply_remaining[i] -= amount
        demand_remaining[j] -= amount

    return np.sum(allocation * np.array(costs))


def simulate_vae_gan_dynamic(warehouses, markets, supply, demand, costs):
    """
    Simulated VAE‑GAN for dynamic conditions.
    In a live application, this function would be replaced by a VAE‑GAN model that continuously updates its prediction via an API.
    Here we use the basic heuristic as a proxy and add a small noise factor.
    """
    base_cost = heuristic_least_cost(warehouses, markets, supply, demand, costs)
    predicted_cost = base_cost + np.random.normal(0, 0.5)
    return predicted_cost


def process_live_data():
    """
    Simulates continuous real-time data updates by repeatedly fetching live data and processing it with the VAE‑GAN.
    """
    warehouses = [0, 1]
    markets = [0, 1, 2]

    iteration = 1
    while True:
        supply, demand, costs = fetch_live_data()
        predicted_cost = simulate_vae_gan_dynamic(warehouses, markets, supply, demand, costs)
        print(f"Iteration {iteration:03d} - Live VAE‑GAN Predicted Cost: {predicted_cost:.2f}")

        # In a real scenario, you might push this output to a dashboard or log.
        time.sleep(1)  # Simulate a 1-second interval between API updates.
        iteration += 1


if __name__ == "__main__":
    process_live_data()
