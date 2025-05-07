import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value


# Fallback synthetic data generator (if your module import fails)
def generate_logistics_data():
    warehouses = [0, 1]
    markets = [0, 1, 2]
    supply = [50, 60]
    demand = [30, 40, 40]
    costs = [
        [2, 3, 1],
        [5, 4, 2]
    ]
    return warehouses, markets, supply, demand, costs


def solve_milp_transportation(warehouses, markets, supply, demand, costs):
    """
    Basic MILP formulation considering only the base transportation cost.
    """
    prob = LpProblem("Transportation_Problem", LpMinimize)
    num_warehouses = len(warehouses)
    num_markets = len(markets)

    # Decision variables: shipment amounts from warehouse i to market j.
    x = LpVariable.dicts("Route",
                         (range(num_warehouses), range(num_markets)),
                         lowBound=0,
                         cat="Continuous")

    # Objective: Minimize base transportation cost.
    prob += lpSum([costs[i][j] * x[i][j]
                   for i in range(num_warehouses)
                   for j in range(num_markets)])

    # Supply constraints.
    for i in range(num_warehouses):
        prob += lpSum([x[i][j] for j in range(num_markets)]) <= supply[i]

    # Demand constraints.
    for j in range(num_markets):
        prob += lpSum([x[i][j] for i in range(num_warehouses)]) >= demand[j]

    prob.solve()
    objective_value = value(prob.objective)
    allocation = {(i, j): x[i][j].varValue for i in range(num_warehouses) for j in range(num_markets)}
    return objective_value, allocation


def heuristic_least_cost(warehouses, markets, supply, demand, costs):
    """
    Basic greedy heuristic allocation based solely on the base cost.
    """
    supply_remaining = supply.copy()
    demand_remaining = demand.copy()
    num_warehouses = len(warehouses)
    num_markets = len(markets)
    allocation = np.zeros((num_warehouses, num_markets))

    # Sort the routes by the base cost (lowest cost first).
    routes = sorted([(i, j) for i in range(num_warehouses) for j in range(num_markets)],
                    key=lambda x: costs[x[0]][x[1]])

    for (i, j) in routes:
        amount = min(supply_remaining[i], demand_remaining[j])
        allocation[i, j] = amount
        supply_remaining[i] -= amount
        demand_remaining[j] -= amount

    total_cost = np.sum(allocation * np.array(costs))
    return total_cost, allocation


def simulate_vae_gan_basic(warehouses, markets, supply, demand, costs):
    """
    Simulated VAE-GAN for the basic scenario.
    (In a real implementation, replace this simulation with your actual API call.)
    """
    # In this basic simulation, we use the same heuristic allocation as a proxy,
    # then add a small noise to simulate prediction uncertainty.
    _, allocation = heuristic_least_cost(warehouses, markets, supply, demand, costs)
    base_cost = np.sum(allocation * np.array(costs))
    predicted_cost = base_cost + np.random.normal(0, 2)
    return predicted_cost


if __name__ == "__main__":
    # Retrieve synthetic/clean logistics data.
    warehouses, markets, supply, demand, costs = generate_logistics_data()

    # Calculate basic models:
    milp_cost, milp_allocation = solve_milp_transportation(warehouses, markets, supply, demand, costs)
    heuristic_cost, heuristic_allocation = heuristic_least_cost(warehouses, markets, supply, demand, costs)
    simulated_vae_gan_cost = simulate_vae_gan_basic(warehouses, markets, supply, demand, costs)

    print("Benchmarking Results (Basic Models Only):")
    print("-----------------------------------------")
    print("Original MILP Optimal Cost:  ", milp_cost)
    print("Heuristic (Basic) Cost:        ", heuristic_cost)
    print("Simulated VAE-GAN (Basic) Cost:", simulated_vae_gan_cost)
