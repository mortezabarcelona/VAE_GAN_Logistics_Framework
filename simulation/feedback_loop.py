import time
import numpy as np
import logging

# For demonstration, we re-use our simulated live data and heuristic-based prediction functions.
# In your project, these should be imported from the appropriate modules.
def fetch_live_data():
    """
    Simulate fetching live data.
    In a deployed system, this would call an API.
    """
    base_supply = [50, 60]
    base_demand = [30, 40, 40]
    base_costs = [
        [2, 3, 1],
        [5, 4, 2]
    ]
    supply = [max(1, s + np.random.randint(-5, 6)) for s in base_supply]
    demand = [max(1, d + np.random.randint(-5, 6)) for d in base_demand]
    costs = [[max(0.1, c + np.random.uniform(-0.5, 0.5)) for c in row] for row in base_costs]
    return supply, demand, costs

def heuristic_least_cost(warehouses, markets, supply, demand, costs):
    """
    Basic greedy heuristic to compute a cost as a proxy for our prediction.
    """
    supply_remaining = supply.copy()
    demand_remaining = demand.copy()
    allocation = np.zeros((len(warehouses), len(markets)))
    routes = sorted([(i, j) for i in range(len(warehouses)) for j in range(len(markets))],
                    key=lambda x: costs[x[0]][x[1]])
    for (i, j) in routes:
        amount = min(supply_remaining[i], demand_remaining[j])
        allocation[i][j] = amount
        supply_remaining[i] -= amount
        demand_remaining[j] -= amount
    total_cost = np.sum(allocation * np.array(costs))
    return total_cost

def simulate_vae_gan_dynamic():
    """
    Simulates the VAE‑GAN prediction under dynamic conditions.
    In practice, this would run inference using your trained model.
    Here we use the heuristic plus noise as a proxy.
    """
    warehouses = [0, 1]
    markets = [0, 1, 2]
    supply, demand, costs = fetch_live_data()
    base_cost = heuristic_least_cost(warehouses, markets, supply, demand, costs)
    predicted_cost = base_cost + np.random.normal(0, 0.5)
    return predicted_cost

def simulate_actual_outcome(predicted_cost):
    """
    Simulates the actual cost outcome.
    In a real system, this might come from a sensor or a result database.
    For now, we simulate it by adding some variation to the predicted cost.
    """
    # For instance, actual outcome might be typically a bit higher or lower than predicted.
    actual_cost = predicted_cost + np.random.normal(0, 1.0)
    return actual_cost

def run_feedback_loop(num_iterations=50, threshold=5.0):
    """
    Runs the feedback loop that compares predicted costs with actual outcomes.
    Logs the prediction error and flags if the error exceeds a threshold.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    errors = []

    for i in range(1, num_iterations + 1):
        # Simulate fetching live data and making a prediction
        predicted_cost = simulate_vae_gan_dynamic()
        # Simulate receiving the actual outcome corresponding to this prediction
        actual_cost = simulate_actual_outcome(predicted_cost)
        # Compute error
        error = abs(actual_cost - predicted_cost)
        errors.append(error)

        # Log the current iteration results
        logging.info(f"Iteration {i:03d}: Predicted Cost = {predicted_cost:.2f}, "
                     f"Actual Cost = {actual_cost:.2f}, Error = {error:.2f}")

        # If error exceeds the threshold, log a warning.
        if error > threshold:
            logging.warning(f"Iteration {i:03d}: High prediction error detected: {error:.2f} > {threshold}")

        time.sleep(1)  # Simulate a delay between updates

    # After all iterations, output summary statistics.
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    logging.info(f"Feedback Loop Summary: Mean Error = {mean_error:.2f} ± {std_error:.2f}")

if __name__ == "__main__":
    run_feedback_loop(num_iterations=50, threshold=5.0)
