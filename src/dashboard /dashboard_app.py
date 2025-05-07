import streamlit as st
import time
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------------
# Functions to simulate live data and VAE‑GAN prediction.
# These are placeholders; in an actual deployment, you would replace these with
# imports from your API integration module and use your trained model for inference.
# ---------------------------------------------------------------------------------

def fetch_live_data():
    """
    Simulates fetching live data from an API.
    Generates perturbed values for supply, demand, and costs.
    """
    base_supply = [50, 60]
    base_demand = [30, 40, 40]
    base_costs = [
        [2, 3, 1],
        [5, 4, 2]
    ]
    # Introduce small random perturbations
    supply = [max(1, s + np.random.randint(-5, 6)) for s in base_supply]
    demand = [max(1, d + np.random.randint(-5, 6)) for d in base_demand]
    costs = [[max(0.1, c + np.random.uniform(-0.5, 0.5)) for c in row] for row in base_costs]
    return supply, demand, costs


def heuristic_least_cost(warehouses, markets, supply, demand, costs):
    """
    A basic greedy heuristic to compute cost as a proxy.
    This is used to simulate baseline behavior.
    """
    supply_remaining = supply.copy()
    demand_remaining = demand.copy()
    allocation = np.zeros((len(warehouses), len(markets)))

    # Sort routes based on cost (lowest cost first)
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
    Simulates the VAE‑GAN output under dynamic live conditions.
    In practice, the trained VAE‑GAN model (loaded from your .pt file) would process
    live inputs and generate predictions. Here, we simulate this by using the heuristic
    and adding small noise.
    """
    # Define warehouses and markets for the simulation.
    warehouses = [0, 1]
    markets = [0, 1, 2]
    supply, demand, costs = fetch_live_data()
    base_cost = heuristic_least_cost(warehouses, markets, supply, demand, costs)
    predicted_cost = base_cost + np.random.normal(0, 0.5)
    return predicted_cost


# ---------------------------------------------------------------------------------
# Streamlit Dashboard Code
# ---------------------------------------------------------------------------------

st.set_page_config(page_title="VAE‑GAN Live Dashboard", layout="wide")
st.title("Live VAE‑GAN Dashboard for Logistics Optimization")
st.write("""
This dashboard demonstrates the real-time predictions of our VAE‑GAN model,
which has been trained on synthetic logistics data and is designed for dynamic,
real-time applications. The VAE‑GAN continuously updates its cost predictions as 
live data (simulated here) is received.
""")

# Create placeholders for the current predicted cost and the live chart.
cost_placeholder = st.empty()
chart_placeholder = st.empty()

# Data storage: We will collect live predictions over time to plot a time series.
predicted_costs = []
iterations = []

# Number of iterations (or, essentially, seconds of monitoring)
num_iterations = 50
for i in range(1, num_iterations + 1):
    predicted_cost = simulate_vae_gan_dynamic()
    predicted_costs.append(predicted_cost)
    iterations.append(i)

    # Update text output with current prediction.
    cost_placeholder.markdown(f"### Current VAE‑GAN Predicted Cost: {predicted_cost:.2f}")

    # Update the chart: Plot the history of predictions.
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(iterations, predicted_costs, marker='o', linestyle='-', color="blue", label="Predicted Cost")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")
    ax.set_title("Live Monitoring: VAE‑GAN Cost Predictions")
    ax.legend()
    ax.grid(True)
    chart_placeholder.pyplot(fig)

    time.sleep(1)  # Simulate a delay between live data updates.

st.write("Monitoring complete.")
