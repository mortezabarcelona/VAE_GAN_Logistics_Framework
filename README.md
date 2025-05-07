# VAE-GAN Logistics Framework

This project implements a hybrid **VAE-GAN (Variational Autoencoder - Generative Adversarial Network)** architecture for intelligent logistics scenario generation, simulation, and optimization. The framework is designed to model and optimize dynamic, multi-modal supply chains under realistic conditions such as disruptions, sustainability constraints, and high demand scenarios.

---

## Project Structure

```bash
VAE_GAN_Logistics_Framework/
├── .gitignore                       # Git ignore rules for untracked files & directories
├── .idea/                          # IDE configuration (e.g., PyCharm)
├── api/                            # API layer (Flask) for serving predictions and scenarios
│   ├── __init__.py                 
│   └── endpoints.py                # API endpoints (e.g., `/predict`)
├── generate_tree.py                # Utility script to generate the directory tree
├── notebooks/                      # Jupyter notebooks for EDA, benchmarking, and evaluation
│   ├── Benchmark_Analysis.ipynb
│   ├── EDA.ipynb
│   ├── Model_Evaluation.ipynb
│   └── simulation_analysis.ipynb
├── README.md                       # Project documentation (this file)
├── requirements.txt                # Python dependencies
├── simulation/                     # Simulation engine and scenarios
│   ├── __init__.py
│   ├── feedback_loop.py            # Implements feedback loop for dynamic simulations
│   ├── live_api_simulation.py      # Simulates live predictions from the VAE-GAN model
│   ├── resilience_testing.py       # Tests model resilience under various conditions
│   ├── simulation_engine.py        # Core simulation runner
│   └── simulation_scenarios.py     # Contains different simulation scenarios
└── src/                            # Source code
    ├── __init__.py
    ├── config.py                   # Centralized configuration settings
    ├── dashboard/                  # Real-time dashboard application
    │   └── dashboard_app.py
    ├── data/                       # Data handling functions
    │   ├── __init__.py
    │   ├── loader.py               # Functions to load data
    │   └── preprocess.py           # Data preprocessing utilities
    ├── evaluation/                 # Model evaluation and benchmarking modules
    │   ├── __init__.py
    │   ├── benchmarking.py         # Benchmark analysis
    │   ├── clustering_analysis.py  # (Optional) Clustering analysis of outputs/data
    │   ├── metrics.py              # Additional performance metrics
    │   ├── reconstruction_metrics.py  # Metrics for VAE reconstruction quality
    │   └── visualize.py            # Visualization routines for evaluation results
    ├── models/                     # Model architectures
    │   ├── __init__.py
    │   ├── optimization/           # Optimization models based on VAE, GAN, and VAE-GAN
    │   │   ├── __init__.py
    │   │   ├── gan.py              # Generative Adversarial Network implementation
    │   │   ├── vae.py              # Variational Autoencoder implementation
    │   │   └── vae_gan.py          # Integrated VAE-GAN model harnessing both VAE and GAN approaches
    │   ├── placeholders/           # Placeholder modules for future extensions
    │   │   ├── __init__.py
    │   │   ├── dynamic_routing.py  # Placeholder for dynamic routing algorithms
    │   │   └── freight_exchange.py # Placeholder for freight exchange modeling
    │   └── trained_model.pt        # Trained model weights (binary file)
    ├── synthetic_data/             # Synthetic data generation and processed datasets
    │   ├── __init__.py
    │   ├── data/                   # Contains CSV files with synthetic logistics data
    │   │   └── processed/
    │   │       ├── synthetic_logistics_data_baseline.csv
    │   │       ├── synthetic_logistics_data_cleaned.csv
    │   │       ├── synthetic_logistics_data_disruption.csv
    │   │       ├── synthetic_logistics_data_high_demand.csv
    │   │       └── synthetic_logistics_data_sustainability.csv
    │   └── generate_synthetic.py   # Script to generate synthetic logistics data
    └── utils/                      # Utility functions and helper modules
        ├── __init__.py
        ├── checkpoint.py           # Functions for saving and loading model checkpoints
        ├── logger.py               # Centralized logging utilities
        └── loss_functions.py       # Custom loss functions for model training
