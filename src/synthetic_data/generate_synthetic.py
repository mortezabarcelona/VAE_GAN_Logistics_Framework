import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
import os


def generate_synthetic_data():
    """
    Generate a dynamic, realistic synthetic logistics dataset that captures:
      - Static attributes: shipment volume, distance, cost, emissions, etc.
      - Dynamic attributes: weather, traffic congestion, disruption events,
        demand fluctuations, capacity and port congestion variations.

    This dataset is generated for multiple scenarios ('baseline', 'disruption',
    'sustainability', and 'high_demand') and saved as separate CSV files in the
    data/processed/ directory.
    """
    # ~~~~~~~~~~~~~~~~~~~~~~~#
    # 1. Configuration Setup  #
    # ~~~~~~~~~~~~~~~~~~~~~~~#

    # Static configuration parameters
    TIME_HORIZON_DAYS = 30
    INTERVALS_PER_DAY = 24  # hourly intervals
    TOTAL_INTERVALS = TIME_HORIZON_DAYS * INTERVALS_PER_DAY

    CITIES = ['Barcelona', 'Rotterdam', 'Hamburg', 'Lisbon', 'Madrid']

    # Each transport mode has associated transit time ranges (in hours), cost factors, and emission factors.
    TRANSPORT_MODES = {
        'road': {'transit_time': (24, 240), 'cost_factor': 0.25, 'emission_factor': 0.07},
        'rail': {'transit_time': (48, 360), 'cost_factor': 0.15, 'emission_factor': 0.05},
        'sea': {'transit_time': (240, 1080), 'cost_factor': 0.10, 'emission_factor': 0.03},
        'air': {'transit_time': (12, 72), 'cost_factor': 1.0, 'emission_factor': 0.1}
    }

    BASE_VOLUME_MEAN = 10  # average shipment volume (tons)
    BASE_VOLUME_STD = 3  # standard deviation for volume
    DISTANCE_RANGE = (50, 1000)  # distance between origin and destination in km

    # Dynamic modifiers and environmental parameters
    WEATHER_CONDITIONS = ['clear', 'rain', 'storm', 'fog']
    WEATHER_SEVERITY = {'clear': 0.0, 'rain': 0.2, 'storm': 0.5, 'fog': 0.3}
    TRAFFIC_CONGESTION_RANGE = (0.8, 1.2)
    DISRUPTION_PROBABILITY = 0.1
    DISRUPTION_IMPACT = {'transit_time': 1.2, 'cost': 1.5, 'emissions': 1.1}
    DEMAND_MULTIPLIER_RANGE = (0.8, 1.2)
    CAPACITY_MODIFIER_RANGE = (0.9, 1.1)
    PORT_CONGESTION_RANGE = (0.9, 1.3)

    # Scenario-specific adjustments
    SCENARIO_ADJUSTMENTS = {
        'baseline': {'num_shipments_per_interval': 100},
        'disruption': {'num_shipments_per_interval': 100, 'disruption_probability': 0.3},
        'sustainability': {'num_shipments_per_interval': 100, 'emission_penalty': 2.0},
        'high_demand': {'num_shipments_per_interval': 150}
    }

    RANDOM_SEED = 42

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 2. Set Up Reproducibility   #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~#
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 3. Prepare Output Directory #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~#
    output_dir = os.path.join('data', 'processed')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 4. Generate Time Steps     #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~#
    start_datetime = datetime.now()
    time_stamps = [start_datetime + timedelta(hours=i) for i in range(TOTAL_INTERVALS)]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 5. Generate Data Records  #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # Dictionary to store records per scenario
    scenario_records = {scenario: [] for scenario in SCENARIO_ADJUSTMENTS.keys()}
    shipment_id = 1

    for ts in time_stamps:
        # Temporal attributes
        day_of_week = ts.strftime('%A')
        hour_of_day = ts.hour

        # Generate dynamic modifiers for this time step:
        weather = random.choice(WEATHER_CONDITIONS)
        weather_severity = WEATHER_SEVERITY[weather]
        traffic_congestion = np.random.uniform(*TRAFFIC_CONGESTION_RANGE)
        # Base disruption indicator (will be re-sampled per scenario)
        base_disruption = 1 if random.random() < DISRUPTION_PROBABILITY else 0
        demand_multiplier = np.random.uniform(*DEMAND_MULTIPLIER_RANGE)
        capacity_modifier = np.random.uniform(*CAPACITY_MODIFIER_RANGE)
        port_congestion = np.random.uniform(*PORT_CONGESTION_RANGE)

        # For each scenario, generate shipments for the current time step
        for scenario, settings in SCENARIO_ADJUSTMENTS.items():
            # Override disruption probability if specified for the scenario
            scenario_disruption_prob = settings.get('disruption_probability', DISRUPTION_PROBABILITY)
            # Re-sample disruption indicator based on scenario probability
            disruption_indicator = 1 if random.random() < scenario_disruption_prob else 0

            num_shipments = settings.get('num_shipments_per_interval')

            for _ in range(num_shipments):
                # Randomly select origin and destination (ensuring they differ)
                origin = random.choice(CITIES)
                destination = random.choice([c for c in CITIES if c != origin])

                # Randomly select a transport mode and retrieve its parameters
                transport_mode = random.choice(list(TRANSPORT_MODES.keys()))
                mode_params = TRANSPORT_MODES[transport_mode]

                # Generate static attributes
                volume = abs(np.random.normal(BASE_VOLUME_MEAN, BASE_VOLUME_STD))
                distance = np.random.uniform(*DISTANCE_RANGE)
                base_transit_time = np.random.uniform(*mode_params['transit_time'])
                base_cost = volume * distance * mode_params['cost_factor']
                base_emissions = volume * distance * mode_params['emission_factor']

                # Apply dynamic modifiers:
                # Adjust transit time by factors: weather, traffic, and if disrupted.
                transit_time = base_transit_time * (1 + weather_severity) * traffic_congestion
                if disruption_indicator:
                    transit_time *= DISRUPTION_IMPACT['transit_time']

                # Adjust cost; if disruption then apply corresponding impact.
                cost = base_cost
                if disruption_indicator:
                    cost *= DISRUPTION_IMPACT['cost']
                # Apply emission penalty in sustainability scenario if defined.
                if scenario == 'sustainability':
                    cost *= settings.get('emission_penalty', 1)

                # Adjust emissions similarly.
                emissions = base_emissions * (1 + weather_severity) * traffic_congestion
                if disruption_indicator:
                    emissions *= DISRUPTION_IMPACT['emissions']
                if scenario == 'sustainability':
                    emissions *= settings.get('emission_penalty', 1)

                # Compile a record with all details.
                record = {
                    'shipment_id': shipment_id,
                    'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
                    'day_of_week': day_of_week,
                    'hour_of_day': hour_of_day,
                    'origin': origin,
                    'destination': destination,
                    'transport_mode': transport_mode,
                    'distance': distance,
                    'volume': volume,
                    'transit_time': transit_time,
                    'cost': cost,
                    'co2_emissions': emissions,
                    'weather_condition': weather,
                    'weather_severity': weather_severity,
                    'traffic_congestion': traffic_congestion,
                    'disruption_indicator': disruption_indicator,
                    'capacity_modifier': capacity_modifier,
                    'port_congestion': port_congestion,
                    'scenario': scenario
                }
                shipment_id += 1
                scenario_records[scenario].append(record)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 6. Save Data as CSV Files  #
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~#
    for scenario, records in scenario_records.items():
        df = pd.DataFrame(records)
        filename = f"synthetic_logistics_data_{scenario}.csv"
        filepath = os.path.join(output_dir, filename)
        try:
            df.to_csv(filepath, index=False)
            print(f"Saved {len(df)} records for scenario '{scenario}' to {filepath}")
        except Exception as e:
            print(f"Error saving file for scenario '{scenario}': {e}")


if __name__ == '__main__':
    generate_synthetic_data()
