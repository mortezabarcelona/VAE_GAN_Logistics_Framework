import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
import os


def generate_synthetic_data():
    """
    Generate a realistic synthetic logistics dataset.

    This module incorporates:
      - A complete symmetric fixed distance matrix for all city pairs.
      - Valid transport mode constraints and realistic cost/emission scaling
        (with fixed base overheads and economies of scale).
      - Variable shipment volumes per hour via hourly multipliers.
      - Directional trade flow biases.
      - Mode-specific dynamic modifiers:
          * Road/Rail: weather/traffic effects.
          * Sea: port congestion (plus assignment of ports).
          * Air: weather effects (plus assignment of airports).
      - Carrier assignment based on mode.
      - Optional multi-leg shipment simulation (≈20% of shipments).
      - Aggregation of all records into one CSV file (with a 'scenario' column).
    """

    # ---------------------------
    # Part 1: Configuration Setup
    # ---------------------------
    TIME_HORIZON_DAYS = 30
    INTERVALS_PER_DAY = 24
    TOTAL_INTERVALS = TIME_HORIZON_DAYS * INTERVALS_PER_DAY

    # Candidate cities
    CITIES = ['Barcelona', 'Rotterdam', 'Hamburg', 'Lisbon', 'Madrid']

    # Fixed distance matrix for all city pairs (km) – symmetric keys
    FIXED_DISTANCES = {
        tuple(sorted(['Barcelona', 'Hamburg'])): 1650,
        tuple(sorted(['Barcelona', 'Lisbon'])): 1000,
        tuple(sorted(['Barcelona', 'Madrid'])): 640,
        tuple(sorted(['Barcelona', 'Rotterdam'])): 1600,
        tuple(sorted(['Hamburg', 'Lisbon'])): 2600,
        tuple(sorted(['Hamburg', 'Madrid'])): 2100,
        tuple(sorted(['Hamburg', 'Rotterdam'])): 400,
        tuple(sorted(['Lisbon', 'Madrid'])): 640,
        tuple(sorted(['Lisbon', 'Rotterdam'])): 2300,
        tuple(sorted(['Madrid', 'Rotterdam'])): 1900
    }

    # Valid transport modes per city pair to reflect geographic constraints.
    VALID_MODES = {
        tuple(sorted(['Barcelona', 'Hamburg'])): ['rail', 'sea', 'air'],
        tuple(sorted(['Barcelona', 'Lisbon'])): ['road', 'rail', 'air'],
        tuple(sorted(['Barcelona', 'Madrid'])): ['road', 'rail', 'air'],
        tuple(sorted(['Barcelona', 'Rotterdam'])): ['sea', 'air'],
        tuple(sorted(['Hamburg', 'Lisbon'])): ['sea', 'air'],
        tuple(sorted(['Hamburg', 'Madrid'])): ['rail', 'air'],
        tuple(sorted(['Hamburg', 'Rotterdam'])): ['road', 'rail'],
        tuple(sorted(['Lisbon', 'Madrid'])): ['road', 'rail'],
        tuple(sorted(['Lisbon', 'Rotterdam'])): ['sea', 'air'],
        tuple(sorted(['Madrid', 'Rotterdam'])): ['rail', 'sea', 'air']
    }

    # Transport mode parameters:
    # Each mode gets a transit time range (hours), fixed base cost, cost factor, and emission factor.
    TRANSPORT_MODES = {
        'road': {'transit_time': (24, 240), 'base_fixed_cost': 100, 'cost_factor': 0.20, 'emission_factor': 0.07},
        'rail': {'transit_time': (48, 360), 'base_fixed_cost': 150, 'cost_factor': 0.15, 'emission_factor': 0.05},
        'sea': {'transit_time': (240, 1080), 'base_fixed_cost': 200, 'cost_factor': 0.08, 'emission_factor': 0.03},
        'air': {'transit_time': (12, 72), 'base_fixed_cost': 300, 'cost_factor': 1.0, 'emission_factor': 0.1}
    }

    # Base shipment volume parameters (tons)
    BASE_VOLUME_MEAN = 10
    BASE_VOLUME_STD = 3

    # Dynamic environmental factors
    WEATHER_CONDITIONS = ['clear', 'rain', 'storm', 'fog']
    WEATHER_SEVERITY = {'clear': 0.0, 'rain': 0.2, 'storm': 0.5, 'fog': 0.3}
    TRAFFIC_CONGESTION_RANGE = (0.8, 1.2)  # applies to road and rail
    PORT_CONGESTION_RANGE = (0.9, 1.3)  # applies only to sea
    DISRUPTION_PROBABILITY = 0.1
    DISRUPTION_IMPACT = {'transit_time': 1.2, 'cost': 1.5, 'emissions': 1.1}
    DEMAND_MULTIPLIER_RANGE = (0.8, 1.2)
    CAPACITY_MODIFIER_RANGE = (0.9, 1.1)

    # Directional biases: Increase volume inbound for hubs and outbound for port cities.
    TRADE_FLOW_BIAS_INBOUND = {'Barcelona': 1.3, 'Madrid': 1.2}
    TRADE_FLOW_BIAS_OUTBOUND = {'Hamburg': 1.4, 'Rotterdam': 1.5}

    # Hourly volume multipliers (simulate realistic daily shipping rhythms)
    HOURLY_VOLUME_MULTIPLIERS = {
        0: 0.6, 1: 0.5, 2: 0.4, 3: 0.4, 4: 0.5, 5: 0.6,
        6: 0.8, 7: 1.0, 8: 1.2, 9: 1.5, 10: 1.6, 11: 1.8,
        12: 1.9, 13: 1.7, 14: 1.6, 15: 1.5, 16: 1.4, 17: 1.3,
        18: 1.2, 19: 1.0, 20: 0.9, 21: 0.8, 22: 0.7, 23: 0.6
    }

    # Scenario-specific adjustments
    SCENARIO_ADJUSTMENTS = {
        'baseline': {'num_shipments_per_interval': 100},
        'disruption': {'num_shipments_per_interval': 100, 'disruption_probability': 0.3},
        'sustainability': {'num_shipments_per_interval': 100, 'emission_penalty': 2.0},
        'high_demand': {'num_shipments_per_interval': 150}
    }

    RANDOM_SEED = 42

    # Additional realistic fields: Define ports, airports, and carriers for each mode.
    PORTS = {
        'Barcelona': ["Port of Barcelona"],
        'Rotterdam': ["Port of Rotterdam"],
        'Hamburg': ["Port of Hamburg"],
        'Lisbon': ["Port of Lisbon"]
        # Madrid is landlocked, so no ports.
    }
    AIRPORTS = {
        'Barcelona': ["Barcelona-El Prat Airport"],
        'Rotterdam': ["Rotterdam The Hague Airport"],
        'Hamburg': ["Hamburg Airport"],
        'Lisbon': ["Lisbon Humberto Delgado Airport"],
        'Madrid': ["Adolfo Suárez Madrid–Barajas Airport"]
    }
    CARRIERS = {
        'road': ["DHL", "FedEx", "DB Schenker", "Dachser"],
        'rail': ["DB Schenker", "PKP Cargo", "SNF"],
        'sea': ["Maersk", "MSC", "CMA CGM", "Hapag-Lloyd"],
        'air': ["Lufthansa", "Air France", "Iberia", "Vueling"]
    }

    # ---------------------------
    # Part 2: Environment Setup & Time Steps Generation
    # ---------------------------
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # Ensure the output directory exists.
    output_dir = os.path.join('data', 'processed')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate hourly timestamps over the simulation horizon.
    start_datetime = datetime.now()
    time_stamps = [start_datetime + timedelta(hours=i) for i in range(TOTAL_INTERVALS)]

    # ---------------------------
    # Part 3: Shipment Records Generation
    # ---------------------------
    records = []
    shipment_id = 1

    for ts in time_stamps:
        day_of_week = ts.strftime('%A')
        hour_of_day = ts.hour

        # Sample dynamic modifiers common for this timestamp.
        weather = random.choice(WEATHER_CONDITIONS)
        weather_severity = WEATHER_SEVERITY[weather]
        traffic_congestion = np.random.uniform(*TRAFFIC_CONGESTION_RANGE)
        port_congestion = np.random.uniform(*PORT_CONGESTION_RANGE)
        demand_multiplier = np.random.uniform(*DEMAND_MULTIPLIER_RANGE)
        capacity_modifier = np.random.uniform(*CAPACITY_MODIFIER_RANGE)

        # Iterate over each scenario.
        for scenario, settings in SCENARIO_ADJUSTMENTS.items():
            scenario_disruption_prob = settings.get('disruption_probability', DISRUPTION_PROBABILITY)
            # Determine number of shipments using hourly multiplier.
            num_shipments = int(settings['num_shipments_per_interval'] * HOURLY_VOLUME_MULTIPLIERS.get(hour_of_day, 1))

            for _ in range(num_shipments):
                # Select origin and destination.
                origin = random.choice(CITIES)
                destination = random.choice([c for c in CITIES if c != origin])
                key = tuple(sorted([origin, destination]))
                # Always use fixed distance from the matrix.
                distance = FIXED_DISTANCES.get(key, np.random.uniform(50, 1000))

                # Select a valid transport mode for this city pair.
                valid_modes = VALID_MODES.get(key, list(TRANSPORT_MODES.keys()))
                transport_mode = random.choice(valid_modes)
                mode_params = TRANSPORT_MODES[transport_mode]

                # Sample shipment volume and apply directional biases.
                volume = abs(np.random.normal(BASE_VOLUME_MEAN, BASE_VOLUME_STD))
                if destination in TRADE_FLOW_BIAS_INBOUND:
                    volume *= TRADE_FLOW_BIAS_INBOUND[destination]
                if origin in TRADE_FLOW_BIAS_OUTBOUND:
                    volume *= TRADE_FLOW_BIAS_OUTBOUND[origin]
                volume *= demand_multiplier

                # Carrier assignment for this mode.
                carrier = random.choice(CARRIERS.get(transport_mode, ["Generic Carrier"]))

                # Optional: 20% chance to simulate a multi-leg (intermodal) shipment.
                is_multi_leg = (random.random() < 0.20)
                if is_multi_leg:
                    intermediate_candidates = [city for city in CITIES if city not in [origin, destination]]
                    if intermediate_candidates:
                        intermediate = random.choice(intermediate_candidates)
                        # --- Leg 1: origin -> intermediate ---
                        key1 = tuple(sorted([origin, intermediate]))
                        dist1 = FIXED_DISTANCES.get(key1, np.random.uniform(50, 1000))
                        valid_modes1 = VALID_MODES.get(key1, list(TRANSPORT_MODES.keys()))
                        mode1 = random.choice(valid_modes1)
                        mode_params1 = TRANSPORT_MODES[mode1]
                        tt1 = np.random.uniform(*mode_params1['transit_time'])
                        # Apply mode-specific modifiers for leg 1.
                        if mode1 == 'road':
                            tt1 *= (1 + weather_severity) * traffic_congestion
                        elif mode1 == 'rail':
                            tt1 *= traffic_congestion
                        elif mode1 == 'sea':
                            tt1 *= port_congestion
                        elif mode1 == 'air':
                            tt1 *= (1 + weather_severity)
                        cost1 = mode_params1['base_fixed_cost'] + (volume ** 0.85) * dist1 * mode_params1['cost_factor']
                        emis1 = (volume ** 0.95) * dist1 * mode_params1['emission_factor']
                        carrier1 = random.choice(CARRIERS.get(mode1, ["Generic Carrier"]))
                        # For sea, assign port info.
                        port_origin1 = random.choice(PORTS.get(origin, [None])) if mode1 == 'sea' else None
                        port_destination1 = random.choice(PORTS.get(intermediate, [None])) if mode1 == 'sea' else None
                        airport_origin1 = random.choice(AIRPORTS.get(origin, [None])) if mode1 == 'air' else None
                        airport_destination1 = random.choice(
                            AIRPORTS.get(intermediate, [None])) if mode1 == 'air' else None

                        # --- Leg 2: intermediate -> destination ---
                        key2 = tuple(sorted([intermediate, destination]))
                        dist2 = FIXED_DISTANCES.get(key2, np.random.uniform(50, 1000))
                        valid_modes2 = VALID_MODES.get(key2, list(TRANSPORT_MODES.keys()))
                        mode2 = random.choice(valid_modes2)
                        mode_params2 = TRANSPORT_MODES[mode2]
                        tt2 = np.random.uniform(*mode_params2['transit_time'])
                        if mode2 == 'road':
                            tt2 *= (1 + weather_severity) * traffic_congestion
                        elif mode2 == 'rail':
                            tt2 *= traffic_congestion
                        elif mode2 == 'sea':
                            tt2 *= port_congestion
                        elif mode2 == 'air':
                            tt2 *= (1 + weather_severity)
                        cost2 = mode_params2['base_fixed_cost'] + (volume ** 0.85) * dist2 * mode_params2['cost_factor']
                        emis2 = (volume ** 0.95) * dist2 * mode_params2['emission_factor']
                        carrier2 = random.choice(CARRIERS.get(mode2, ["Generic Carrier"]))
                        port_origin2 = random.choice(PORTS.get(intermediate, [None])) if mode2 == 'sea' else None
                        port_destination2 = random.choice(PORTS.get(destination, [None])) if mode2 == 'sea' else None
                        airport_origin2 = random.choice(AIRPORTS.get(intermediate, [None])) if mode2 == 'air' else None
                        airport_destination2 = random.choice(
                            AIRPORTS.get(destination, [None])) if mode2 == 'air' else None

                        # Apply disruption effects for each leg.
                        if random.random() < scenario_disruption_prob:
                            tt1 *= DISRUPTION_IMPACT['transit_time']
                            cost1 *= DISRUPTION_IMPACT['cost']
                            emis1 *= DISRUPTION_IMPACT['emissions']
                        if random.random() < scenario_disruption_prob:
                            tt2 *= DISRUPTION_IMPACT['transit_time']
                            cost2 *= DISRUPTION_IMPACT['cost']
                            emis2 *= DISRUPTION_IMPACT['emissions']

                        total_distance = dist1 + dist2
                        total_transit_time = tt1 + tt2
                        total_cost = cost1 + cost2
                        total_emissions = emis1 + emis2
                        # Apply scenario-specific sustainability penalty if applicable.
                        if scenario == 'sustainability':
                            penalty = settings.get('emission_penalty', 1)
                            total_cost *= penalty
                            total_emissions *= penalty

                        record = {
                            'shipment_id': shipment_id,
                            'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
                            'day_of_week': day_of_week,
                            'hour_of_day': hour_of_day,
                            'origin': origin,
                            'destination': destination,
                            'intermediate': intermediate,
                            'transport_mode': f"{mode1}/{mode2}",
                            'route': "Multi-leg",
                            'distance': total_distance,
                            'volume': volume,
                            'transit_time': total_transit_time,
                            'cost': total_cost,
                            'co2_emissions': total_emissions,
                            'carrier': f"{carrier1}/{carrier2}",
                            'port_origin': port_origin1 if port_origin1 else None,
                            'port_destination': port_destination2 if port_destination2 else None,
                            'airport_origin': airport_origin1 if airport_origin1 else None,
                            'airport_destination': airport_destination2 if airport_destination2 else None,
                            'weather_condition': weather,
                            'weather_severity': weather_severity,
                            'traffic_congestion': traffic_congestion if mode1 in ['road', 'rail'] or mode2 in ['road',
                                                                                                               'rail'] else None,
                            'port_congestion': port_congestion if mode1 == 'sea' or mode2 == 'sea' else None,
                            'disruption_indicator': 1 if random.random() < scenario_disruption_prob else 0,
                            'capacity_modifier': capacity_modifier,
                            'scenario': scenario,
                            'multi_leg': True
                        }
                        shipment_id += 1
                        records.append(record)
                        continue  # Skip regular processing for multi-leg shipments.

                # --- Single-leg shipment ---
                valid_modes = VALID_MODES.get(key, list(TRANSPORT_MODES.keys()))
                transport_mode = random.choice(valid_modes)
                mode_params = TRANSPORT_MODES[transport_mode]
                base_cost = mode_params['base_fixed_cost'] + (volume ** 0.85) * distance * mode_params['cost_factor']
                base_emissions = (volume ** 0.95) * distance * mode_params['emission_factor']
                transit_time = np.random.uniform(*mode_params['transit_time'])
                if transport_mode == 'road':
                    transit_time *= (1 + weather_severity) * traffic_congestion
                elif transport_mode == 'rail':
                    transit_time *= traffic_congestion
                elif transport_mode == 'sea':
                    transit_time *= port_congestion
                elif transport_mode == 'air':
                    transit_time *= (1 + weather_severity)
                if random.random() < scenario_disruption_prob:
                    transit_time *= DISRUPTION_IMPACT['transit_time']
                    base_cost *= DISRUPTION_IMPACT['cost']
                    base_emissions *= DISRUPTION_IMPACT['emissions']
                if scenario == 'sustainability':
                    penalty = settings.get('emission_penalty', 1)
                    base_cost *= penalty
                    base_emissions *= penalty
                # Carrier assignment for single leg.
                carrier = random.choice(CARRIERS.get(transport_mode, ["Generic Carrier"]))
                # For sea mode, assign ports; for air, assign airports.
                port_origin = random.choice(PORTS.get(origin, [None])) if transport_mode == 'sea' else None
                port_destination = random.choice(PORTS.get(destination, [None])) if transport_mode == 'sea' else None
                airport_origin = random.choice(AIRPORTS.get(origin, [None])) if transport_mode == 'air' else None
                airport_destination = random.choice(
                    AIRPORTS.get(destination, [None])) if transport_mode == 'air' else None

                record = {
                    'shipment_id': shipment_id,
                    'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
                    'day_of_week': day_of_week,
                    'hour_of_day': hour_of_day,
                    'origin': origin,
                    'destination': destination,
                    'intermediate': None,
                    'transport_mode': transport_mode,
                    'route': "Direct",
                    'distance': distance,
                    'volume': volume,
                    'transit_time': transit_time,
                    'cost': base_cost,
                    'co2_emissions': base_emissions,
                    'carrier': carrier,
                    'port_origin': port_origin,
                    'port_destination': port_destination,
                    'airport_origin': airport_origin,
                    'airport_destination': airport_destination,
                    'weather_condition': weather,
                    'weather_severity': weather_severity,
                    'traffic_congestion': traffic_congestion if transport_mode in ['road', 'rail'] else None,
                    'port_congestion': port_congestion if transport_mode == 'sea' else None,
                    'disruption_indicator': 1 if random.random() < scenario_disruption_prob else 0,
                    'capacity_modifier': capacity_modifier,
                    'scenario': scenario,
                    'multi_leg': False
                }
                shipment_id += 1
                records.append(record)

    # ---------------------------
    # Part 4: Data Aggregation & Saving
    # ---------------------------
    df = pd.DataFrame(records)
    output_path = os.path.join(output_dir, "synthetic_logistics_data.csv")
    try:
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} shipment records to {output_path}")
    except Exception as e:
        print(f"Error saving CSV: {e}")


if __name__ == '__main__':
    generate_synthetic_data()
