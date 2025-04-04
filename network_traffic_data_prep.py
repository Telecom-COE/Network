import numpy as np
import pandas as pd
from datetime import timedelta, datetime


# Seed for reproducibility
np.random.seed(42)

# Generating synthetic data
n_samples = 5000

# Independent variables
signal_strength = np.random.normal(-50, 5, n_samples)  # Signal Strength in dBm
network_latency = np.random.normal(30, 10, n_samples)  # Network Latency in ms
packet_loss = np.random.normal(0.5, 0.1, n_samples)   # Packet Loss in percentage
connection_type = np.random.choice([1,2,3], size=n_samples)  # Connection type
#connection_type = np.random.choice(['4G','5G','WI-FI'], size=n_samples)  # Connection type
network_congestion = np.random.normal(0.7, 0.1, n_samples)  # Network Congestion Level (0-1)
avg_data_throughput = np.random.normal(50, 10, n_samples)  # Average Data Throughput in Mbps
num_users = np.random.randint(100, 1000, n_samples)  # Number of Users
network_availability = np.random.normal(99.5, 0.5, n_samples)  # Network Availability in percentage
snr = np.random.normal(20, 5, n_samples)  # Signal to Noise Ratio in dB
num_active_connections = np.random.randint(10, 100, n_samples)  # Number of Active Connections
weather_temp = np.random.normal(22, 5, n_samples)  # Weather Temperature in °C
wifi_interference = np.random.normal(10, 3, n_samples)  # Wi-Fi Interference in dB
network_downtime = np.random.normal(2, 0.5, n_samples)  # Network Downtime in hours
customer_satisfaction = np.random.normal(4, 1, n_samples)  # Customer Satisfaction Score (1-5)

# Target variable
traffic_volume = 0.3 * signal_strength + 0.1 * network_latency - 0.2 * packet_loss + \
                 0.5 * network_congestion + 0.6 * avg_data_throughput + 0.02 * num_users + \
                 0.4 * network_availability - 0.05 * snr + 0.3 * num_active_connections - \
                 0.1 * weather_temp - 0.2 * wifi_interference + 0.1 * network_downtime - \
                 0.4 * customer_satisfaction + np.random.normal(0, 10, n_samples)  # Traffic Volume (GB)

# Creating DataFrame
data = pd.DataFrame({
    'Signal Strength (dBm)': signal_strength,
    'Network Latency (ms)': network_latency,
    'Packet Loss (%)': packet_loss,
    'Connection Type': connection_type,
    'Network Congestion Level': network_congestion,
    'Average Data Throughput (Mbps)': avg_data_throughput,
    'Number of Users': num_users,
    'Network Availability (%)': network_availability,
    'Signal to Noise Ratio (SNR)': snr,
    'Number of Active Connections': num_active_connections,
    'Weather (Temperature in °C)': weather_temp,
    'Wi-Fi Interference': wifi_interference,
    'Network Downtime (hours)': network_downtime,
    'Customer Satisfaction Score': customer_satisfaction,
    'Traffic Volume (GB)': traffic_volume
})

# Save the DataFrame to a CSV
data.to_csv('network_traffic.csv', index=False)

