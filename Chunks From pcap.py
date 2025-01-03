from scapy.all import rdpcap
import pandas as pd
from collections import defaultdict
import os
from math import ceil
from sklearn.preprocessing import StandardScaler

def load_chunks_from_pcap(pcap_file, overlap, max_packets = 500000, time_interval = 1):
    packet_times = [] 
    features_list = []

    if not (0 <= overlap < 1): # 
        raise ValueError("Overlap must be between 0 and 1")
    
    packets = rdpcap(pcap_file, count=max_packets)  # Read all packets
    
    if not packets:
        print("No packets found in the PCAP file.")
        return

    for packet in packets:
        if packet.haslayer("IP"): # Ensure only processing packets on the IP layer
            timestamp = packet.time 
            packet_times.append(timestamp)
    
    start_time = min(packet_times)
    start_time = float(start_time) 

    time_chunks = defaultdict(list)
    step_size = time_interval * (1 - overlap)
    
    for packet in packets:
        if packet.haslayer("IP"):
            timestamp = packet.time
            for i in range(ceil((timestamp - start_time) / step_size)): # Round up
                window_start = start_time + i * step_size
                if window_start <= timestamp < window_start + time_interval:
                    time_chunks[window_start].append(packet)

    for time_interval_start, packet_chunk in time_chunks.items():
        features = extract_features_from_pcap(packet_chunk)
        features_list.append(features)

    features_df = pd.DataFrame(features_list, columns=["packet_count", "byte_count", "unique_ips", 
                                                       "unique_ports", "tcp_syn_count", "tcp_ack_count", 
                                                       "tcp_fin_count", "tcp_count", "udp_count", "icmp_count"])
    features_df = normalize_data(features_df)
    print("\nExtracted Features:")
    save_features_to_csv(features_df)
    return features_df

def extract_features_from_pcap(packets):
    features = defaultdict(int) 
    
    # Counting protocols
    protocol_count = {"TCP": 0, "UDP": 0, "ICMP": 0}
    byte_count = 0
    packet_count = len(packets)
    unique_ips = set()
    unique_ports = set()
    tcp_flags = {"SYN": 0, "ACK": 0, "FIN": 0}

    for packet in packets:
        if packet.haslayer("IP"):
            unique_ips.add(packet["IP"].src)
            unique_ips.add(packet["IP"].dst)
        
        if packet.haslayer("TCP"):
            protocol_count["TCP"] += 1
            unique_ports.add(packet["TCP"].sport)
            unique_ports.add(packet["TCP"].dport)
            if packet["TCP"].flags & 0x02:
                tcp_flags["SYN"] += 1
            if packet["TCP"].flags & 0x10:
                tcp_flags["ACK"] += 1
            if packet["TCP"].flags & 0x01:
                tcp_flags["FIN"] += 1

        elif packet.haslayer("UDP"):
            protocol_count["UDP"] += 1
            unique_ports.add(packet["UDP"].sport)
            unique_ports.add(packet["UDP"].dport)

        elif packet.haslayer("ICMP"):
            protocol_count["ICMP"] += 1
        
        byte_count += len(packet)

    # Packet count, byte count, protocol counts, unique IPs and ports
    features["packet_count"] = packet_count
    features["byte_count"] = byte_count
    features["unique_ips"] = len(unique_ips)
    features["unique_ports"] = len(unique_ports)
    features["tcp_syn_count"] = tcp_flags["SYN"]
    features["tcp_ack_count"] = tcp_flags["ACK"]
    features["tcp_fin_count"] = tcp_flags["FIN"]
    features["tcp_count"] = protocol_count["TCP"]
    features["udp_count"] = protocol_count["UDP"]
    features["icmp_count"] = protocol_count["ICMP"]
    
    return list(features.values()) 

def normalize_data(dataframe):
    numeric_columns = dataframe.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    dataframe[numeric_columns] = scaler.fit_transform(dataframe[numeric_columns])
    return dataframe

def save_features_to_csv(features_df):
    csv = "chunks_normalized.csv"
    features_df.to_csv(csv, mode='w', header=True, index=False)
    print(f"Saved features to {csv}")
    return

def load_features_from_csv(file_path):
    try:
        data = pd.read_csv(file_path)
        print("CSV Data Loaded Successfully!")
        return data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    file_format = "pcap" # Change this to pcap to get new features and chunks 
    csv = "chunks_normalized.csv"
    if file_format == "pcap":
        pcap_file = "1.pcap"
        print(f"Loading data from PCAP file: {pcap_file}...")
        # Change overlap for the "ladder" chunking, go between 0 and 0.99
        chunks = load_chunks_from_pcap(pcap_file, overlap=0.25)
    elif file_format == "csv":
        chunks = load_features_from_csv(csv)
    print(chunks)
            