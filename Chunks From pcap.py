from scapy.all import rdpcap
import pandas as pd
from collections import defaultdict
from math import ceil

def load_chunks_from_pcap(pcap_file, overlap, max_packets = 1000000, time_interval = 1):
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

    prev_chunk_stats = None
    for time_interval_start, packet_chunk in time_chunks.items():
        features, current_chunk_stats = extract_features_from_pcap(packet_chunk, prev_chunk_stats)
        features_list.append(features)
        prev_chunk_stats = current_chunk_stats

    feature_names = list(prev_chunk_stats.keys()) if prev_chunk_stats else list(features.keys())
    features_df = pd.DataFrame(features_list, columns=feature_names)
    print("\nExtracted Features:")
    save_features_to_csv(features_df)
    return features_df

def extract_features_from_pcap(packets, prev_chunk_stats=None):
    features = defaultdict(int)
    
    # Counting protocols
    protocol_count = {"TCP": 0, "UDP": 0, "ICMP": 0}
    packet_sizes = []
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
        
        # Collect packet sizes
        packet_size = len(packet)
        packet_sizes.append(packet_size)
        byte_count += packet_size

    features["packet_count"] = packet_count # No packets
    features["byte_count"] = byte_count # Bytes for all packets
    features["unique_ips"] = len(unique_ips) # Unique Ips 
    features["unique_ports"] = len(unique_ports) # Unique ports
    features["tcp_syn_count"] = tcp_flags["SYN"] # Sycnhronize flags
    features["tcp_ack_count"] = tcp_flags["ACK"] # Acknowledge flags
    features["tcp_fin_count"] = tcp_flags["FIN"] # Finish flags
    features["tcp_count"] = protocol_count["TCP"] # TCP packets
    features["udp_count"] = protocol_count["UDP"] # UDP packets
    features["icmp_count"] = protocol_count["ICMP"] # ICMP packets
    features["avg_packet_size"] = sum(packet_sizes) / packet_count if packet_count > 0 else 0 # Average size of packets
    features["median_packet_size"] = sorted(packet_sizes)[len(packet_sizes) // 2] if packet_sizes else 0 # Median of packets
    features["max_packet_size"] = max(packet_sizes) if packet_sizes else 0 # Max packet size
    features["min_packet_size"] = min(packet_sizes) if packet_sizes else 0 # Min packet size
    features["std_dev_packet_size"] = (
        (sum((x - features["avg_packet_size"])**2 for x in packet_sizes) / packet_count)**0.5 
        if packet_count > 0 else 0) # Standard deviation

    # Rate of change
    if prev_chunk_stats: # If previous chunk exists
        features["rate_of_change_packet_count"] = packet_count - prev_chunk_stats["packet_count"]
        features["rate_of_change_byte_count"] = byte_count - prev_chunk_stats["byte_count"]
    else:
        features["rate_of_change_packet_count"] = 0
        features["rate_of_change_byte_count"] = 0

    return list(features.values()), features

def save_features_to_csv(features_df):
    csv = "train_data.csv"
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
    csv = "train_data.pcap"
    if file_format == "pcap":
        pcap_file = "Monday-WorkingHours.pcap"
        print(f"Loading data from PCAP file: {pcap_file}...")
        # Change overlap for the "ladder" chunking, go between 0 and 0.99
        chunks = load_chunks_from_pcap(pcap_file, overlap=0.25)
    elif file_format == "csv":
        chunks = load_features_from_csv(csv)
    print(chunks)
            