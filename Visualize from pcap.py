import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from scapy.all import rdpcap
import matplotlib.patches as mpatches

# map the protocols to their names and colors
PROTOCOL_MAP = {
    1: 'ICMP',
    6: 'TCP',
    17: 'UDP'
}
PROTOCOL_COLORS = {
    'TCP': 'blue',
    'UDP': 'green',
    'ICMP': 'red',
    'Other': 'gray'
}

def extract_data(pcap_file, max_packets=100000):
    edge_counter = Counter()
    packets = rdpcap(pcap_file, count=max_packets)

    if not packets:
        print("No packets found in the PCAP file.")
        return []

    for pkt in packets:
        if 'IP' in pkt:
            src = pkt['IP'].src
            dst = pkt['IP'].dst
            proto = PROTOCOL_MAP.get(pkt['IP'].proto, 'Other')  # Map protocol number to name
            
            # Use None if sport/dport fields are not present
            sport = pkt.sport if hasattr(pkt, 'sport') else "N/A"
            dport = pkt.dport if hasattr(pkt, 'dport') else "N/A"
            
            edge_key = (src, dst, proto, sport, dport)
            edge_counter[edge_key] += 1

    edges = [
        (src, dst, {
            'protocol': proto,
            'src_port': sport,
            'dst_port': dport,
            'packet_count': count
        })
        for (src, dst, proto, sport, dport), count in edge_counter.items()
    ]
    return edges

def draw_graph(pcap_file):
    Graph = nx.DiGraph()
    edges = extract_data(pcap_file)

    for src, dst, attributes in edges:
        Graph.add_edge(src, dst, **attributes)

    plt.figure(figsize=(12, 10))
    
    # Increase the spacing between the nodes
    pos = nx.spring_layout(Graph, k=0.8, iterations=50) 

    # Color the arrows based on the protocol
    edge_colors = []
    for (src, dst) in Graph.edges:
        data = Graph.edges[src, dst]
        proto = data.get('protocol', 'Other')
        edge_colors.append(PROTOCOL_COLORS.get(proto, 'gray'))

    nx.draw(
        Graph, pos, with_labels=False, arrows=True,  # Disable default labels
        node_size=500, node_color="skyblue",
        font_size=10, font_weight="bold", edge_color=edge_colors
    )

    # source port number, destination port number, no. packets
    edge_labels = {}
    for (src, dst) in Graph.edges:
        data = Graph.edges[src, dst]
        sport = data.get('src_port', 'N/A')
        dport = data.get('dst_port', 'N/A')
        packet_count = data.get('packet_count', 0)
        
        # Label the edges
        edge_labels[(src, dst)] = f"{sport}->{dport} ({packet_count} packets)"
    
    nx.draw_networkx_edge_labels(Graph, pos, edge_labels=edge_labels, font_color='black')

    node_labels = {node: node for node in Graph.nodes}
    label_pos = {node: (x, y + 0.05) for node, (x, y) in pos.items()}  # Plot label above the node
    nx.draw_networkx_labels(Graph, label_pos, labels=node_labels, font_size=10, font_color="black", font_weight="bold")
    plt.title("Network Graph")

    # Create the legend
    tcp_patch = mpatches.Patch(color=PROTOCOL_COLORS['TCP'], label='TCP (Blue)')
    udp_patch = mpatches.Patch(color=PROTOCOL_COLORS['UDP'], label='UDP (Green)')
    icmp_patch = mpatches.Patch(color=PROTOCOL_COLORS['ICMP'], label='ICMP (Red)')
    other_patch = mpatches.Patch(color=PROTOCOL_COLORS['Other'], label='Other (Gray)')

    plt.legend(handles=[tcp_patch, udp_patch, icmp_patch, other_patch], loc='upper right')
    plt.show()

if __name__ == "__main__":
    pcap_file = "1.pcap"
    draw_graph(pcap_file)