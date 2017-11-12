##########################################################
#                                                        #
#   Author: Petar Tonkovikj                              #
#                                                        #
# ########################################################

import numpy as np
import csv
import os

#Initialize network size parameters
num_nodes = 1230
num_attributes = 9

# Sample nodes with the specified sampling rate

def reduce_node_list(nodes, sampling_rate):
    count = 0
    list = []
    for node in nodes:
        if count % sampling_rate == 0:
            list.append(node)
        count += 1
    return list

# Find the nodes in the network and save them to a file

def extract_nodes():
    nodes = set()
    # Search for nodes involed in attacks
    path = "Attacks-Network-CSV"
    files = os.listdir(path)
    print("Attacks")
    for file in files:
        with open(os.path.join(path, file), newline='\n') as f:
            if not file.endswith("30.csv"):
                reader = csv.reader(f, delimiter=',', quotechar='|')
                for line in reader:
                    if int(line[1]) not in nodes:
                        print("Added:", line[1], file)
                        nodes.add(int(line[1]))
                    if int(line[2]) not in nodes:
                        print("Added:", line[2], file)
                        nodes.add(int(line[2]))
    # Search for nodes involved in messages
    path = "Messages-Network-CSV"
    files = os.listdir(path)
    print("Messages")
    for file in files:
        with open(os.path.join(path, file), newline='\n') as f:
            if not file.endswith("30.csv"):
                reader = csv.reader(f, delimiter=',', quotechar='|')
                for line in reader:
                    if int(line[1]) not in nodes:
                        print("Added:", line[1], file)
                        nodes.add(int(line[1]))
                    if int(line[2]) not in nodes:
                        print("Added:", line[2], file)
                        nodes.add(int(line[2]))
    # Search for nodes involved in trades
    path = "Trades-Network-CSV"
    files = os.listdir(path)
    print("Trades")
    for file in files:
        with open(os.path.join(path, file), newline='\n') as f:
            if not file.endswith("30.csv"):
                reader = csv.reader(f, delimiter=',', quotechar='|')
                for line in reader:
                    if int(line[1]) not in nodes:
                        print("Added:", line[1], file)
                        nodes.add(int(line[1]))
                    if int(line[2]) not in nodes:
                        print("Added:", line[2], file)
                        nodes.add(int(line[2]))
    # Sort node IDs
    nodes = list(nodes)
    nodes.sort()
    nodes = reduce_node_list(nodes, 4)
    print(nodes)
    print(len(nodes))
    # Write node IDs to file
    with open("node_mappings", 'w', newline='\n') as f:
        writer = csv.writer(f, delimiter=',')
        for i in range(len(nodes)):
            writer.writerow([nodes[i], i])

# Read nodes from file
def get_node_map():
    # Read file defined in previous function
    with open("node_mappings", 'r', newline='\n') as f:
        map = {}
        reader = csv.reader(f, delimiter=',', quotechar='|')
        for line in reader:
            map[line[0]] = line[1]
    return map

# Get the data about attacks in a given day
# Returns a matrix of number of attacks between pairs of nodes and number of initiated and endured attacks of each node
def get_attacks_of_day(day, map):
    path = "Attacks-Network-CSV/attacks-timestamped-2009-12-" + str(day) + ".csv"
    attacks = np.zeros((len(map), len(map)))
    sent = np.zeros((len(map)))
    received = np.zeros((len(map)))
    with open(path, 'r', newline='\n') as f:
        reader = csv.reader(f, delimiter=',', quotechar='|')
        for line in reader:
            if line[1] in map.keys() and line[2] in map.keys():
                attacks[int(map[line[1]])][int(map[line[2]])] += 1
                sent[int(map[line[1]])] += 1
                received[int(map[line[2]])] += 1
    return attacks, sent, received

# Get the data about messages in a given day
# Returns a matrix of number of messages between pairs of nodes and number of sent and received messages of each node
def get_messages_of_day(day, map):
    path = "Messages-Network-CSV/messages-timestamped-2009-12-" + str(day) + ".csv"
    messages = np.zeros((len(map), len(map)))
    sent = np.zeros((len(map)))
    received = np.zeros((len(map)))
    with open(path, 'r', newline='\n') as f:
        reader = csv.reader(f, delimiter=',', quotechar='|')
        for line in reader:
            if line[1] in map.keys() and line[2] in map.keys():
                messages[int(map[line[1]])][int(map[line[2]])] += 1
                sent[int(map[line[1]])] += 1
                received[int(map[line[2]])] += 1
    return messages, sent, received

# Get the data about trades in a given day
# Returns a matrix of number of trades between pairs of nodes and number of total trades done by each node
def get_trades_of_day(day, map):
    path = "Trades-Network-CSV/trades-timestamped-2009-12-" + str(day) + ".csv"
    trades = np.zeros((len(map), len(map)))
    total = np.zeros((len(map)))
    with open(path, 'r', newline='\n') as f:
        reader = csv.reader(f, delimiter=',', quotechar='|')
        for line in reader:
            if line[1] in map.keys() and line[2] in map.keys():
                trades[int(map[line[1]])][int(map[line[2]])] += 1
                trades[int(map[line[2]])][int(map[line[1]])] += 1
                total[int(map[line[1]])] += 1
                total[int(map[line[2]])] += 1
    return trades, total

# Get the data about communities in a given day
# Returns a boolean matrix containing information whether pairs of nodes belonged to the same community
def get_mutual_communities_of_day(day, map):
    path = "Communities/communities-2009-12-" + str(day) + ".txt"
    communities = np.zeros((len(map), len(map)))
    with open(path, 'r', newline='\n') as f:
        for row in f:
            line = row.split()
            for i in range(len(line)-1):
                for j in range(i+1, len(line)):
                    if line[i] in map.keys() and line[j] in map.keys():
                        if line[i] in map.keys() and line[j] in map.keys():
                            communities[int(map[line[i]])][int(map[line[j]])] += 1
                            communities[int(map[line[j]])][int(map[line[i]])] += 1
    return communities

# Generate the matrices of feature vectors for each layer and store them in a file
def generate_layer_matrices(alpha, map):
    beta = 1 - alpha
    # Initialize elements with data from first day
    attacks, sent_attacks, received_attacks = get_attacks_of_day(1, map)
    messages, sent_messages, received_messages = get_messages_of_day(1, map)
    trades, total = get_trades_of_day(1, map)
    communities = get_mutual_communities_of_day(1, map)
    for i in range(2, 30):
        # Get data from next day
        new_attacks, new_sent_attacks, new_received_attacks = get_attacks_of_day(i, map)
        new_messages, new_sent_messages, new_received_messages = get_messages_of_day(i, map)
        new_trades, new_total = get_trades_of_day(i, map)
        new_communities = get_mutual_communities_of_day(i, map)
        # Apply aging algorithm on edge attributes
        attacks = alpha * attacks + beta * new_attacks
        messages = alpha * messages + beta * new_messages
        trades = alpha * trades + beta * new_trades
        communities = alpha * communities + beta * new_communities
        # Accumulate node attributes
        sent_attacks = sent_attacks + new_sent_attacks
        received_attacks = received_attacks + new_received_attacks
        sent_messages = sent_messages + new_sent_messages
        received_messages = received_messages + new_received_messages
        total = total + new_total
    # Create matrices of feature vectors for each layer
    attack_features = np.zeros((len(map), len(map), 11))
    message_features = np.zeros((len(map), len(map), 11))
    trades_features = np.zeros((len(map), len(map), 11))
    # For node attributes calculate difference, for edge attributes set the value
    for i in range(len(map)):
        for j in range(len(map)):
            print('Progress:', i * len(map) + j, 'out of', len(map) * len(map))
            attack_features[i][j][0] = abs(sent_attacks[i] - sent_attacks[j])
            attack_features[i][j][1] = abs(received_attacks[i] - received_attacks[j])
            attack_features[i][j][2] = attacks[i][j]
            attack_features[i][j][3] = attacks[j][i]
            attack_features[i][j][10] = communities[i][j]
            message_features[i][j][4] = abs(sent_messages[i] - sent_messages[j])
            message_features[i][j][5] = abs(received_messages[i] - received_messages[j])
            message_features[i][j][6] = messages[i][j]
            message_features[i][j][7] = messages[j][i]
            message_features[i][j][10] = communities[i][j]
            trades_features[i][j][8] = abs(total[i] - total[j])
            trades_features[i][j][9] = trades[i][j]
            trades_features[i][j][10] = communities[i][j]
    feature_matrices = np.array([attack_features, message_features, trades_features])
    np.save("stored_data/feature_matrices", feature_matrices)
    # Calculate layer transition probabilities
    layer_transitions = np.zeros((len(map), 3))
    for i in range(len(map)):
        total_interactions = sent_attacks[i] + sent_messages[i] + total[i] + received_attacks[i] + received_messages[i]
        if total_interactions != 0:
            layer_transitions[i][0] = (sent_attacks[i] + received_attacks[i]) / total_interactions
            layer_transitions[i][1] = (sent_messages[i] + received_messages[i]) / total_interactions
            layer_transitions[i][2] = total[i] / total_interactions
        else:
            layer_transitions[i][0] = 1/3
            layer_transitions[i][1] = 1/3
            layer_transitions[i][2] = 1/3
    # Save the matrices in a file for caching purposes
    np.save("stored_data/layer_transitions", layer_transitions)
    # Calculate layer adjacency matrices
    adjacency_matrices = np.array([attacks>0, messages>0, trades>0])
    # Save the matrices in a file for caching purposes
    np.save("stored_data/adjacency_matrices", adjacency_matrices)

# Generate the adjacency matrix of the near future (last day)
def generate_future_adjacencies(map):
    # read the interactions between node pairs during the last day
    attacks, sent_attacks, received_attacks = get_attacks_of_day(30, map)
    messages, sent_messages, received_messages = get_messages_of_day(30, map)
    trades, total = get_trades_of_day(30, map)
    adjacency_matrices = np.array([attacks > 0, messages > 0, trades > 0])
    #construct the extended adjacency matrix that includes information for all layers
    num_layers = adjacency_matrices.shape[0]
    num_nodes = adjacency_matrices.shape[1]
    future_adjacency_matrix = np.empty((num_layers * num_nodes, num_layers * num_nodes))
    total_num_nodes = num_nodes * num_layers
    for i in range(total_num_nodes):
        for j in range(total_num_nodes):
            print('Progress:', i * total_num_nodes + j, 'out of', total_num_nodes * total_num_nodes)
            src_node = i % num_nodes
            dest_node = j % num_nodes
            dest_layer = j // num_nodes
            if adjacency_matrices[dest_layer][src_node][dest_node] == 1:
                future_adjacency_matrix[i][j] = 1
            else:
                future_adjacency_matrix[i][j] = 0
    print(future_adjacency_matrix)
    # Save the matrix to a file for caching purposes
    np.save("stored_data/future_adjacency_matrix", future_adjacency_matrix)

# Construct the extended adjacency matrix which contains information for all layers
def generate_adjacency_matrix():
    # Read the separate adjacency matrices for the layers
    adjacency_matrices = np.load("stored_data/adjacency_matrices.npy")
    num_layers = adjacency_matrices.shape[0]
    num_nodes = adjacency_matrices.shape[1]
    # Create the extended adjacency matrix
    adjacency_matrix = np.empty((num_layers * num_nodes, num_layers * num_nodes))
    total_num_nodes = num_nodes * num_layers
    for i in range(total_num_nodes):
        for j in range(total_num_nodes):
            print('Progress:', i * total_num_nodes + j, 'out of', total_num_nodes * total_num_nodes)
            src_node = i % num_nodes
            dest_node = j % num_nodes
            dest_layer = j // num_nodes
            if adjacency_matrices[dest_layer][src_node][dest_node] == 1:
                adjacency_matrix[i][j] = 1
            else:
                adjacency_matrix[i][j] = 0
    # Save the extended matrix to a file for caching purposes
    np.save("stored_data/adjacency_matrix", adjacency_matrix)

# Construct the extended feature matrix which contains information for all layers
def generate_feature_matrix():
    # Read the separate feature matrices for the layers
    feature_matrices = np.load("stored_data/feature_matrices.npy")
    num_layers = feature_matrices.shape[0]
    num_nodes = feature_matrices.shape[1]
    total_num_nodes = num_nodes * num_layers
    # Create the extended feature matrix
    feature_matrix = np.zeros((total_num_nodes, total_num_nodes, feature_matrices.shape[3]))
    for i in range(total_num_nodes):
        for j in range(total_num_nodes):
            src_layer = i // num_nodes
            dest_layer = j // num_nodes
            src_node = i % num_nodes
            dest_node = j % num_nodes
            feature_matrix[i][j] = feature_matrix[i][j] + feature_matrices[src_layer][src_node][dest_node]
            feature_matrix[i][j] = feature_matrix[i][j] + feature_matrices[dest_layer][src_node][dest_node]
            print('Progress:', i * total_num_nodes + j, 'out of', total_num_nodes * total_num_nodes)
    # Save the extended matrix to a file for caching purposes
    np.save("stored_data/feature_matrix", feature_matrix)

# Generate the D and L sets of the SRW algorithm for each node
def generate_labels_set():
    # Read the required data from the cache files
    adjacency_matrix = np.load('stored_data/adjacency_matrix.npy')
    future_adjacencies = np.load('stored_data/future_adjacency_matrix.npy')
    # Initialize the list of L and D sets for the nodes
    L = []
    D = []
    num_nodes = adjacency_matrix.shape[0]
    for i in range(num_nodes):
        # For each node, initialize the L and D sets to be empty
        L.append([])
        D.append([])
        for j in range(num_nodes):
            # If there is a future adjacency between the nodes, add j to the destination set
            if j not in D[i] and future_adjacencies[i][j] == 1:
                D[i].append(j)
            # If there is no future adjacency between the nodes, and the nodes are not connected, add j to the L set
            if j not in L[i] and future_adjacencies[i][j] == 0 and adjacency_matrix[i][j] == 0:
                L[i].append(j)
        print('Progress:', i * num_nodes + j, 'out of', num_nodes * num_nodes)
    # Store the L and D sets to files for caching purposes
    np.save('stored_data/positive_samples', np.array(D))
    np.save('stored_data/negative_samples', np.array(L))

#extract_nodes()

#map = get_node_map()
#generate_layer_matrices(0.5, map)
#generate_future_adjacencies(map)
#generate_feature_matrix()

