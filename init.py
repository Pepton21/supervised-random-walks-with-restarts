##########################################################
#                                                        #
#   Author: Petar Tonkovikj                              #
#                                                        #
#   General Notation:                                    #
#   N - number of nodes                                  #
#   S - number of layers                                 #
#   W - number of attributes                             #
#                                                        #
#   This file contains functions for initializing the    #
#   training and testing sets. The data is located in    #
#   .csv files. The goal is to represent the data in     #
#   a way suitable for further processing.               #
#                                                        #
# ########################################################

import numpy as np
import csv
import os

#Initialize network size parameters
num_nodes = 1230
num_attributes = 9

#######
#
#   Reads edge attributes between nodes from a .csv file            #
#                                                                   #
#   Returns: SxNxNxW array containing vectors of edge attributes    #
#   of size W for each pair of nodes in each layer                  #
                                                               ######
# Find the nodes in the network and save them to a file

def reduce_node_list(nodes, sampling_rate):
    count = 0
    list = []
    for node in nodes:
        if count % sampling_rate == 0:
            list.append(node)
        count += 1
    return list

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
# Returns a matrix
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
    np.save("stored_data/layer_transitions", layer_transitions)
    # Calculate layer adjacency matrices
    adjacency_matrices = np.array([attacks>0, messages>0, trades>0])
    np.save("stored_data/adjacency_matrices", adjacency_matrices)


def generate_future_adjacencies(map):
    attacks, sent_attacks, received_attacks = get_attacks_of_day(30, map)
    messages, sent_messages, received_messages = get_messages_of_day(30, map)
    trades, total = get_trades_of_day(30, map)
    adjacency_matrices = np.array([attacks > 0, messages > 0, trades > 0])

    num_layers = adjacency_matrices.shape[0]
    num_nodes = adjacency_matrices.shape[1]
    future_adjacency_matrix = np.empty((num_layers * num_nodes, num_layers * num_nodes))
    total_num_nodes = num_nodes * num_layers
    print(num_nodes)
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
    np.save("stored_data/future_adjacency_matrix", future_adjacency_matrix)

def load_edge_attributes():
    f = open('graph_output/edges.csv', newline='\n')
    reader = csv.reader(f, delimiter='\t', quotechar='|')
    edge_attributes = np.empty([num_nodes, num_nodes, num_attributes], dtype=np.float64)
    first_row = True
    for row in reader:
        if first_row:
            first_row = False
            continue
        source_node = int(row[2])
        dest_node = int(row[3])
        for j in range (4, len(row)):
            edge_attributes[source_node][dest_node][j-4] = float(row[j])
    return edge_attributes

#######
#
#   Reads the adjacency matrix of nodes                         #
#                                                               #
#   Returns: SxNxN array containing 1s if there is an edge      #
#   between a given pair of nodes in a given layer and 0s       #
#   otherwise                                                   #
                                                          #######

def generate_adjacency_matrix():
    adjacency_matrices = np.load("stored_data/adjacency_matrices.npy")
    num_layers = adjacency_matrices.shape[0]
    num_nodes = adjacency_matrices.shape[1]
    adjacency_matrix = np.empty((num_layers * num_nodes, num_layers * num_nodes))
    total_num_nodes = num_nodes * num_layers
    print(num_nodes)
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
    print(adjacency_matrix)
    np.save("stored_data/adjacency_matrix", adjacency_matrix)

def generate_feature_matrix():
    feature_matrices = np.load("stored_data/feature_matrices.npy")
    num_layers = feature_matrices.shape[0]
    num_nodes = feature_matrices.shape[1]
    total_num_nodes = num_nodes * num_layers
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
    np.save("stored_data/feature_matrix", feature_matrix)

#######
#
#   Creates the t set                                        #
#                                                               #
#   Returns: Two collections of data L and D, for positive      #
#   and negative t examples respectively. Each collection    #
#   has NxS lists of positive/negative examples (one for each   #
#   node in the multiplex network).                             #
                                                          #######

def generate_labels_set():
    adjacency_matrix = np.load('stored_data/adjacency_matrix.npy')
    future_adjacencies = np.load('stored_data/future_adjacency_matrix.npy')
    print(adjacency_matrix.shape)
    print(future_adjacencies.shape)
    L = []
    D = []
    num_nodes = adjacency_matrix.shape[0]
    for i in range(num_nodes):
        L.append([])
        D.append([])
        for j in range(num_nodes):
            if j not in D[i] and future_adjacencies[i][j] == 1:
                D[i].append(j)
            if j not in L[i] and future_adjacencies[i][j] == 0 and adjacency_matrix[i][j] == 0:
                L[i].append(j)
        print('Progress:', i * num_nodes + j, 'out of', num_nodes * num_nodes)
    np.save('stored_data/positive_samples', np.array(D))
    np.save('stored_data/negative_samples', np.array(L))

#extract_nodes()

#map = get_node_map()
#generate_layer_matrices(0.5, map)
#generate_future_adjacencies(map)
generate_feature_matrix()

