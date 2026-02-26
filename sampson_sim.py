import networkx as nx
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

def dfs(G, node, visited):
    neighbors = list(G.neighbors(node))
    for neighbor in neighbors:
        if visited[neighbor-1]==0:
            visited[neighbor-1]=1
            dfs(G, neighbor, visited)

def get_class_sets(G,s1,s2):
    G1=G.copy()
    n=G1.number_of_nodes()

    nodes_in_s1=np.zeros((n,1))
    nodes_in_s2=np.zeros((n,1))
    for u in list(G.predecessors(s1)):
        G1.remove_edge(u, s1)
    dfs(G1,s1,nodes_in_s1)
    dfs(G1,s2,nodes_in_s2)
    nodes_in_s1[s1-1]=1
    nodes_in_s2[s2-1]=1


    classification={}
    for i in range(0,n):
        if(nodes_in_s1[i] and nodes_in_s2[i]):
            classification[i+1]=3
        elif(nodes_in_s1[i]):
            classification[i+1]=1
        elif(nodes_in_s2[i]):
            classification[i+1]=2
        else:
            classification[i+1]=4
    return classification


def calculate_Fe(W,beta,n):
    W = np.array(W, dtype=float)
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    W_normalized = W / row_sums
    return np.linalg.solve(np.eye(n)-np.dot((np.eye(n)-beta),W_normalized),np.eye(n))

def get_endorsers(G,s1,rem):
    G1=G.copy()
    for u in list(G.predecessors(s1)):
        G1.remove_edge(u, s1)
    H=nx.condensation(G1)
    # print(s1)
    print(H.graph["mapping"])


def calculate_delta(F,b,d,beta_s1,s1,w_tilde,sig):

    return  beta_s1*w_tilde*sig*(F[s1][s1]-F[d][s1])/(1-w_tilde*F[s1][b]+w_tilde*F[d][b])

def update_inverse(F,a,b,d,w_tilde):
    u = (w_tilde * np.eye(n)[b]).reshape(n, 1)      # n×1 column vector
    v = (np.eye(n)[a] - np.eye(n)[d]).reshape(1, n)
    # print(F @ u @ v @ F/(1-w_tilde*F[a][b]+w_tilde*F[d][b]))
    return F+F @ u @ v @ F/(1-w_tilde*F[a][b]+w_tilde*F[d][b])

def get_influence_centrality(G,s1,s2):
    adj_matrix=nx.adjacency_matrix(G)
    adj_matrix_dense = adj_matrix.todense()

    adj=np.array(adj_matrix_dense)
    in_degree=adj.sum(axis=0)
    adj=adj/in_degree
    adj=adj.T
    length=adj.shape[0]
    stub=np.zeros((length,1))
    stub[s1-1,0]=0.1
    stub[s2-1,0]=0.6
    beta=np.diag(stub.flatten())
    P=np.dot(np.linalg.inv(np.eye(length)-np.dot((np.eye(length)-beta),adj)),beta)
    F= np.linalg.inv(np.eye(length)-np.dot((np.eye(length)-beta),adj.T))
    return np.dot(np.ones(length).T,P)/length,beta,F

def get_delta(G_num,a,b,d,w_tilde,s1,s2):
    G1=G_num.copy()
    if G1.has_edge(a,b):
        G1[a][b]['weight'] += w_tilde 
    else:
        G1.add_edge(a,b, weight=w_tilde)
    G1[d][b]['weight'] -= w_tilde
    initial,_,_=get_influence_centrality(G_num,s1,s2)
    final,_,_= get_influence_centrality(G1,s1,s2)
    return final[s1-1]-initial[s1-1]

with open("sampson.pkl", "rb") as f:
    G = pickle.load(f)

node_list = list(G.nodes())  # Get existing node names
node_mapping = {node: i+1 for i, node in enumerate(node_list)}
G_num = nx.relabel_nodes(G, node_mapping)
reverse_mapping = {v: k for k, v in node_mapping.items()}

s1=node_mapping['HUGH']
s2=node_mapping['BONAVENTURE']
infl,beta,_=get_influence_centrality(G_num,s1,s2)


classif=get_class_sets(G_num,s1,s2)
z1_endorse={node for node in classif if classif[node]==1}
rem=[node for node in G_num.nodes() if node not in z1_endorse]
nodes_b = set()
for node in rem:
    nodes_b.update(G_num.successors(node))
A = nx.to_numpy_array(G, weight='weight')
A=A.T
n=A.shape[0]
F=calculate_Fe(A,beta,n)


R=1 


influ=np.zeros((R+1))

inf,_,_=get_influence_centrality(G_num,s1,s2)
influ[0]=inf[s1-1]
mod_edges=[]
new_edges=[]



for r in range(1,R+1):
    max_del=0
    max_w=0
    for b in nodes_b:
        sig=F[b-1,:].mean()
        for d in G_num.predecessors(b):
            if d in z1_endorse:
                continue
            w_tilde =  0.9*G_num[d][b]['weight']
            if d==2 and b==5:
                delta = calculate_delta(F, b-1, d-1, beta[s1-1][s1-1], 11-1, w_tilde,sig)
                print(delta)
                



