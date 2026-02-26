import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse.linalg import gmres
from scipy.sparse import identity
import scipy.sparse as sp
import matplotlib.pyplot as plt
from pyvis.network import Network
plt.rcParams.update({
    'font.size': 15,
    'axes.titlesize': 15,
    'axes.labelsize': 15,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 20,
    'pdf.fonttype': 42,  # Use Type 42 (TrueType)
    'ps.fonttype': 42,    # Use Type 42 (TrueType)
    'font.family': 'Arial'
})
def generate_custom_beta(n, p, seed):

    np.random.seed(seed)

    beta_diag = np.zeros(n)

    indices = np.random.choice(n, size=p, replace=False)

    s1 = np.random.choice(indices, size=1)
    rem = [i for i in indices if i != s1]
    beta_diag[indices]=np.random.uniform(0.1,0.3,size=p)


    return np.diag(beta_diag), s1[0], indices

def generate_random_graph(n, p, seed):
    G = nx.erdos_renyi_graph(n=n, p=p, directed=True,seed=seed)

    # For each node, set incoming edge weights so they sum to 1
    in_degrees = dict(G.in_degree())

    for node in G.nodes:
        indeg = in_degrees[node]
        if indeg > 0:
            incoming_edges = G.in_edges(node)
            weight = 1.0 / indeg
            for u, v in incoming_edges:
                G[u][v]['weight'] = weight


    return G

def calculate_Fe(W,beta,n):
    return np.linalg.solve(np.eye(n)-np.dot((np.eye(n)-beta),W),np.eye(n))

def get_endorsers(G,s1,rem):
    G1=G.copy()
    for u in list(G.predecessors(s1)):
        G1.remove_edge(u, s1)
    H=nx.condensation(G1)
    # print(s1)
    print(H.graph["mapping"])

def calculate_Fe_iterative(W_sparse, beta, n):
    I = identity(n, format='csr')
    A = I - (I - beta) * W_sparse  # equivalent to: A = I - (I - beta) * W
    Fe = np.zeros((n, n))
    
    # Solve A X = I column by column (or use block solvers for speedup)
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1
        x, _ = gmres(A, e_i, atol=1e-6)
        Fe[:, i] = x

    return Fe

def calculate_delta(F,b,d,beta_s1,s1,n,w_tilde,sig):

    return  beta_s1*w_tilde*sig*(F[s1][s1]-F[d][s1])/(1-w_tilde*F[s1][b]+w_tilde*F[d][b])

def update_inverse(F,a,b,d,w_tilde):
    u = (w_tilde * np.eye(n)[b]).reshape(n, 1)      # n×1 column vector
    v = (np.eye(n)[a] - np.eye(n)[d]).reshape(1, n)
    # print(F @ u @ v @ F/(1-w_tilde*F[a][b]+w_tilde*F[d][b]))
    return F+F @ u @ v @ F/(1-w_tilde*F[a][b]+w_tilde*F[d][b])

def get_influence_centrality(G,beta):

    adj = nx.to_numpy_array(G, weight='weight')  # includes weights if present
    adj=adj.T


    # print(adj)
    length=adj.shape[0]
    P=np.dot(np.linalg.inv(np.eye(length)-np.dot((np.eye(length)-beta),adj)),beta)
    return np.dot(np.ones(length).T,P)/length


n = 1000
p =  (np.log(n) +2.5)/ n
# p=0.009

seed = 42
print(p)
G = generate_random_graph(n,p,seed)

print(G.number_of_edges())
if nx.is_strongly_connected(G):
    print("Graph is strongly connected.")
beta,s1,agents=generate_custom_beta(n,10,seed)
# nt = Network(height="800px", width="100%", notebook=False)
# nt.barnes_hut()
# nt.from_nx(G)
# for node in nt.nodes:
#     if node['id'] == s1:
#         node['size'] = 100  # Increased size for more visibility
#         node['color'] = 'rgba(255, 50, 50, 1.0)'  # Brighter red with full opacity
#     elif node['id'] in agents:
#         node['size'] = 75
#         node['color'] = 'rgba(230, 0, 0, 0.7)'  # Slightly less bright with reduced opacity
#     else:
#         node['size'] = 45
#         node['color'] = 'rgba(255, 165, 0, 0.7)'  # orange

# for edge in nt.edges:
#     edge['width'] = 0.001
#     edge['smooth'] = False
#     edge['color'] = 'rgba(0,0,0,0.2)'

# nt.show('nx.html', notebook=False)
# print(beta)
A = nx.to_numpy_array(G, weight='weight')
A=A.T
F=calculate_Fe_iterative(A,beta,n)


# get_endorsers(G,s1,rem)
# s1_endorsers=
R=60
# # # print(A.T)
influ=np.zeros((R+1))
influ[0]=get_influence_centrality(G,beta)[s1]
mod_edges=[]
new_edges=[]

for r in range(1,R+1):
    max_del=0
    max_w=0
    for b in G.nodes:
        sig=F[:,b].mean()
        if(b==s1):
            continue
        for d in G.predecessors(b):
            if(d==s1):
                continue
            w_tilde =  0.9*G[d][b]['weight']
            if([d,b] in mod_edges):
                continue
            delta = calculate_delta(F, b, d, beta[s1][s1], s1, n, w_tilde,sig)
            if delta > max_del:
                max_del = delta
                max_d = d
                max_b = b
                max_w = w_tilde
                
    F=update_inverse(F,s1,max_b,max_d,max_w)
    if G.has_edge(s1, max_b):
        G[s1][max_b]['weight'] += max_w  
    else:
        G.add_edge(s1, max_b, weight=max_w)
    G[max_d][max_b]['weight'] -= max_w 
    mod_edges.append([max_d,max_b])
    new_edges.append([s1,max_b])
    # print(max_del)
    influ[r]=get_influence_centrality(G,beta)[s1]
    if(r%10==0):
        print(r)
# new_edges=set(new_edges)

nt2 = Network(height="800px", width="100%", notebook=False)
nt2.barnes_hut()
nt2.from_nx(G)
for node in nt2.nodes:
    if node['id'] == s1:
        node['size'] = 100  # Increased size for more visibility
        node['color'] = 'rgba(255, 50, 50, 1.0)'  # Brighter red with full opacity
    elif node['id'] in agents:
        node['size'] = 75
        node['color'] = 'rgba(230, 0, 0, 0.7)'
    else:
        node['size'] = 45
        node['color'] = 'rgba(255, 165, 0, 0.7)'  # orange
for edge in nt2.edges:
    if [edge['from'], edge['to']] in new_edges:
        edge['width'] = 20
        edge['smooth'] = False
        edge['color'] = 'rgba(0,0,0, 1)'
    else:
        edge['width'] = 0.001
        edge['smooth'] = False
        edge['color'] = 'rgba(0,0,0,0.2)'
nt2.show('nx_mod.html',notebook=False)
plt.plot(range(R+1), influ, linestyle='-')
plt.xlabel('No of Edge modifications')
plt.ylabel('Influence centrality \nof target stubborn agent')
plt.ylim(0, 1)
x_markers = range(0, R+1, 10)
y_markers = [influ[i] for i in x_markers]
plt.scatter(x_markers, y_markers, color='red', zorder=5)
# Show x-axis ticks every 10 steps
plt.xticks(ticks=x_markers)
for idx, i in enumerate(x_markers):
    if idx == 0:
        plt.annotate(f"{influ[i]:.2f}", (i, influ[i]),
                     textcoords="offset points", xytext=(20, 0), ha='center')
    else:
        plt.annotate(f"{influ[i]:.2f}", (i, influ[i]),
                     textcoords="offset points", xytext=(0, 10), ha='center')
plt.tight_layout()
plt.savefig("influence_centrality_er.pdf", format='pdf', bbox_inches='tight')
plt.show()
