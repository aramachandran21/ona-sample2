import csv
from operator import itemgetter
import networkx as nx
from networkx.algorithms import community
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import pandas as pd
import os
from itertools import product
with open('nodelist.csv', 'r') as nodecsv: # Open the file
 nodereader = csv.reader(nodecsv) # Read the csv
 # Retrieve the data (using Python list comprhension and list slicing to remove the header row, see footnote 3)
 nodes = [n for n in nodereader][1:]
node_names = [n[0] for n in nodes] # Get a list of only the node names
path = "C:/users/arvind.ramachandran/.spyder-py3/Images"
dirs = os.listdir(path)
d = { int(stu.split(".")[0]) : path+"/"+stu for stu in dirs } 
lookup={1:'one',2:'two',3:'three',4:'four',5:'five',6:'six',7:'seven',8:'eight',9:'nine',10:'ten',11:'eleven',12:'twelve',13:'thirteen',14:'fourteen',15:'fifteen',16:'sixteen',17:'seventeen',18:'eighteen',19:'nineteen',20:'twenty',21:'twenty one',22:'twenty two',23:'twenty three',24:'twenty four',25:'twenty five',26:'twenty six',27:'twenty seven',28:'twenty eight',29:'twenty nine',30:'thirty',31:'thirty one',32:'thirty two',33:'thirty three',34:'thirty four',35:'thirty five',36:'thirty six',37:'thirty seven',38:'thirty eight',39:'thirty nine',40:'forty',41:'forty one',42:'forty two',43:'forty three',44:'forty four',45:'forty five',46:'forty six'
}
# with open('sampledata.csv', 'r') as edgecsv: # Open the file
# edgereader = csv.reader(edgecsv) # Read the csv
# edges = [tuple(e) for e in edgereader][1:] # Retrieve the data
df = pd.read_csv('sampledata.csv', sep =',') 
G = nx.from_pandas_edgelist(df, source = 'source', target = 'target', 
 edge_attr = 'score',create_using = nx.DiGraph())
centrality_nx = nx.eigenvector_centrality(G, weight='score')
sorted_ev = sorted(centrality_nx.items(), key=itemgetter(1), reverse=True)
centrality_nx1 = nx.eigenvector_centrality(G,weight='NA')
sorted_ev1 = sorted(centrality_nx1.items(), key=itemgetter(1), reverse=True)
print(sorted_ev)
print(sorted_ev1)
#print(sorted_ev)
out_dict = {}
for idx, (key, _) in enumerate(sorted_ev):
 out_dict[key] = idx + 1
#print(out_dict)
df = pd.DataFrame(list(centrality_nx.items()),columns = ['node','size'])
nodename = [row['node'] for index, row in df.iterrows()]
nodesize = [row['size']*100 for index, row in df.iterrows()]
for i in range(0, len(df)):
 G.add_node(nodename[i], size=nodesize[i])
filt = []
def draw_graph3(networkx_graph,notebook=True,
 show_buttons=True,only_physics_buttons=False):
 
 import pyvis
 from pyvis import network as net
 
 # make a pyvis network
 pyvis_graph = net.Network(height="1500px", width="100%",bgcolor="#222222",
 font_color="white")
 
 #pyvis_graph.hrepulsion(node_distance=140, central_gravity=0.0, spring_length=100, spring_strength=0.01, damping=0.09)
 
 for node in networkx_graph.nodes(data=True):
 hv=node[0]
 filt=node[0]
 #neighbors = pyvis_graph.neighbors(hv)
 #neighbor_map = pyvis_graph.get_adj_list()
 if filt in d: 
 pyvis_graph.add_node(str(node[0]),shape='image', image =d[filt]
 ,size=15,title="Name: "+str(lookup[hv])+"<br>"+"Influencer Rank: "+str(out_dict[hv]))
 else:
 pyvis_graph.add_node(str(node[0]),size=15,title="Name: "+str(lookup[hv])+"<br>"+"Influencer Rank: "+str(out_dict[hv]))
 for edges in networkx_graph.edges(data=True):
 edge_color=edges[2]['score']
 pyvis_graph.add_edge(str(edges[0]),str(edges[1]),
 width=edge_color)
 neighbor_map = pyvis_graph.get_adj_list()
 list_neighbors = pyvis_graph.neighbors(str(node[0]))
 
 pyvis_graph.barnes_hut()
 if show_buttons:
 pyvis_graph.width="100%"
 pyvis_graph.show_buttons(filter_=['nodes','edges','physics'])
 else:
 pyvis_graph.width="100%"
 pyvis_graph.show_buttons(filter_=['nodes','edges','physics'])
 pyvis_graph.set_options("""
 var options = {
 "configure": {
 "enabled": true,
 "filter": [
 "nodes",
 "edges",
 "physics"
 ]
 },
 "nodes": {
 "color": {
 "highlight": {
 "border": "rgba(152,233,74,1)"
 }
 }
 },
 "edges": {
 "color": {
 "highlight": "rgba(59,132,58,1)",
 "inherit": false
 },
 "smooth": false
 },
 "physics": {
 "barnesHut": {
 "gravitationalConstant": -80000,
 "springLength": 250,
 "springConstant": 0.001
 },
 "minVelocity": 0.75
 },
 "enabled": true,
 "stabilization": {
 "enabled": true,
 "fit": true,
 "iterations": 1000,
 "onlyDynamicEdges": false,
 "updateInterval": 50
 }
}""") 
 pyvis_graph.show("graph_new1.html")
draw_graph3(G)
