"""
cluster.py
"""
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
from TwitterAPI import TwitterAPI
from nltk.tokenize import word_tokenize
import pandas as pd
import networkx as networkx
import matplotlib.pyplot as plt
import numpy as np
##matplotlib inline

def create_graph():
	data=['Maryland1.csv','California1.csv','Oklahama1.csv','West Virginia1.csv']
	graph=nx.Graph()
	edges=[]
	for f in data:
		print(f)
		frame=pd.read_csv(f,sep='\t')
		#to=frame['friends'].tolist()
		
		for index,row in frame.iterrows():
			l=list(row['friends'].split(','))
			for x in l:
				edges.append((row['Id'],x))
	print(len(np.unique(edges)))	
	graph.add_edges_from(edges)

	nx.draw(graph)
	plt.show(block=False)
	plt.savefig("cluster.png")
	n=str(nx.nodes(graph))
	e=str(nx.info(graph))
	#print("Number of nodes="+n)
	print("Info="+e)
	return graph
def girvan_newman(depth=0):
    """ Recursive implementation of the girvan_newman algorithm.
    See http://www-rohan.sdsu.edu/~gawron/python_for_ss/course_core/book_draft/Social_Networks/Networkx.html
    
    Args:
    G.....a networkx graph

    Returns:
    A list of all discovered communities,
    a list of lists of nodes. """
    G=create_graph()
    if G.order() == 1:
        return [G.nodes()]
    min_nodes=10
    max_nodes=800
    result_components=[]
    file1=open("cluster.txt","w+")
    def largest(component):
    	max_n=0
    	index=0
    	for idx,l in enumerate(component):
    		if max_n<len(l.nodes()):
    			max_n=len(l.nodes())
    			index=idx
    	return index
	
    def find_best_edge(G0):
        eb = nx.edge_betweenness_centrality(G0)
        # eb is dict of (edge, score) pairs, where higher is better
        # Return the edge with the highest score.
        return sorted(eb.items(), key=lambda x: x[1], reverse=True)

   
    component = [c for c in nx.connected_component_subgraphs(G)]
     # calculate the component with the maximum nodes
    print(type(find_best_edge(G)))
    i=1;
    if(len(component)>1):
    	index=largest(component)
    else:
    	index=0

    indent = '   ' * depth  # for printing
    #print(max_n)
    print("Selected cluster with "+str(len(component[index].nodes())))
    components=[component[index]]
    #print(len(components))
    edrm=0
    for i,c in enumerate(component):
    	if i!=index:
    		result_components.append(c)

    #edge_to_remove = find_best_edge(component[index])
    flag=0
    ex=0
    while len(components) == 1 and ex==0:
    	g=nx.Graph(components[0])
    	edge_to_remove = find_best_edge(g)

    	edge =edge_to_remove[0][0]
    	print(indent + 'removing ' + str(edge))
    	g.remove_edge(*edge)

    	comp = [c for c in nx.connected_component_subgraphs(g)]
    	#print("Do banaya:"+str(len(comp)))
    	#print(result_components)
    	if( len(comp)>1):
    		components=comp
    		'''if(len(comp[0].nodes())>len(comp[1].nodes())):
    			if(len(comp[0].nodes())>max_nodes):
    				print("Itne bane"+str(len(comp[0].nodes())))
    				components=[comp[0]]
    				result_components.append(comp[1])
    			else:
    				ex=1
    				result_components.append(comp[0])
    				result_components.append(comp[1])
    		else:
    			if(len(comp[1].nodes())>max_nodes):
    				components=[comp[1]]
    				print("Itne bane"+str(len(comp[0].nodes())))
    				result_components.append(comp[0]) 
    			else:
    				ex=1
    				result_components.append(comp[0])
    				result_components.append(comp[1])'''
    	else:
    		components=[g]
    		flag=1

    #result = [c.nodes() for c in components]
    #print(indent + 'components=' + str(result))
    #for c in components:
     #   result.extend(girvan_newman(c, depth + 1))
    #print(result)
    result_components.extend(components)
    print("Resultant components=")
    print(len(result_components))
    file1.write("\n Total number of communities discovered="+str(len(result_components)))
    avg= len(G.nodes())/len(result_components)
    file1.write("\n Average number of user per community ="+str(avg))
    file1.close()

    #print("cluster 1 is "+strtr(len(components[0].nodes())))
    #print("cluster 2 is "+str(len(components[1].nodes())))
    return components
def main():
    """ Main method. You should not modify this. """
    #print("le bhai data")
    girvan_newman()



if __name__ == '__main__':
    main()