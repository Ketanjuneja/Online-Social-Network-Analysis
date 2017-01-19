from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
from TwitterAPI import TwitterAPI
from nltk.tokenize import word_tokenize
import pandas as pd

def main():
    """ Main method. You should not modify this. """
    #print("le bhai data")
    #collect_data()
    pos=0
    neg=0
    ps=pd.DataFrame(columns=['tweets', 'Id'])
    ns=pd.DataFrame(columns=['tweets', 'Id'])
    frame=pd.read_csv("train.csv",sep=',',encoding = "ISO-8859-1")
    print(frame.loc[800500])
    for i in range(0,499):
    	row=frame.loc[i]
    	l=[str(row[5]),0]
    	ns.loc[len(ns)]=l
    for i in range(0,499):
   		row=frame.loc[800000+i]
   		l=[str(row[5]),1]
   		ps.loc[len(ps)]=l

    """for index,row in frame.iterrows():
    	label=row[0]
    	print(label)
    	print(type(label))
    	if label==0:
    		l=[str(row[5]),0]
    		ns.loc[len(ns)]=l
    		neg=neg+1
    	if label==4:
    		l=[str(row[5]),0]
    		ps.loc[len(ps)]=l
    		pos=pos+1

    	if pos>500 and neg>500:
    		break
	"""
    print(ps)
    ps.to_csv("pos.csv",sep="\t",encoding="utf-8")
    ns.to_csv("neg.csv",sep="\t",encoding="utf-8")




if __name__ == '__main__':
    main()
