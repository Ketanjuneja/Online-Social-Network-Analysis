"""
sumarize.py
"""
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
from TwitterAPI import TwitterAPI
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
def main():
	output=['collect.txt','cluster.txt','classify.txt']
	sumarize=open("sumarize.txt","w+")
	for r in output:
		file1=open(r, 'r')
		comp=file1.read()
		lines=comp.splitlines()
		for l in lines:
			sumarize.write("\n "+l)



if __name__ == '__main__':
    main()
