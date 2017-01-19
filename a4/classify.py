"""
classify.py
"""
from collections import Counter, defaultdict
from itertools import chain, combinations
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import pandas as pd
from scipy.sparse import csr_matrix
import string
import nltk
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
def tokenize(docs, keep_internal_punct=False):
    """
    Tokenize a string.
    The string should be converted to lowercase.
    If keep_internal_punct is False, then return only the alphanumerics (letters, numbers and underscore).
    If keep_internal_punct is True, then also retain punctuation that
    is inside of a word. E.g., in the example below, the token "isn't"
    is maintained when keep_internal_punct=True; otherwise, it is
    split into "isn" and "t" tokens.

    Params:
      doc....a string.
      keep_internal_punct...see above
    Returns:
      a numpy array containing the resulting tokens.

    >>> tokenize(" Hi there! Isn't this fun?", keep_internal_punct=False)
    array(['hi', 'there', 'isn', 't', 'this', 'fun'], 
          dtype='<U5')
    >>> tokenize("Hi there! Isn't this fun? ", keep_internal_punct=True)
    array(['hi', 'there', "isn't", 'this', 'fun'], 
          dtype='<U5')
    >>> tokenize("??necronomicon?? geträumte sünden.<br>Hi", True)
    array(['necronomicon', 'geträumte', 'sünden.<br>hi'], 
      dtype='<U13')
    >>> tokenize("??necronomicon?? geträumte sünden.<br>Hi", False)
    array(['necronomicon', 'geträumte', 'sünden', 'br', 'hi'], 
      dtype='<U12')
    """
    doc=str(docs)
    l= doc.lower()
    l1=[]
    if  not keep_internal_punct :
        l1=re.sub("\W+",' ',l).split()
    else:
      for x in l.split():
        l1.append(x.strip(string.punctuation))
        '''l1=[token.strip(string.punctuation) for token in l.split()]'''
    return np.array([d.lower() for d in l1 if len(d)>=3])

def create_csv(vocab,x):
	r=[]
	c=[]
	d=[]
	i=0
	for l in x:
		wordlist=nltk.FreqDist(l)
		for a,b in wordlist.items():
			if a in vocab.keys():
				d.append(b)
				r.append(i)
				c.append(vocab[a])
		i=i+1
	data=np.array(d)
	row=np.array(r)
	col=np.array(c)
	X=csr_matrix((data,(row,col)),shape=(i,len(vocab.keys())))	
	
	return X

def classify():
	data=['Maryland.csv','California.csv','Oklahama.csv','West Virginia.csv']
	###colect positive and negative tweets and train 
	
	x=[]
	y=[]
	labels=[]
	file1 = open('classify.txt', 'w+')
	ps=pd.read_csv("pos.csv",sep='\t')
	tweets=ps['tweets'].tolist()
	for t in tweets:
		l=list(tokenize(t))
		x.extend(l)
		y.append(l)
		labels.append(1)


	ns=pd.read_csv("neg.csv",sep='\t')
	tweetsn=ns['tweets'].tolist()
	for t in tweetsn:
		l=list(tokenize(t))
		x.extend(l)
		y.append(l)
		labels.append(0)

	wordlist=nltk.FreqDist(x)
	#feats=wordlist.keys()
	#print(y)
	vocab=dict()
	min_freq=2
	i=0
	for key,values in wordlist.items():
		if values>=min_freq:
		  vocab[key]=i
		  i+=1
	X=create_csv(vocab,y)
	
	#clf=GaussianNB()
	clf=LogisticRegression()
	clf.fit(X.toarray(),labels)
	posv=0
	negv=0
	flag=0
	for name in data:
		y=[]
		ns=pd.read_csv(name,sep='\t')
		tweetsn=ns['text'].tolist()
		#print(tweetsn)
		for t in tweetsn:
			l=list(tokenize(t))
			y.append(l)
		
		X=create_csv(vocab,y)
		lb=clf.predict(X.toarray())
		lab=list(lb)
		print("For State "+name+":")
		print("Total tweets:"+str(len(lb)))
		print("Total positive tweets:"+str(lab.count(1)))
		print("Total negative tweets:"+str(lab.count(0)))
		posv=posv+lab.count(1)
		negv=negv+lab.count(0)
		if flag==0:
			#print(len(X))
			ind=lab.index(1)
			pt=tweetsn[ind]
			ind=lab.index(0)
			nt=tweetsn[ind]

	file1.write("\n Total number of positive tweets:"+str(posv))
	file1.write("\n Example of positive tweet:"+pt)
	file1.write("\n Total number of negative tweets:"+str(negv))
	file1.write("\n Example of negative tweet:"+nt)
	file1.close()
		#print(lb)
		#print(len(lb))
	#print(y)		
	#for f in data:



def main():
    """ Main method. You should not modify this. """
    #print("le bhai data")
    print(tokenize("RT @marcobonzanini: just an example! :D http://example.com #NLP"))
    classify()
if __name__ == '__main__':
    main()
