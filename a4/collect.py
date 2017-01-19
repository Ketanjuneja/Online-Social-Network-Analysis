"""
collect.py
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
consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''


def get_twitter():
    """ Construct an instance of TwitterAPI using the tokens you entered above.
    Returns:
      An instance of TwitterAPI.
    """
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)

def robust_request(twitter, resource, params, max_tries=5):
    """ If a Twitter request fails, sleep for 15 minutes.
    Do this at most max_tries times before quitting.
    Args:
      twitter .... A TwitterAPI object.
      resource ... A resource string to request; e.g., "friends/ids"
      params ..... A parameter dict for the request, e.g., to specify
                   parameters like screen_name or count.
      max_tries .. The maximum number of tries to attempt.
    Returns:
      A TwitterResponse object, or None if failed.
    """
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)
def get_friends(twitter, screen_name):
   
   
    ###TODO
    send=[]
    string= screen_name+'&count=5000'
    response=robust_request(twitter,"friends/list",{'screen_name':screen_name,'count':'200'},5)
    for r in response:
    	#print(r)
    	send.append(r['screen_name'])
        #print(sorted(send, key=int))
    return send


def collect_data():
	#filename="states.txt"
	twitter = get_twitter()
	screen=[]
	coords=dict()
	users=[]
	tweets=0
	file1 = open('collect.txt', 'w+')
	#states=[]
	#f = open(filename, 'r') 
	#'geocode':'38.9071923 -77.03687070000001 50mi' - Washington 
	#39.045755,-76.641271 - Maryland
	# 36.778261 -119.41793239999998 - California
	# 35.0077519 -97.09287699999999- Oklahama
	# 38.59762619999999 -80.45490259999997- West Virginia 
	coords['Maryland']=('39.045755,-76.641271,50mi')
	coords['California']=('36.778261,-119.41793239999998,50mi')
	coords['Oklahama']=('35.0077519,-97.09287699999999,50mi')
	coords['West Virginia']=('38.59762619999999,-80.45490259999997,50mi')

	#line=f.readline()[0:-1]
	data=dict()

	for b,c in coords.items():
		response=robust_request(twitter,"search/tweets",{'q':'#Trump','count':'200','geocode':c},5)
		#print(response)
		
		#data[b]=response
		i=0
		res=[]
		ulist=[]
		for r in response:
			res.append(r)
			if not r['user']['screen_name'] in ulist:
				if i<5:
					ulist.append(r['user']['screen_name'])
				tweets=tweets+1	
				users.append(r['user']['screen_name'])		
				
			i=i+1
		data[b]=ulist
		#print(r['place'])
			#print(r['text'])
		frame=pd.DataFrame(res)
		fname=b+".csv"
		frame.to_csv(fname,sep='\t')

	
	for b,c in data.items():
		clus=pd.DataFrame(columns=['Id', 'friends'])
		for a in c:
			clus.loc[len(clus)]=[a,get_friends(twitter,a)]
		fname=b+"1.csv"
		clus.to_csv(fname,sep='\t')
	print(data)
	u=np.unique(np.array(users))
	file1.write("\n Number of users Collected:"+str(len(u)))
	file1.write("\n Number of messages Collected:"+str(tweets))
	file1.close()



def main():
    """ Main method. You should not modify this. """
    #print("le bhai data")
    collect_data()


if __name__ == '__main__':
    main()

