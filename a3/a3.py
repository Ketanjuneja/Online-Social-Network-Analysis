# coding: utf-8

# # Assignment 3:  Recommendation systems
#
# Here we'll implement a content-based recommendation algorithm.
# It will use the list of genres for a movie as the content.
# The data come from the MovieLens project: http://grouplens.org/datasets/movielens/

# Please only use these imports.
from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.

    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns: 
      The movies DataFrame, augmented to include a new column called 'tokens'.

    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    ###TODO
    #print(movies)
    movies['tokens']=movies['genres'].map(lambda x: tokenize_string(x))
    return movies
    pass


def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i

    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
    """
    ###TODO
    l=movies['tokens'].tolist()
    #ar=list(np.array(l).flat)
    #ar=reduce(operator.concat,l)
    ar=sum(l,[])
    c=Counter()
    for x in l:
        un=[]
        un= list(np.unique(x))
        c.update(un)
    #print(c)
    a=np.unique(ar)
    df=dict(c)
    #print(df)

    v1=sorted(a)
    vocab_length=len(v1)
    #print(v1)
    vocab={x:i for i,x in enumerate(v1)}
    #print(vocab)
    N=len(movies)
    final=[]
    for index,row in movies.iterrows():
        c1=Counter()
        l1=[]
        l1=row['tokens']
        c1.update(l1)
        r=[]
        c=[]
        d=[]
        x,max_k=c1.most_common()[0]
        #print(c1)
        #print(x)
        #print(max_k)
        #break;
        res=dict(c1)
        for x,y in c1.items():
            tf=y
            df_v=df[x]
            val=(tf/max_k)*math.log10(N/df_v)
            #print(tf/max_k)
            #print(N)
            #print(df_v)

            #print(val)
            #break
            r.append(0)
            c.append(vocab[x])
            d.append(val)
        data=np.array(d)
        row1=np.array(r)
        col=np.array(c)    
        #movies.loc[i,'features']=csr_matrix((data,(row1,col)),shape=(1,vocab_length))
        final.append(csr_matrix((data,(row1,col)),shape=(1,vocab_length)).toarray())
        #print(csr_matrix((data,(row1,col)),shape=(1,vocab_length)).toarray())
        #break
    movies['features']=final
    return movies,vocab
    pass


def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """
    ###TODO
    def norm_form(a):
        return np.sqrt(np.sum(np.square(a)))
    a1=a.values[0]
    b1=b.values[0]
    #print(a1)
    #print(b1)
    num=np.dot(a1,np.transpose(b1))
    #print(type(num))
    #np.sqrt(sum(np.square()))
    d=norm_form(a1)*norm_form(b1)    
    #print(num[0]/d)
    return num[0]/d
    pass


def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.

    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.

    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.

    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """
    ###TODO
    result=[]
    for index,row in ratings_test.iterrows():
        user_id=row['userId']
        movie_id=row['movieId']
        movie_csr=movies.loc[movies['movieId']==movie_id]
        mvc=movie_csr['features']
        #print(type(mvc))

        avg=0
        cnt=0
        mv=ratings_train.loc[ratings_train['userId']==user_id]
        s=[]
        r=[]
        for index1,row1 in mv.iterrows():
            #print(row1)
            mvid=row1['movieId']
            rate=row1['rating']
            r.append(rate)
            mvc2=movies['features'][movies['movieId']==mvid ]
            sim=cosine_sim(mvc,mvc2)
            s.append(sim)
            #cnt=cnt+1
            #if(sim>0):
             #   avg=avg+sim*rate
            #else:
             #   avg=avg+rate
        sum1=np.sum(s)
        #print(sum1)  
        if(sum1>0):
            n=np.sum([a*b for a,b in zip(s,r)])
            final=n/sum1
        else:
            final=np.mean(r)
        #print(final)   
        result.append(final)
    #print(result)
    return np.array(result)

    pass


def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    #print(ratings)
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()
