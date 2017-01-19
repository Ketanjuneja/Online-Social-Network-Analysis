# coding: utf-8

"""
CS579: Assignment 2

In this assignment, you will build a text classifier to determine whether a
movie review is expressing positive or negative sentiment. The data come from
the website IMDB.com.

You'll write code to preprocess the data in different ways (creating different
features), then compare the cross-validation accuracy of each approach. Then,
you'll compute accuracy on a test set and do some analysis of the errors.

The main method takes about 40 seconds for me to run on my laptop. Places to
check for inefficiency include the vectorize function and the
eval_all_combinations function.

Complete the 14 methods below, indicated by TODO.

As usual, completing one method at a time, and debugging with doctests, should
help.
"""

# No imports allowed besides these.
from collections import Counter, defaultdict
from itertools import chain, combinations
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.sparse import csr_matrix
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
import string
import tarfile
import urllib.request
feats = defaultdict(lambda: 0)

def download_data():
    """ Download and unzip data.
    DONE ALREADY.
    """
    url = 'https://www.dropbox.com/s/xk4glpk61q3qrg2/imdb.tgz?dl=1'
    urllib.request.urlretrieve(url, 'imdb.tgz')
    tar = tarfile.open("imdb.tgz")
    tar.extractall()
    tar.close()


def read_data(path):
    """
    Walks all subdirectories of this path and reads all
    the text files and labels.
    DONE ALREADY.

    Params:
      path....path to files
    Returns:
      docs.....list of strings, one per document
      labels...list of ints, 1=positive, 0=negative label.
               Inferred from file path (i.e., if it contains
               'pos', it is 1, else 0)
    """
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'pos', '*.txt'))])
    data = [(1, open(f).readlines()[0]) for f in sorted(fnames)]
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'neg', '*.txt'))])
    data += [(0, open(f).readlines()[0]) for f in sorted(fnames)]
    data = sorted(data, key=lambda x: x[1])
    #print(data)
    return np.array([d[1] for d in data]), np.array([d[0] for d in data])


def tokenize(doc, keep_internal_punct=False):
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

    l= doc.lower()
    l1=[]
    if  not keep_internal_punct :
        l1=re.sub("\W+",' ',l).split()
    else:
      for x in l.split():
        l1.append(x.strip(string.punctuation))
        '''l1=[token.strip(string.punctuation) for token in l.split()]'''
    return np.array([d.lower() for d in l1])     


    pass


def token_features(tokens, feats):
    """
    Add features for each token. The feature name
    is pre-pended with the string "token=".
    Note that the feats dict is modified in place,
    so there is no return value.

    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.

    >>> feats = defaultdict(lambda: 0)
    >>> token_features(['hi', 'there', 'hi'], feats)
    >>> sorted(feats.items())
    [('token=hi', 2), ('token=there', 1)]
    """
    ###TODO
    s="token="
    c=Counter()
    c.update(list(tokens))
    '''for x in l:
      cnt=l.count(x)
      key=s+x
      feats[key]=cnt'''
    for x,y in c.most_common():
      key=s+x
      feats[key]=y
    pass


def token_pair_features(tokens, feats, k=3):
    """
    Compute features indicating that two words occur near
    each other within a window of size k.

    For example [a, b, c, d] with k=3 will consider the
    windows: [a,b,c], [b,c,d]. In the first window,
    a_b, a_c, and b_c appear; in the second window,
    b_c, c_d, and b_d appear. This example is in the
    doctest below.
    Note that the order of the tokens in the feature name
    matches the order in which they appear in the document.
    (e.g., a__b, not b__a)

    Params:
      tokens....array of token strings from a document.
      feats.....a dict from feature to value
      k.........the window size (3 by default)
    Returns:
      nothing; feats is modified in place.

    >>> feats = defaultdict(lambda: 0)
    >>> token_pair_features(np.array(['a', 'b', 'c', 'd']), feats)
    >>> sorted(feats.items())
    [('token_pair=a__b', 1), ('token_pair=a__c', 1), ('token_pair=b__c', 2), ('token_pair=b__d', 1), ('token_pair=c__d', 1)]
    """
    ###TODO
    s="token_pair="
    l=list(tokens)
    l1=[l[i-1:i-1+k] for i in range(1,len(l))]
    #print(l1)
    l2=[]
    for x in l1:
      if len(x)>=k:
        #l1.remove(x)a=1
        l2.append(x)
    ###l=set(combinations(tokens,k))
    #print(l2)
    c=Counter()
    for x in l2:
      ##print(x)
      combi=list(combinations(x,2))
      #print(combi)
      c.update(combi)

      '''for y in combi:
        key=s+str(y[0])+str('__')+str(y[1])
        feats[key]+=1'''
    for x,y in c.most_common():
      a,b=x
      key=s+str(a)+str('__')+str(b)
      feats[key]=y
    
    pass


neg_words = set(['bad', 'hate', 'horrible', 'worst', 'boring'])
pos_words = set(['awesome', 'amazing', 'best', 'good', 'great', 'love', 'wonderful'])

def lexicon_features(tokens, feats):
    """
    Add features indicating how many time a token appears that matches either
    the neg_words or pos_words (defined above). The matching should ignore
    case.

    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.

    In this example, 'LOVE' and 'great' match the pos_words,
    and 'boring' matches the neg_words list.
    >>> feats = defaultdict(lambda: 0)
    >>> lexicon_features(np.array(['i','love','love','LOVE', 'this', 'great', 'boring', 'movie']), feats)
    >>> sorted(feats.items())
    [('neg_words', 1), ('pos_words', 2)]
    """
    ###TODO
    ##case=[x.lower() for x in tokens]
    #low=[x.lower() for x in tokens]
    for x in tokens:
      if x.lower() in pos_words:
        feats['pos_words']+=1
      if x.lower() in neg_words:
        feats['neg_words']+=1
    #for x in tokens:
      
      #s=list(x.lower())
      #low=set(s)
      #print(low)
      #print(len(pos_words & low))
      #if len(pos_words & low)>1:
    #feats['pos_words']=len(pos_words & low)
      #if len(neg_words & set(low))>1:
    #feats['neg_words']=len(neg_words & low)  
    pass


def featurize(tokens, feature_fns):
    """
    Compute all features for a list of tokens from
    a single document.

    Params:
      tokens........array of token strings from a document.
      feature_fns...a list of functions, one per feature
    Returns:
      list of (feature, value) tuples, SORTED alphabetically
      by the feature name.

    >>> feats = featurize(np.array(['i', 'LOVE', 'this', 'great', 'movie','Great','movie']), [token_pair_features, lexicon_features])
    >>> feats
    [('neg_words', 0), ('pos_words', 2), ('token=LOVE', 1), ('token=great', 1), ('token=i', 1), ('token=movie', 1), ('token=this', 1)]
    """
    ###TODO
    
    f=[]
    ##feats['pos_words']=0
    feats=defaultdict(lambda: 0)
    def run_fn(fn, x, y):
      return fn(x, y)
    if isinstance(feature_fns, tuple):
      ##print('mai idhar hu')
      ##for x,y in feature_fns:
      x,y=feature_fns
      run_fn(x,tokens,feats)
      run_fn(y,tokens,feats)      
    elif isinstance(feature_fns,list):
      for x in feature_fns:
        run_fn(x,tokens,feats)
    else:
      run_fn(feature_fns,tokens,feats)
    
    '''for x,y in feats.items():
      t=(x,y)
      f.append(t)
    ##print(f)
    result=sorted(f,key=lambda x:x[0])'''
    ##return list(zip(feats.keys(),feats.values()))  
    ###return sorted([(a,b) for a,b in feats.items()])
    return sorted(zip(feats.keys(),feats.values()))
    pass


def vectorize(tokens_list, feature_fns, min_freq, vocab=None):
    """
    Given the tokens for a set of documents, create a sparse
    feature matrix, where each row represents a document, and
    each column represents a feature.

    Params:
      tokens_list...a list of lists; each sublist is an
                    array of token strings from a document.
      feature_fns...a list of functions, one per feature
      min_freq......Remove features that do not appear in
                    at least min_freq different documents.
    Returns:
      - a csr_matrix: See https://goo.gl/f5TiF1 for documentation.
      This is a sparse matrix (zero values are not stored).
      - vocab: a dict from feature name to column index. NOTE
      that the columns are sorted alphabetically (so, the feature
      "token=great" is column 0 and "token=horrible" is column 1
      because "great" < "horrible" alphabetically),

    >>> docs = ["Isn't this movie great?", "Horrible, horrible movie"]
    >>> tokens_list = [tokenize(d) for d in docs]
    >>> feature_fns = [token_features]
    >>> X, vocab = vectorize(tokens_list, feature_fns, min_freq=1)
    >>> type(X)
    <class 'scipy.sparse.csr.csr_matrix'>
    >>> X.toarray()
    array([[1, 0, 1, 1, 1, 1],
           [0, 2, 0, 1, 0, 0]], dtype=int64)
    >>> sorted(vocab.items(), key=lambda x: x[1])
    [('token=great', 0), ('token=horrible', 1), ('token=isn', 2), ('token=movie', 3), ('token=t', 4), ('token=this', 5)]
    """
    ###TODO
    ##tokenize and calculate count 
    ### vocab = featurize()
    vocab1=defaultdict(lambda: 0)
    vb=defaultdict(lambda: 0)
    l=[]
    l5=[]
    for tokens in tokens_list:
      vo=featurize(tokens,feature_fns)
      l.append(vo)
      l5+=vo
    
    if vocab==None:
      vocab=defaultdict(lambda:0)
      for tup in l5:
        #print(tup)
        a,b=tup
        if b>0:
          vocab1[a]+=1

      i=0
      voc=sorted(vocab1.items(), key=lambda x:x[0])    
      #print(voc)
      for key,value in voc:
        if value>=min_freq:
          vocab[key]=i
          i+=1
   
    vocab2=list(vocab.keys())
    column=sorted(vocab2)
    #print('column mila')
    
    #row=np.array([0])
    #col=np.array([0])
    #data=np.array([0])
    r=[]
    c=[]
    d=[]
    i=0
    if len(column)>0:
      for x in l:
        #c_vocab=featurize(l,feature_fns)
        #print('featurize kiya')
        #j=0
        for a,b in x:
          #t=[b for a,b in c_vocab if a==x ]  
          if a in vocab.keys():
            #data=np.append(data,y)
            #row=np.append(row,i)
            #col=np.append(col,vb[x])
            d.append(b)
            r.append(i)
            c.append(vocab[a])
            

        i=i+1


    #print(len(d))
    #  print(i)
    '''if len(column)>0:
      for l in tokens_list:
        c_vocab=featurize(l,feature_fns)
        #print('featurize kiya')
        j=0
        for x,y in c_vocab:
          #t=[b for a,b in c_vocab if a==x ]  
          if vb[x]>=0:
            #data=np.append(data,y)
            #row=np.append(row,i)
            #col=np.append(col,vb[x])
            d.append(y)
            r.append(i)
            c.append(vb[x])
              
        i=i+1
    #print('matrix bana')  '''  
    data=np.array(d)
    row=np.array(r)
    col=np.array(c)
    if len(column)>0:
      x=csr_matrix((data,(row,col)),shape=(i,len(column)))
      #print('matrix diya')
    else:
      x=None
      vb=None    
    # print(x.toarray())
    ##print(vocab)
    
    return x,vocab
    pass


def accuracy_score(truth, predicted):
    """ Compute accuracy of predictions.
    DONE ALREADY
    Params:
      truth.......array of true labels (0 or 1)
      predicted...array of predicted labels (0 or 1)
    """
    return len(np.where(truth==predicted)[0]) / len(truth)


def cross_validation_accuracy(clf, X, labels, k):
    """
    Compute the average testing accuracy over k folds of cross-validation. You
    can use sklearn's KFold class here (no random seed, and no shuffling
    needed).

    Params:
      clf......A LogisticRegression classifier.
      X........A csr_matrix of features.
      labels...The true labels for each instance in X
      k........The number of cross-validation folds.

    Returns:
      The average testing accuracy of the classifier
      over each fold of cross-validation.
    """
    ###TODO
    cv = KFold(len(labels), k)
    accuracies = []
    for train_ind, test_ind in cv:
        clf.fit(X[train_ind], labels[train_ind])
        predictions = clf.predict(X[test_ind])
        accuracies.append(accuracy_score(labels[test_ind], predictions))
    #print('Average 5-fold cross validation accuracy=%.2f (std=%.2f)' %
    ##(np.mean(accuracies), np.std(accuracies))) 
    return np.mean(accuracies)
    pass


def eval_all_combinations(docs, labels, punct_vals,
                          feature_fns, min_freqs):
    """
    Enumerate all possible classifier settings and compute the
    cross validation accuracy for each setting. We will use this
    to determine which setting has the best accuracy.

    For each setting, construct a LogisticRegression classifier
    and compute its cross-validation accuracy for that setting.

    In addition to looping over possible assignments to
    keep_internal_punct and min_freqs, we will enumerate all
    possible combinations of feature functions. So, if
    feature_fns = [token_features, token_pair_features, lexicon_features],
    then we will consider all 7 combinations of features (see Log.txt
    for more examples).

    Params:
      docs..........The list of original training documents.
      labels........The true labels for each training document (0 or 1)
      punct_vals....List of possible assignments to
                    keep_internal_punct (e.g., [True, False])
      feature_fns...List of possible feature functions to use
      min_freqs.....List of possible min_freq values to use
                    (e.g., [2,5,10])

    Returns:
      A list of dicts, one per combination. Each dict has
      four keys:
      'punct': True or False, the setting of keep_internal_punct
      'features': The list of functions used to compute features.
      'min_freq': The setting of the min_freq parameter.
      'accuracy': The average cross_validation accuracy for this setting, using 5 folds.

      This list should be SORTED in descending order of accuracy.

      This function will take a bit longer to run (~20s for me).
    """
    ###TODO
    #print(len(labels))
    #print('400 kidher hai')
    ##for x in feature_fns:
      ##print(x.__name__)
    clf=LogisticRegression()
    fns_combi=list(chain(feature_fns))
    t=list(combinations(feature_fns,2))
    combi=fns_combi+t
    combi.append(feature_fns)
    final=[]
    ##print(combi)
    for x in punct_vals:
      tokens=[]
      for doc in docs:
        tokens.append(tokenize(doc,x))
      #print(x)
      for y in combi:
        ##print('idhar aya')
        for z in min_freqs:
          d=dict()
          #print('andar aya')
          d['punct']=x
          d['features']=y
          d['min_freq']=z
          #print(y)
          Xz,vocab=vectorize(tokens,y,z)
          #print('vectorize kiya')
          d['accuracy']=cross_validation_accuracy(clf,Xz,labels,5)
          final.append(d)
          
          #print(z)
          #print(d['accuracy'])
    return sorted(final,key=lambda k:k['accuracy'],reverse=True)
    pass


def plot_sorted_accuracies(results):
    """
    Plot all accuracies from the result of eval_all_combinations
    in ascending order of accuracy.
    Save to "accuracies.png".
    """
    ###TODO
    i=0
    l=sorted(results,key=lambda k:k['accuracy'])
    
    x=[]
    for d in l:
      x.append(d['accuracy'])

    plt.plot(x)
    #print('show karta')
    #plt.show(block=False)
    plt.savefig("accuracies.png")
    pass


def mean_accuracy_per_setting(results):
    """
    To determine how important each model setting is to overall accuracy,
    we'll compute the mean accuracy of all combinations with a particular
    setting. For example, compute the mean accuracy of all runs with
    min_freq=2.

    Params:
      results...The output of eval_all_combinations
    Returns:
      A list of (accuracy, setting) tuples, SORTED in
      descending order of accuracy.
    """
    ###TODO
    fn=[]
    fq=[]
    p=[]
    l=[]
    c=Counter()
    v=defaultdict(lambda:0)
    cn=defaultdict(lambda:0)
    for d in results:
      x=d['features']
      l='features='
      if isinstance(x,list):
        for y in x:
          l+=str((y.__name__))
          l+=' '
        #print(l)
      elif isinstance(x,tuple):
        a,b=x
        l+=str(a.__name__)
        l+=' ' 
        l+=str(b.__name__)
        #print(l)
      else:
        l+=str(x.__name__)   
        #print(l)
      #if l in fn:
      v[l]+=d['accuracy']
        #c.update(l)
      cn[l]+=1
      '''else:
        fn.append(l)
        v[l]+=d['accuracy']
        #c.update(l)
        cn[l]+=1'''
      ## for punct
      l='punct='+str(d['punct'])
      #print(l)
      #if l in fq:
      v[l]+=d['accuracy']
        #c.update(l)
      cn[l]+=1
      ''' else:
        fq.append(l)
        v[l]+=d['accuracy']
        #c.update(l)
        cn[l]+=1'''
      l='min_freq='+str(d['min_freq'])
      #print(l)
      #if l in p:
      v[l]+=d['accuracy']
        #c.update(l)
      cn[l]+=1
      '''else:
        p.append(l)
        v[l]+=d['accuracy']
        #c.update(l)
        cn[l]+=1'''
    l1=[]
    c1=dict(c)    
    for key,value in v.items():
      if cn[key]>0:
        t=float(value)/float(cn[key])
        t1=(float(t),key)
        l1.append(t1)

    #print(l1)
    return sorted(l1,key=lambda k:k[0],reverse=True)
    pass


def fit_best_classifier(docs, labels, best_result):
    """
    Using the best setting from eval_all_combinations,
    re-vectorize all the training data and fit a
    LogisticRegression classifier to all training data.
    (i.e., no cross-validation done here)

    Params:
      docs..........List of training document strings.
      labels........The true labels for each training document (0 or 1)
      best_result...Element of eval_all_combinations
                    with highest accuracy
    Returns:
      clf.....A LogisticRegression classifier fit to all
            training data.
      vocab...The dict from feature name to column index.
    """
    ###TODO
    d=best_result
    tokens=[]
    for doc in docs:
        tokens.append(tokenize(doc,d['punct']))

    X,vocab=vectorize(tokens,d['features'],d['min_freq'])
    clf=LogisticRegression()
    clf.fit(X,labels)
    return clf,vocab
    pass


def top_coefs(clf, label, n, vocab):
    """
    Find the n features with the highest coefficients in
    this classifier for this label.
    See the .coef_ attribute of LogisticRegression.

    Params:
      clf.....LogisticRegression classifier
      label...1 or 0; if 1, return the top coefficients
              for the positive class; else for negative.
      n.......The number of coefficients to return.
      vocab...Dict from feature name to column index.
    Returns:
      List of (feature_name, coefficient) tuples, SORTED
      in descending order of the coefficient for the
      given class label.
    """
    ###TODO

    l=list(zip(sorted(vocab),clf.coef_[0]))
    #print(l)
    l1=[]
    l2=[]
    #print(l)
    for a,b in l:
      if b>=0:
        l1.append((a,b))
      else:
        l2.append((a,-b))

    if label==0:
      l2=sorted(l2,key=lambda k:k[1],reverse=True)
      return l2[:n]

    else:
      #l1=sorted(l1,lambda key=key[1],reverse=True)
      return sorted(l1,key=lambda k:k[1],reverse=True)[:n]  
    pass


def parse_test_data(best_result, vocab):
    """
    Using the vocabulary fit to the training data, read
    and vectorize the testing data. Note that vocab should
    be passed to the vectorize function to ensure the feature
    mapping is consistent from training to testing.

    Note: use read_data function defined above to read the
    test data.

    Params:
      best_result...Element of eval_all_combinations
                    with highest accuracy
      vocab.........dict from feature name to column index,
                    built from the training data.
    Returns:
      test_docs.....List of strings, one per testing document,
                    containing the raw.
      test_labels...List of ints, one per testing document,
                    1 for positive, 0 for negative.
      X_test........A csr_matrix representing the features
                    in the test data. Each row is a document,
                    each column is a feature.
    """
    ###TODO
    docs, labels = read_data(os.path.join('data', 'test'))
    #tokens=[]
    d=best_result
    tokens=[]
    test_docs=[]
    for doc in docs:
        tokens.append(tokenize(doc,d['punct']))
        test_docs.append(doc)

    X,vocab=vectorize(tokens,d['features'],d['min_freq'],vocab)
    #clf=LogisticRegression()
    #clf.fit(X,labels)
    return test_docs,labels,X
    pass


def print_top_misclassified(test_docs, test_labels, X_test, clf, n):
    """
    Print the n testing documents that are misclassified by the
    largest margin. By using the .predict_proba function of
    LogisticRegression <https://goo.gl/4WXbYA>, we can get the
    predicted probabilities of each class for each instance.
    We will first identify all incorrectly classified documents,
    then sort them in descending order of the predicted probability
    for the incorrect class.
    E.g., if document i is misclassified as positive, we will
    consider the probability of the positive class when sorting.

    Params:
      test_docs.....List of strings, one per test document
      test_labels...Array of true testing labels
      X_test........csr_matrix for test data
      clf...........LogisticRegression classifier fit on all training
                    data.
      n.............The number of documents to print.

    Returns:
      Nothing; see Log.txt for example printed output.
    """
    ###TODO
    #print(clf.predict_proba(X))
    l=clf.predict_proba(X_test)
    predict= clf.predict(X_test)
    #d=dict()
    results=[]
    for i in range(0,len(predict)):
     
      if predict[i]!=test_labels[i]:
        d=dict()
        d['truth']=test_labels[i]
        d['predicted']=predict[i]
        d['proba']=l[i]
        d['doc']=test_docs[i]
        results.append(d)

    l1=sorted(results,key=lambda x:max(x['proba']),reverse=True)[:n]
    #print(len(results))
    for d in l1:
      print("truth=%d predicted=%d proba=%.5f"%(d['truth'],d['predicted'],max(d['proba'])))
      print(d['doc'])

    pass


def main():
    """
    Put it all together.
    ALREADY DONE.
    """
    feature_fns = [token_features, token_pair_features, lexicon_features]
    # Download and read data.
    download_data()
    docs, labels = read_data(os.path.join('data', 'train'))
    # Evaluate accuracy of many combinations
    # of tokenization/featurization.
    results = eval_all_combinations(docs, labels,
                                    [True, False],
                                    feature_fns,
                                    [2,5,10])
    # Print information about these results.
    best_result = results[0]
    worst_result = results[-1]
    print('best cross-validation result:\n%s' % str(best_result))
    print('worst cross-validation result:\n%s' % str(worst_result))
    plot_sorted_accuracies(results)
    print('\nMean Accuracies per Setting:')
    print('\n'.join(['%s: %.5f' % (s,v) for v,s in mean_accuracy_per_setting(results)]))

    # Fit best classifier.
    clf, vocab = fit_best_classifier(docs, labels, results[0])

    # Print top coefficients per class.
    print('\nTOP COEFFICIENTS PER CLASS:')
    print('negative words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 0, 5, vocab)]))
    print('\npositive words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 1, 5, vocab)]))

    # Parse test data
    test_docs, test_labels, X_test = parse_test_data(best_result, vocab)

    # Evaluate on test set.
    predictions = clf.predict(X_test)
    print('testing accuracy=%f' %
          accuracy_score(test_labels, predictions))

    print('\nTOP MISCLASSIFIED TEST DOCUMENTS:')
    print_top_misclassified(test_docs, test_labels, X_test, clf, 5)


if __name__ == '__main__':
    main()
