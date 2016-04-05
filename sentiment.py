import sys
import collections
import sklearn.naive_bayes
import sklearn.linear_model
import nltk
import random
random.seed(0)
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
#nltk.download("stopwords")          # Download the stop words from nltk


# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
if len(sys.argv) != 3:
    print "python sentiment.py <path_to_data> <0|1>"
    print "0 = NLP, 1 = Doc2Vec"
    exit(1)
path_to_data = sys.argv[1]
method = int(sys.argv[2])



def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)
    if method == 0:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
    if method == 1:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)
    print "Naive Bayes"
    print "-----------"
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
    print ""
    print "Logistic Regression"
    print "-------------------"
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)



def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for i,line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_pos.append(words)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_neg.append(words)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_pos.append(words)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_neg.append(words)

    return train_pos, train_neg, test_pos, test_neg



def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # English stopwords from nltk
    stopwords = set(nltk.corpus.stopwords.words('english'))
    
    # Determine a list of words that will be used as features. 
    # This list should have the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.
    # YOUR CODE HERE
    x=int(0.01*len(train_pos))
    y=int(0.01*len(train_neg))

    merged_set=set([])
    
    for i in train_pos:
        merged_set.update(set(i))

    for i in train_neg:
        merged_set.update(set(i))
    
   
    merged_set_stop=merged_set-stopwords
    merged_set_filtered=set([])
    train_pos_modify=[]
    train_neg_modify=[]

    
    
    for i in train_pos:
    	
        for j in list(set(i)):
    	    train_pos_modify.append(j)

    for i in train_neg:
    	
    	for j in list(set(i)):
    	    train_neg_modify.append(j)

    c=collections.Counter(train_pos_modify)
    d=collections.Counter(train_neg_modify)


    
    for j in merged_set_stop:
    	
        if (c[j]>x or d[j]>y) and (c[j]>=2*d[j] or d[j]>=2*c[j]):
    	    merged_set_filtered.add(j)

   
    
    
 
    
    train_pos_vec = []
    for i in train_pos:
        l=[]
        for j in merged_set_filtered:
            if j in i:
                l.append(1)
            else:
                l.append(0)
        train_pos_vec.append(l)


    train_neg_vec = []
    for i in train_neg:
        l=[]
        for j in merged_set_filtered:
            if j in i:
                l.append(1)
            else:
                l.append(0)
        train_neg_vec.append(l)

    test_pos_vec = []
    for i in test_pos:
        l=[]
        for j in merged_set_filtered:
            if j in i:
                l.append(1)
            else:
                l.append(0)
        test_pos_vec.append(l)
    
    
    test_neg_vec = []
    for i in test_neg:
        l=[]
        for j in merged_set_filtered:
            if j in i:
                l.append(1)
            else:
                l.append(0)
        test_neg_vec.append(l)

    
        
        
        
   

 
        

     


    
  	

    
    


    # Using the above words as features, construct binary vectors for each text in the training and test set.
    # These should be python lists containing 0 and 1 integers.
    # YOUR CODE HERE

    # Return the four feature vectors
    
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Doc2Vec requires LabeledSentence objects as input.
    # Turn the datasets from lists of words to lists of LabeledSentence objects.
    # YOUR CODE HERE
    labeled_train_pos = []
    for k, i in enumerate(train_pos):
        labeled_train_pos.append(LabeledSentence(words = i,tags = ['TRAIN_POS_' + str(k)]))

    labeled_train_neg = []
    for k, i in enumerate(train_neg):
        labeled_train_neg.append(LabeledSentence(words = i,tags = ['TRAIN_NEG_' + str(k)]))
        
    labeled_test_pos = []
    for k, i in enumerate(test_pos):
        labeled_test_pos.append(LabeledSentence(words = i,tags = ['TEST_POS_' + str(k)]))
        
    labeled_test_neg = []
    for k, i in enumerate(test_neg):
        labeled_test_neg.append(LabeledSentence(words = i,tags = ['TEST_NEG_' + str(k)]))

   
    
    
                                 

    # Initialize model
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
    model.build_vocab(sentences)

    # Train the model
    # This may take a bit to run 
    for i in range(5):
        print "Training iteration %d" % (i)
        random.shuffle(sentences)
        model.train(sentences)

    # Use the docvecs function to extract the feature vectors for the training and test data
    # YOUR CODE HERE
    
    train_pos_vec = []
    for k, i in enumerate(labeled_train_pos):
        train_pos_vec.append(model.docvecs['TRAIN_POS_' + str(k)])

    train_neg_vec = []
    for k, i in enumerate(labeled_train_neg):
        train_neg_vec.append(model.docvecs['TRAIN_NEG_' + str(k)])
        
    test_pos_vec = []
    for k, i in enumerate(labeled_test_pos):
        test_pos_vec.append(model.docvecs['TEST_POS_' + str(k)])
    
    test_neg_vec = []
    for k, i in enumerate(labeled_test_neg):
        test_neg_vec.append(model.docvecs['TEST_NEG_' + str(k)])
    
    # Return the four feature vectors
    
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
    # For BernoulliNB, use alpha=1.0 and binarize=None
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    X=train_pos_vec+train_neg_vec
    nb_model=sklearn.naive_bayes.BernoulliNB(alpha=1.0,binarize=None)
    nb_model.fit(X,Y)

    lr_model=sklearn.linear_model.LogisticRegression()
    lr_model.fit(X,Y)    
    return nb_model, lr_model



def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    X=train_pos_vec+train_neg_vec
    nb_model=sklearn.naive_bayes.GaussianNB()
    nb_model.fit(X,Y)

    lr_model=sklearn.linear_model.LogisticRegression()
    lr_model.fit(X,Y) 
    
    return nb_model, lr_model



def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.
    # YOUR CODE HERE
    X=test_pos_vec+test_neg_vec
    Y=model.predict(X)
    tp=0.0
    tn=0.0
    fp=0.0
    fn=0.0
    Z=["pos"]*len(test_pos_vec) + ["neg"]*len(test_neg_vec)
    for i in range(0,len(Z)):

        if Z[i]==Y[i] and Z[i]=="pos":
            tp=tp+1
        if Z[i]==Y[i] and Z[i]=="neg":
            tn=tn+1
        if Z[i]!=Y[i] and Y[i]=="pos":
            fn=fn+1
        if Z[i]!=Y[i] and Y[i]=="neg":
            fp=fp+1

    accuracy=float((tp+tn)/(tp+tn+fp+fn))
        
            
        
    
    
    if print_confusion:
        print "predicted:\tpos\tneg"
        print "actual:"
        print "pos\t\t%d\t%d" % (tp, fn)
        print "neg\t\t%d\t%d" % (fp, tn)
    print "accuracy: %f" % (accuracy)



if __name__ == "__main__":
    main()
