import glob
import os.path
import numpy as np
import sys
import codecs
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

train_folder = "datasets/train-articles" # check that the path to the datasets folder is correct, 
dev_folder = "datasets/dev-articles"     # if not adjust these variables accordingly
train_labels_file = "datasets/train-task2-TC.labels"
dev_template_labels_file = "datasets/dev-task-TC-template.out"
task_TC_output_file = "softmax-output-TC.txt"


def read_articles_from_file_list(folder_name, file_pattern="*.txt"):
    """
    Read articles from files matching patterns <file_pattern> from  
    the directory <folder_name>. 
    The content of the article is saved in the dictionary whose key
    is the id of the article (extracted from the file name).
    Each element of <sentence_list> is one line of the article.
    """
    file_list = glob.glob(os.path.join(folder_name, file_pattern))
    articles = {}
    article_id_list, sentence_id_list, sentence_list = ([], [], [])
    for filename in sorted(file_list):
        article_id = os.path.basename(filename).split(".")[0][7:]
        with codecs.open(filename, "r", encoding="utf8") as f:
            articles[article_id] = f.read()
    return articles

def read_predictions_from_file(filename):
    """
    Reader for the gold file and the template output file. 
    Return values are four arrays with article ids, labels 
    (or ? in the case of a template file), begin of a fragment, 
    end of a fragment. 
    """
    articles_id, span_starts, span_ends, gold_labels = ([], [], [], [])
    with open(filename, "r") as f:
        for row in f.readlines():
            article_id, gold_label, span_start, span_end = row.rstrip().split("\t")
            articles_id.append(article_id)
            gold_labels.append(gold_label)
            span_starts.append(span_start)
            span_ends.append(span_end)
    return articles_id, span_starts, span_ends, gold_labels

def feature_extraction(X_train):
    vectorizer = CountVectorizer(ngram_range=(1, 1),max_features=10000)
    vectorizer.fit_transform(X_train)
    return vectorizer

def text_to_feature(X,vectorizer,if_tfidf=0):
    counts = vectorizer.transform(X).toarray()
    if if_tfidf==1:    
        transformer = TfidfTransformer(smooth_idf=False)
        tfidf = transformer.fit_transform(counts)
        return tfidf.toarray()
    else:
        return counts















def h_w(w,x):## x is dxN, w is dxk, return kxN
    return w.T*x

def softmax(z): ## z is kxN, return kxN probablity matrix
    return np.exp(z) / np.sum(np.exp(z), axis=0)


def get_labelization_dict(y):
    y = set(y)
    y = list(y)
    label_to_name = {}
    name_to_label = {}
    for i in range(len(y)):
        label_to_name[i]=y[i]
        name_to_label[y[i]]=i
    return (label_to_name,name_to_label)
    
def labelize(y,name_to_label):
    y_label = []
    for i in range(len(y)):
        y_label.append(name_to_label[y[i]])
    return y_label

def delabelize(y,label_to_name):
    y_name = []
    for i in range(len(y)):
        y_name.append(label_to_name[y[i]])
    return y_name

    
    
def pred_label(softmax_z):
    y_pred = softmax_z.argmax(axis=0)
    return y_pred



def loglikelihood(softmax_z,y):## soft_max is kxN, y is 1xN (y has been processed so names have been tranformed to labels)
    l = 0
    for j in range(softmax_z.shape[1]):
        l+=np.log(softmax_z[y[0,j],j])
    return l
    
def init_w(x,k):## y is non duplicate list
    d = x.shape[0]
    w = np.zeros((d,k))
    w = np.asmatrix(w)
    w[:,-1]=0
    return w
    
def gradient(x,y,w):## x is dxN, w is dxk, y is 1xN, softmax_z is kxN,return dxk
    ## this function makes sure the last w, w_K, doens't change and always is zero.
    h = h_w(w,x)

    softmax_z = softmax(h)   
    k = w.shape[1]
    N = y.shape[1]
    y_I = np.zeros((k,N))## y_I is kxN
    for j in range(N):
        y_I[y[0,j],j]=1
    
    g = x*(y_I-softmax_z).T
    g[:,-1]=0
    return g
    
    
    
def gradient_ascent(x,y,rate=0.0005):
    k = y.max() + 1
    print("k:{}".format(k))
    w = init_w(x,k)
    h = h_w(w,x)   
    print("h shape:{}".format(h.shape))
    softmax_z = softmax(h)
    print("softmax shape:{}".format(softmax_z.shape))
    l = loglikelihood(softmax_z,y)        
    g = gradient(x,y,w)
    
    stop_delta = 1
    max_iteration = 10000
    iter_count =0
    delta = np.Infinity
    t0=time.time()
    while abs(delta)>stop_delta and iter_count<max_iteration:
#         print("w:{}".format(w))
#         print("h:{}".format(h))
#         print("softmax:{}".format(softmax_z))
        # print("gradient:{}".format(g))
        # print("iter:{}, loglikelihood:{}".format(iter_count,l))
        iter_count+=1
        g = gradient(x,y,w)
        w = w + rate*g
        h = h_w(w,x)
        softmax_z = softmax(h)
        l_new = loglikelihood(softmax_z,y)
        delta = l_new - l
        l = l_new
        if iter_count%10==0:
            print("It's iteration {} now. It takes {} to finish 10 iterations. To finish all, it will take {} hours".format(iter_count,time.time()-t0,(time.time()-t0)*100/3600))
            t0=time.time()
        if iter_count%100==0:
            print("loglikelihood:{}".format(l))

    return w

def evaluate(y_pred, y_test):
    total = len(y_pred)
    correct = 0
    for i in range(total):
        if y_pred[i]==y_test[i]:
            correct+=1
    accuracy = correct/total
    return accuracy

def load_train_output():
    articles = read_articles_from_file_list(train_folder)
    ref_articles_id, ref_span_starts, ref_span_ends, train_gold_labels = read_predictions_from_file(train_labels_file)
    print("Loaded %d annotations from %d articles" % (len(ref_span_starts), len(set(ref_articles_id))))
    y_train = []
    x_train = []
    for i in range(len(ref_articles_id)):
        article = articles[ref_articles_id[i]]
        x_piece = article[int(ref_span_starts[i]):int(ref_span_ends[i])]
        x_train.append(x_piece)
        y_train.append(train_gold_labels[i])
    
    vectorizer=feature_extraction(x_train)
    x_vec_train = text_to_feature(x_train,vectorizer,if_tfidf=0)

    # x_vec_test = text_to_feature(x_test,vectorizer,if_tfidf=0)
    print("feature transform done. In total there are {} features".format(len(x_vec_train[0])))
    
    dev_articles = read_articles_from_file_list(dev_folder)
    dev_article_ids, dev_span_starts, dev_span_ends, dev_labels = read_predictions_from_file(dev_template_labels_file)
    x_dev = []
    for i in range(len(dev_article_ids)):
        article = dev_articles[dev_article_ids[i]]
        x_piece = article[int(dev_span_starts[i]):int(dev_span_ends[i])]
        x_dev.append(x_piece)
    x_vec_dev = text_to_feature(x_dev,vectorizer,if_tfidf=0)
 

    x=np.asmatrix(x_vec_train)
    x=x.T
    y=y_train

    label_to_name, name_to_label = get_labelization_dict(y)
    y = labelize(y,name_to_label)

    y=np.asmatrix(y)
    print("Starting training...")
    w = gradient_ascent(x,y)
    print("training done.")
    x_dev=np.asmatrix(x_vec_dev)
    x_dev=x_dev.T
    # print(y_test)
    h = h_w(w,x_dev)
    # print(h.shape)
    softmax_z = softmax(h)
    print(softmax_z)
    # print(softmax_z.shape)
    y_devPred = pred_label(softmax_z)

    y_devPred= np.asarray(y_devPred)
    y_devPred=y_devPred.tolist()[0]
    y_devPred = delabelize(y_devPred,label_to_name)
    with open(task_TC_output_file, "w") as fout:
        for article_id, prediction, span_start, span_end in zip(dev_article_ids, y_devPred, dev_span_starts, dev_span_ends):
            fout.write("%s\t%s\t%s\t%s\n" % (article_id, prediction, span_start, span_end))
    print("Predictions written to file " + task_TC_output_file)


load_train_output()