import glob
import os
import os.path
import numpy as np
import sys
import codecs
import time
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# nltk.download('wordnet')


train_folder = "datasets/train-articles" # check that the path to the datasets folder is correct, 
dev_folder = "datasets/dev-articles"     # if not adjust these variables accordingly
train_labels_file = "datasets/train-task2-TC.labels"
dev_template_labels_file = "datasets/dev-task-TC-template.out"
task_TC_output_file = "my-output-TC_v3.txt"

stop_words = set(stopwords.words('english'))

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
    vectorizer = CountVectorizer(ngram_range=(1, 3),max_features=2000)
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

def text_preprocess(x):
## remove punctuations
    punctuation_list = ["!",'"',"#","$","%","&","'","(",")","*","+",",","-",".","/",":",";","?","@","[","\\","]","^","_","{","|","}","~","“","”","‘","’","''"]
    wordnet_lemmatizer = WordNetLemmatizer()

    for i in range(len(x)):
        x[i] = word_tokenize(x[i])
        temp_list = [] 
        for j in range(len(x[i])):
            if x[i][j] not in punctuation_list:
                temp_word = x[i][j].lower()## lowercase
                if temp_word not in stop_words:## remove stop words
                    temp_word = wordnet_lemmatizer.lemmatize(temp_word)## lemmatization
                    temp_list.append(temp_word)
        x[i]=" ".join(temp_list)
    # print(x_train)
    return x




def load_train_output():

######################### data loading ###################################
    articles = read_articles_from_file_list(train_folder)
    ref_articles_id, ref_span_starts, ref_span_ends, train_gold_labels = read_predictions_from_file(train_labels_file)
    print("Loaded %d annotations from %d articles" % (len(ref_span_starts), len(set(ref_articles_id))))
    article_list = []
    y_train = []
    x_train = []
    for i in range(len(ref_articles_id)):
        article = articles[ref_articles_id[i]]
        article_list.append(article) ## feature extraction later will use all texts, not only the fragments.
        x_piece = article[int(ref_span_starts[i]):int(ref_span_ends[i])]
        x_train.append(x_piece)
        y_train.append(train_gold_labels[i])
    y_train = np.array(y_train)  

    dev_articles = read_articles_from_file_list(dev_folder)
    dev_article_ids, dev_span_starts, dev_span_ends, dev_labels = read_predictions_from_file(dev_template_labels_file)
    x_dev = []
    for i in range(len(dev_article_ids)):
        article = dev_articles[dev_article_ids[i]]
        x_piece = article[int(dev_span_starts[i]):int(dev_span_ends[i])]
        x_dev.append(x_piece)
######################### preprocessing ##############################
## nothing because the performance actually goes down after preprocessing


######################### feature extraction ############################
    vectorizer=feature_extraction(article_list)
    x_vec_train = text_to_feature(x_train,vectorizer,if_tfidf=0)
    x_vec_dev = text_to_feature(x_dev,vectorizer,if_tfidf=0)
    x_train = x_vec_train
    x_dev = x_vec_dev
    # x_vec_test = text_to_feature(x_test,vectorizer,if_tfidf=0)
    print("feature transform done. In total there are {} features".format(len(x_train[0])))
    


    # label_to_name, name_to_label = get_labelization_dict(y)
    # y = labelize(y,name_to_label)





######################### train ######################################
    print("Starting training...")
    clf = LogisticRegression(multi_class="multinomial",solver="newton-cg")
    clf.fit(x_train,y_train)
    print("training done.")
######################### predict #####################################

    y_dev_pred = clf.predict(x_dev)

########################## Output ######################################
    with open(task_TC_output_file, "w") as fout:
        for article_id, prediction, span_start, span_end in zip(dev_article_ids, y_dev_pred, dev_span_starts, dev_span_ends):
            fout.write("%s\t%s\t%s\t%s\n" % (article_id, prediction, span_start, span_end))
    print("Predictions written to file " + task_TC_output_file)


load_train_output()