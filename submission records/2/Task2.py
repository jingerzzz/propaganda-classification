import glob
import os.path
import numpy as np
import sys
import codecs
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
train_folder = "datasets/train-articles" # check that the path to the datasets folder is correct, 
dev_folder = "datasets/dev-articles"     # if not adjust these variables accordingly
train_labels_file = "datasets/train-task2-TC.labels"
dev_template_labels_file = "datasets/dev-task-TC-template.out"
task_TC_output_file = "my-output-TC_v2.txt"


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
    x = x_vec_train
    y = y_train
    y = np.array(y)


    # label_to_name, name_to_label = get_labelization_dict(y)
    # y = labelize(y,name_to_label)

######################### train ######################################
    print("Starting training...")
    clf = LogisticRegression(multi_class="multinomial",solver="newton-cg")
    clf.fit(x,y)
    print("training done.")
######################### predict #####################################
    x_dev=np.asarray(x_vec_dev)
    y_dev_pred = clf.predict(x_dev)

########################## Output ######################################
    with open(task_TC_output_file, "w") as fout:
        for article_id, prediction, span_start, span_end in zip(dev_article_ids, y_dev_pred, dev_span_starts, dev_span_ends):
            fout.write("%s\t%s\t%s\t%s\n" % (article_id, prediction, span_start, span_end))
    print("Predictions written to file " + task_TC_output_file)


load_train_output()