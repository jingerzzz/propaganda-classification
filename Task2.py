import glob
import os
import os.path
import numpy as np
import sys
import random
import codecs
import time
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from pylatex import Document, LongTable, MultiColumn


# nltk.download('wordnet')


train_folder = "drive/My Drive/propaganda/datasets/train-articles" # check that the path to the datasets folder is correct, 
dev_folder = "drive/My Drive/propaganda/datasets/dev-articles"     # if not adjust these variables accordingly
train_labels_file = "drive/My Drive/propaganda/datasets/train-task2-TC.labels"
dev_template_labels_file = "drive/My Drive/propaganda/datasets/dev-task-TC-template.out"
task_TC_output_file = "drive/My Drive/propaganda/my-output-TC.txt"

if not os.path.exists("drive/My Drive/propaganda/"):
    train_folder = "datasets/train-articles" 
    dev_folder = "datasets/dev-articles"     
    train_labels_file = "datasets/train-task2-TC.labels"
    dev_template_labels_file = "datasets/dev-task-TC-template.out"
    task_TC_output_file = "my-output-TC.txt"

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
    vectorizer = CountVectorizer(ngram_range=(1, 2))
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
    punctuation_list = ["!",'"',"#","$","%","&","'","(",")","*","+",",","-",".","/",":",";","?","@","[","\\","]","^","_","{","|","}","~"]
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


def add_context(article, start_id,end_id,extra_words_num=3):
    before_count = 0
    after_count = 0
    i = start_id
    j = end_id  
    while(before_count <= extra_words_num and i>=1):
        if article[i]==" ":
            if article[i-1]!=" ":
                after_count+=1
        i-=1
    new_start_id=i+2

    while(after_count <= extra_words_num and j<=(len( article)-2)):
        if article[j]==" ":
            if article[j+1]!=" ":
                before_count+=1
        j+=1
    new_end_id=j-2      

    return new_start_id,new_end_id

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
        # try:
        #     new_start_id,new_end_id = add_context(article,int(ref_span_starts[i]),int(ref_span_ends[i]))
        #     x_piece = article[new_start_id:new_end_id]
        # except:
        #     print("Using original start and end indices.")
        x_piece = article[int(ref_span_starts[i]):int(ref_span_ends[i])]
        x_train.append(x_piece)
        y_train.append(train_gold_labels[i])
    # print(x_train)
    y_train = np.array(y_train)  
    dev_articles = read_articles_from_file_list(dev_folder)
    dev_article_ids, dev_span_starts, dev_span_ends, dev_labels = read_predictions_from_file(dev_template_labels_file)
    x_dev = []
    for i in range(len(dev_article_ids)):
        article = dev_articles[dev_article_ids[i]]
        x_piece = article[int(dev_span_starts[i]):int(dev_span_ends[i])]
        x_dev.append(x_piece)

    # print("training size:{}".format(len(x_train)))

    ######################### preprocessing ##############################
    # x_train = text_preprocess(x_train)
    # x_vec = text_preprocess(x_dev)



    ######################### feature extraction ############################
    vectorizer=feature_extraction(x_train) ## Use only located fragments
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
    t0 = time.time()
    clf = LogisticRegression(multi_class="multinomial",solver="newton-cg")
    clf.fit(x_train,y_train)
    t1 = time.time()
    print("It takes {} seconds ({} minutes) to finish training".format(t1-t0,(t1-t0)/60))
    print("training done.")
    ######################### predict #####################################

    y_train_pred = clf.predict(x_train)
    f1 = f1_score(y_train,y_train_pred,average=None)
    f1_micro = f1_score(y_train,y_train_pred,average='micro')
    print("training score:\n micro average:{}\nfor each class:{}".format(f1_micro,f1))
    y_dev_pred = clf.predict(x_dev) 

    ########################## Output ######################################
    with open(task_TC_output_file, "w") as fout:
        for article_id, prediction, span_start, span_end in zip(dev_article_ids, y_dev_pred, dev_span_starts, dev_span_ends):
            fout.write("%s\t%s\t%s\t%s\n" % (article_id, prediction, span_start, span_end))
    print("Predictions written to file " + task_TC_output_file)


def load_train_output_crossvalidation(clf):

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
        # try:
        #     new_start_id,new_end_id = add_context(article,int(ref_span_starts[i]),int(ref_span_ends[i]))
        #     x_piece = article[new_start_id:new_end_id]
        # except:
        #     print("Using original start and end indices.")
        x_piece = article[int(ref_span_starts[i]):int(ref_span_ends[i])]
        x_train.append(x_piece)
        y_train.append(train_gold_labels[i])
    # print(x_train)
    y_train = np.array(y_train)  
    dev_articles = read_articles_from_file_list(dev_folder)
    dev_article_ids, dev_span_starts, dev_span_ends, dev_labels = read_predictions_from_file(dev_template_labels_file)
    x_dev = []
    for i in range(len(dev_article_ids)):
        article = dev_articles[dev_article_ids[i]]
        x_piece = article[int(dev_span_starts[i]):int(dev_span_ends[i])]
        x_dev.append(x_piece)

    # print("training size:{}".format(len(x_train)))

    ######################### preprocessing ##############################
    # x_train = text_preprocess(x_train)
    # x_vec = text_preprocess(x_dev)

    ## Using 10-fold crossvalidation
    total_size = len(x_train)
    print("total training data size:{}".format(total_size))
    k = 10
    dev_size = int(total_size/k)-1
    print("dev size:{}".format(dev_size))

    ## shuffle randomly the total data set
    xy_total = list(zip(x_train,y_train))
    random.shuffle(xy_total)

    f1_train_micro_list = []
    f1_train_list = []
    f1_dev_micro_list = []
    f1_dev_list = []
    ## for each fold
    for i in range(k):
        print("Crossvalidation fold {}----------------------------------------------------".format(i+1))
        start_id = i*dev_size
        end_id = (i+1)*dev_size
        xy_dev = xy_total[start_id:end_id]
        xy_train = [item for item in xy_total if item not in xy_dev]
        x_dev, y_dev = zip(*xy_dev)
        x_dev = list(x_dev)
        y_dev = list(y_dev)
        x_train, y_train = zip(*xy_train)
        x_train = list(x_train)
        y_train = list(y_train)  
    ######################### feature extraction ############################
        vectorizer=feature_extraction(x_train) ## Use only located fragments
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
        t0 = time.time()
        # clf = LogisticRegression(multi_class="multinomial",solver="newton-cg")
        clf.fit(x_train,y_train)
        t1 = time.time()
        print("It takes {} seconds ({} minutes) to finish training".format(t1-t0,(t1-t0)/60))
        print("training done.")
    ######################### predict #####################################

        y_train_pred = clf.predict(x_train)
        f1_train = f1_score(y_train,y_train_pred,average=None)
        f1_train_micro = f1_score(y_train,y_train_pred,average='micro')
        f1_train_list.append(f1_train)
        f1_train_micro_list.append(f1_train_micro)
        y_dev_pred = clf.predict(x_dev) 
        f1_dev = f1_score(y_dev,y_dev_pred,average=None)
        f1_dev_micro = f1_score(y_dev,y_dev_pred,average='micro')
        f1_dev_list.append(f1_dev)
        f1_dev_micro_list.append(f1_dev_micro)

    print("Cross validation is done. Here are the scores:")
    f1_train_average = [0]*len(f1_train_list[0])
    for i in range(len(f1_train_list)):
        for j in range(len(f1_train_list[i])):
            f1_train_average[j] += f1_train_list[i][j]/k

    f1_dev_average = [0]*len(f1_dev_list[0])
    for i in range(len(f1_dev_list)):
        for j in range(len(f1_dev_list[i])):
            f1_dev_average[j] += f1_dev_list[i][j]/k

    f1_train_micro_average = 0
    for i in range(len(f1_train_micro_list)):
        f1_train_micro_average += f1_train_micro_list[i]/k


    f1_dev_micro_average = 0
    for i in range(len(f1_dev_micro_list)):
        f1_dev_micro_average += f1_dev_micro_list[i]/k

    f1_scores_dict = {"f1_train_average":f1_train_average,"f1_dev_average":f1_dev_average,"f1_train_micro_average":f1_train_micro_average,"f1_dev_micro_average":f1_dev_micro_average}
    return f1_scores_dict
    # print("training score:\n micro average:{}\nfor each class:{}".format(f1_train_micro_average,f1_train_average))
    # print("validation score:\n micro average:{}\nfor each class:{}".format(f1_dev_micro_average,f1_dev_average))

def round_f1_score(f1_scores_dict):
    f1_train_average = f1_scores_dict["f1_train_average"]
    f1_dev_average = f1_scores_dict["f1_dev_average"]
    f1_train_micro_average = f1_scores_dict["f1_train_micro_average"]
    f1_dev_micro_average = f1_scores_dict["f1_dev_micro_average"]

    f1_train_micro_average = round(f1_train_micro_average,4)    
    f1_dev_micro_average = round(f1_dev_micro_average,4)    
    for i in range(len(f1_train_average)):
        f1_train_average[i] = round(f1_train_average[i],4)    
    for i in range(len(f1_dev_average)):
        f1_dev_average[i] = round(f1_dev_average[i],4)    
    f1_scores_dict = {"f1_train_average":f1_train_average,"f1_dev_average":f1_dev_average,"f1_train_micro_average":f1_train_micro_average,"f1_dev_micro_average":f1_dev_micro_average}
    return f1_scores_dict       

def basic_generate_row(model_name,f1_scores_dict):
    row = []
    row.append(model_name)
    row.append(f1_scores_dict["f1_train_micro_average"])
    row.append(f1_scores_dict["f1_dev_micro_average"])
    row.append(f1_scores_dict["f1_train_average"])
    row.append(f1_scores_dict["f1_dev_average"])
    return row

def basic_run():
    geometry_options = {
        "margin": "2.54cm",
        "includeheadfoot": True
    }
    doc = Document(page_numbers=True, geometry_options=geometry_options)
    clf_1 = LinearSVC(dual=False)
    clf_2 = LogisticRegression(multi_class="multinomial",solver="newton-cg")
    clf_3 = MultinomialNB()
    clf_4 = RandomForestClassifier()
    LinearSVC_f1_scores_dict = load_train_output_crossvalidation(clf_1)
    LogisticRegression_f1_scores_dict = load_train_output_crossvalidation(clf_2)
    MultinomialNB_f1_scores_dict = load_train_output_crossvalidation(clf_3)
    RandomForest_f1_scores_dict = load_train_output_crossvalidation(clf_4)
    row_1 = basic_generate_row("LinearSVC",LinearSVC_f1_scores_dict)
    row_2 = basic_generate_row("LogisticRegression",LogisticRegression_f1_scores_dict)
    row_3 = basic_generate_row("MultinomialNB",MultinomialNB_f1_scores_dict)
    row_4 = basic_generate_row("RandomForest",RandomForest_f1_scores_dict)
    rows = []
    rows.append(row_1)
    rows.append(row_2)
    rows.append(row_3)
    rows.append(row_4)

    # Generate data table
    with doc.create(LongTable("l l l l l")) as data_table:
            data_table.add_hline()
            data_table.add_row(["Model Name",  "training f1", "valid f1","training f1 for each technique","valid f1 for each technique"])
            data_table.add_hline()
            data_table.end_table_header()
            data_table.add_hline()
            for i in range(len(rows)):
                data_table.add_row(rows[i])

    doc.generate_pdf("basic_run", clean_tex=False)

def tuning_LinearSVC():

    param_grid = {"tol": [1e-3,1e-4,1e-5],
              "C":  [1e-5,3e-5,1e-4,3e-4,1e-3],
              "max_iter":[100,200,300]}
    result_list = []
    optimized_param=[0,0,0,0,0,0,0]
    for tol in param_grid["tol"]:
        for C in param_grid["C"]:
            for max_iter in param_grid["max_iter"]:
                    # try:
                        current_param_and_eval = [tol,C,max_iter]
                        clf = LinearSVC(tol=tol,C=C,max_iter=max_iter,dual=False)
                        f1_scores_dict=load_train_output_crossvalidation(clf)
                        f1_scores_dict=round_f1_score(f1_scores_dict)
                        f1_train_average = f1_scores_dict["f1_train_average"]
                        f1_dev_average = f1_scores_dict["f1_dev_average"]
                        f1_train_micro_average = f1_scores_dict["f1_train_micro_average"]
                        f1_dev_micro_average = f1_scores_dict["f1_dev_micro_average"]
                        current_param_and_eval.append(f1_train_micro_average)
                        current_param_and_eval.append(f1_dev_micro_average)
                        current_param_and_eval.append(f1_train_average)
                        current_param_and_eval.append(f1_dev_average)

                        result_list.append(current_param_and_eval)
                        if current_param_and_eval[4]>optimized_param[4]:
                            optimized_param=current_param_and_eval
                    # except:
                    #     print("An exception occurs.")

    # Generate data table
    geometry_options = {
        "margin": "2.54cm",
        "includeheadfoot": True
    }
    doc = Document(page_numbers=True, geometry_options=geometry_options)   
    with doc.create(LongTable("l l l l l l l")) as data_table:
            data_table.add_hline()
            data_table.add_row(["tol", "C", "max_iter", "training f1", "valid f1","training f1 for each technique","valid f1 for each technique"])
            data_table.add_hline()
            data_table.end_table_header()
            data_table.add_hline()
            for i in range(len(result_list)):
                data_table.add_row(result_list[i][0:7])
            data_table.add_hline()
    with doc.create(LongTable("l l l l l l l")) as data_table:
            data_table.add_hline()
            data_table.add_row(["tol", "C","max_iter", "training f1", "valid f1","training f1 for each technique","valid f1 for each technique"])
            data_table.add_hline()
            data_table.end_table_header()
            data_table.add_hline()
            data_table.add_row(optimized_param)
            data_table.add_hline()
    print("This is for LinearSVC.")
    doc.generate_pdf("tuning_LinearSVC", clean_tex=False)

def tuning_LogisticRegression():
    param_grid = {"tol": [1e-3,1e-4,1e-5],
              "C":  [1e-5,3e-5,1e-4,3e-4,1e-3],
              "max_iter":[100,200,300]}
    result_list = []
    optimized_param=[0,0,0,0,0,0,0]
    for tol in param_grid["tol"]:
        for C in param_grid["C"]:
            for max_iter in param_grid["max_iter"]:
                    # try:
                        current_param_and_eval = [tol,C,max_iter]
                        clf = LogisticRegression(tol=tol,C=C,max_iter=max_iter,multi_class="multinomial",solver="newton-cg")
                        f1_scores_dict=load_train_output_crossvalidation(clf)
                        f1_scores_dict=round_f1_score(f1_scores_dict)
                        f1_train_average = f1_scores_dict["f1_train_average"]
                        f1_dev_average = f1_scores_dict["f1_dev_average"]
                        f1_train_micro_average = f1_scores_dict["f1_train_micro_average"]
                        f1_dev_micro_average = f1_scores_dict["f1_dev_micro_average"]
                        current_param_and_eval.append(f1_train_micro_average)
                        current_param_and_eval.append(f1_dev_micro_average)
                        current_param_and_eval.append(f1_train_average)
                        current_param_and_eval.append(f1_dev_average)

                        result_list.append(current_param_and_eval)
                        if current_param_and_eval[4]>optimized_param[4]:
                            optimized_param=current_param_and_eval
                    # except:
                    #     print("An exception occurs.")

    # Generate data table
    geometry_options = {
        "margin": "2.54cm",
        "includeheadfoot": True
    }
    doc = Document(page_numbers=True, geometry_options=geometry_options)   
    with doc.create(LongTable("l l l l l l l")) as data_table:
            data_table.add_hline()
            data_table.add_row(["tol", "C", "max_iter", "training f1", "valid f1","training f1 for each technique","valid f1 for each technique"])
            data_table.add_hline()
            data_table.end_table_header()
            data_table.add_hline()
            for i in range(len(result_list)):
                data_table.add_row(result_list[i][0:7])
            data_table.add_hline()
    with doc.create(LongTable("l l l l l l l")) as data_table:
            data_table.add_hline()
            data_table.add_row(["tol", "C","max_iter", "training f1", "valid f1","training f1 for each technique","valid f1 for each technique"])
            data_table.add_hline()
            data_table.end_table_header()
            data_table.add_hline()
            data_table.add_row(optimized_param)
            data_table.add_hline()
    print("This is for LogisticRegression.")
    doc.generate_pdf("tuning_LogisticRegression", clean_tex=False)

def tuning_MultinomialNB():
    param_grid = {"alpha": [1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9]}
    result_list = []
    optimized_param=[0,0,0,0,0]
    for alpha  in param_grid["alpha"]:
                    # try:
                        current_param_and_eval = [alpha]
                        clf = MultinomialNB(alpha=alpha)
                        f1_scores_dict=load_train_output_crossvalidation(clf)
                        f1_scores_dict=round_f1_score(f1_scores_dict)
                        f1_train_average = f1_scores_dict["f1_train_average"]
                        f1_dev_average = f1_scores_dict["f1_dev_average"]
                        f1_train_micro_average = f1_scores_dict["f1_train_micro_average"]
                        f1_dev_micro_average = f1_scores_dict["f1_dev_micro_average"]
                        current_param_and_eval.append(f1_train_micro_average)
                        current_param_and_eval.append(f1_dev_micro_average)
                        current_param_and_eval.append(f1_train_average)
                        current_param_and_eval.append(f1_dev_average)

                        result_list.append(current_param_and_eval)
                        if current_param_and_eval[2]>optimized_param[2]:
                            optimized_param=current_param_and_eval
                    # except:
                    #     print("An exception occurs.")

    # Generate data table
    geometry_options = {
        "margin": "2.54cm",
        "includeheadfoot": True
    }
    doc = Document(page_numbers=True, geometry_options=geometry_options)   
    with doc.create(LongTable("l l l l l")) as data_table:
            data_table.add_hline()
            data_table.add_row(["alpha", "training f1", "valid f1","training f1 for each technique","valid f1 for each technique"])
            data_table.add_hline()
            data_table.end_table_header()
            data_table.add_hline()
            for i in range(len(result_list)):
                data_table.add_row(result_list[i][0:5])
            data_table.add_hline()
    with doc.create(LongTable("l l l l l")) as data_table:
            data_table.add_hline()
            data_table.add_row(["alpha", "training f1", "valid f1","training f1 for each technique","valid f1 for each technique"])
            data_table.add_hline()
            data_table.end_table_header()
            data_table.add_hline()
            data_table.add_row(optimized_param)
            data_table.add_hline()
    print("This is for MultinomialNB.")
    doc.generate_pdf("tuning_MultinomialNB", clean_tex=False)

def tuning_RandomForest():
    param_grid = {"max_features":["auto","sqrt","log2"],
              "n_estimators": [50,100,200],
              "min_sample_leaf":[25,50,100],
              "max_depth": [None,10,20,40,80],
              }
    result_list = []
    optimized_param=[0,0,0,0,0,0,0,0]
    for max_features in param_grid["max_features"]:
        for n_estimators in param_grid["n_estimators"]:
            for min_sample_leaf in param_grid["min_sample_leaf"]:
                    for max_depth in param_grid["max_depth"]:
                    # try:
                        current_param_and_eval = [max_features,n_estimators,min_sample_leaf,max_depth]
                        clf = RandomForestClassifier(max_features=max_features,n_estimators=n_estimators,min_samples_leaf=min_sample_leaf,max_depth=max_depth)
                        f1_scores_dict=load_train_output_crossvalidation(clf)
                        f1_scores_dict=round_f1_score(f1_scores_dict)
                        f1_train_average = f1_scores_dict["f1_train_average"]
                        f1_dev_average = f1_scores_dict["f1_dev_average"]
                        f1_train_micro_average = f1_scores_dict["f1_train_micro_average"]
                        f1_dev_micro_average = f1_scores_dict["f1_dev_micro_average"]
                        current_param_and_eval.append(f1_train_micro_average)
                        current_param_and_eval.append(f1_dev_micro_average)
                        current_param_and_eval.append(f1_train_average)
                        current_param_and_eval.append(f1_dev_average)

                        result_list.append(current_param_and_eval)
                        if current_param_and_eval[5]>optimized_param[5]:
                            optimized_param=current_param_and_eval
                    # except:
                    #     print("An exception occurs.")

    # Generate data table
    geometry_options = {
        "margin": "2.54cm",
        "includeheadfoot": True
    }
    doc = Document(page_numbers=True, geometry_options=geometry_options)   
    with doc.create(LongTable("l l l l l l l l")) as data_table:
            data_table.add_hline()
            data_table.add_row(["max_features","n_estimators","min_sample_leaf","max_depth", "training f1", "valid f1","training f1 for each technique","valid f1 for each technique"])
            data_table.add_hline()
            data_table.end_table_header()
            data_table.add_hline()
            for i in range(len(result_list)):
                data_table.add_row(result_list[i][0:8])
            data_table.add_hline()
    with doc.create(LongTable("l l l l l l l l")) as data_table:
            data_table.add_hline()
            data_table.add_row(["max_features","n_estimators","min_sample_leaf","max_depth", "training f1", "valid f1","training f1 for each technique","valid f1 for each technique"])
            data_table.add_hline()
            data_table.end_table_header()
            data_table.add_hline()
            data_table.add_row(optimized_param)
            data_table.add_hline()
    print("This is for RandomForest.")
    doc.generate_pdf("tuning_RandomForest", clean_tex=False)
## model training and evaluatiion #########################
# tuning_LinearSVC()
# tuning_LogisticRegression()
# tuning_MultinomialNB()
tuning_RandomForest()
# basic_run()