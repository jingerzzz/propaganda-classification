import torch

# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

import pandas as pd
import glob
import os
import os.path
import numpy as np
import sys
import random
import codecs
import time
import nltk
from sklearn.metrics import f1_score

# !pip install transformers

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

def get_label_dict(labels):
    label_to_idx_dict = {}
    idx_to_label_dict = {}
    idx = 0
    for label in labels:
        if label not in label_to_idx_dict.keys():
            label_to_idx_dict[label] = idx
            idx += 1

    for key,value in label_to_idx_dict.items():
        idx_to_label_dict[value] = key

    return label_to_idx_dict,idx_to_label_dict

def label_to_idx(labels,label_to_idx_dict):
    indices = []
    for label in labels:
        indices.append(label_to_idx_dict[label])
    return indices

def idx_to_label(indices, idx_to_label_dict):
    labels = []
    for index in indices:
        labels.append(idx_to_label_dict[index])
    return labels

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def flat_f1score(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    f1_micro = f1_score(labels_flat,pred_flat,average="micro")
    f1_each = f1_score(labels_flat,pred_flat,average=None)
    return f1_micro, f1_each




import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

    
def myBERT_crossvalidation():

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
    # print(x_train)
    y_train = np.array(y_train)  
    dev_articles = read_articles_from_file_list(dev_folder)
    dev_article_ids, dev_span_starts, dev_span_ends, dev_labels = read_predictions_from_file(dev_template_labels_file)
    x_dev = []
    for i in range(len(dev_article_ids)):
        article = dev_articles[dev_article_ids[i]]
        x_piece = article[int(dev_span_starts[i]):int(dev_span_ends[i])]
        x_dev.append(x_piece)

    label_to_idx_dict, idx_to_label_dict = get_label_dict(y_train)
    y_train = label_to_idx(y_train,label_to_idx_dict)
    # print("training size:{}".format(len(x_train)))

    ######################### preprocessing ##############################
    # x_train = text_preprocess(x_train)
    # x_vec = text_preprocess(x_dev)








    ######################### train ######################################
    from transformers import BertTokenizer

    # Load the BERT tokenizer.
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []

    # For every sentence...
    for sent in x_train:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_sent = tokenizer.encode(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'

                            # This function also supports truncation and conversion
                            # to pytorch tensors, but we need to do padding, so we
                            # can't use these features :( .
                            #max_length = 128,          # Truncate all sentences.
                            #return_tensors = 'pt',     # Return pytorch tensors.
                    )
        
        # Add the encoded sentence to the list.
        input_ids.append(encoded_sent)

    # Print sentence 0, now as a list of IDs.
    print('Original: ', x_train[0])
    print('Token IDs:', input_ids[0])
    print('Max sentence length: ', max([len(sen) for sen in input_ids]))

    # We'll borrow the `pad_sequences` utility function to do this.
    from keras.preprocessing.sequence import pad_sequences

    # Set the maximum sequence length.
    # I've chosen 64 somewhat arbitrarily. It's slightly larger than the
    # maximum training sentence length of 47...
    MAX_LEN = 64

    print('\nPadding/truncating all sentences to %d values...' % MAX_LEN)

    print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))

    # Pad our input tokens with value 0.
    # "post" indicates that we want to pad and truncate at the end of the sequence,
    # as opposed to the beginning.
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", 
                            value=0, truncating="post", padding="post")

    print('\nDone.')

    # Create attention masks
    attention_masks = []

    # For each sentence...
    for sent in input_ids:
        
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]
        
        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)

    # training
    

    from sklearn.model_selection import train_test_split

    # Use 90% for training and 10% for validation.
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, y_train, 
                                                                random_state=2018, test_size=0.1)
    # Do the same for the masks.
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, y_train,
                                                random_state=2018, test_size=0.1)

    ## Using 10-fold crossvalidation
    total_size = len(y_train)
    print("total training data size:{}".format(total_size))
    k = 10
    dev_size = int(total_size/k)-1
    print("dev size:{}".format(dev_size))

    ## shuffle randomly the total data set
    xy_total = list(zip(input_ids,attention_masks,y_train))
    random.shuffle(xy_total)

    f1_train_micro_list = []
    f1_train_list = []
    f1_validation_micro_list = []
    f1_validation_list = []
    ## for each fold
    for i in range(k):
        print("Crossvalidation fold {}----------------------------------------------------".format(i+1))

        f1_final_validation_micro = 0

        start_id = i*dev_size
        end_id = (i+1)*dev_size
        xy_dev = xy_total[start_id:end_id]
        xy_train = xy_total[0:start_id]+xy_total[end_id:]
        
        validation_inputs,validation_masks,validation_labels = zip(*xy_dev)
        
        validation_inputs = list(validation_inputs)
        validatioin_inputs = np.array(validation_inputs)

        validation_masks = list(validation_masks)
        validatioin_masks = np.array(validation_masks)

        validation_labels = list(validation_labels)
        validatioin_labels = np.array(validation_labels)

        train_inputs,train_masks,train_labels = zip(*xy_train)
        
        train_inputs = list(train_inputs)
        train_inputs = np.array(train_inputs)

        train_masks = list(train_masks)
        train_masks = np.array(train_masks)

        train_labels = list(train_labels)
        train_labels = np.array(train_labels)
        # Convert all inputs and labels into torch tensors, the required datatype 
        # for our model.

        train_inputs = torch.tensor(train_inputs)
        validation_inputs = torch.tensor(validation_inputs)

        train_labels = torch.tensor(train_labels)
        validation_labels = torch.tensor(validation_labels)

        train_masks = torch.tensor(train_masks)
        validation_masks = torch.tensor(validation_masks)

        from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

        # The DataLoader needs to know our batch size for training, so we specify it 
        # here.
        # For fine-tuning BERT on a specific task, the authors recommend a batch size of
        # 16 or 32.

        batch_size = 32

        # Create the DataLoader for our training set.
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        # Create the DataLoader for our validation set.
        validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

        from transformers import BertForSequenceClassification, AdamW, BertConfig

        # Load BertForSequenceClassification, the pretrained BERT model with a single 
        # linear classification layer on top. 
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = 14, # The number of output labels--2 for binary classification.
                            # You can increase this for multi-class tasks.   
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
        )

        # Tell pytorch to run this model on the GPU.
        model.cuda()

        # Get all of the model's parameters as a list of tuples.
        params = list(model.named_parameters())

        print('The BERT model has {:} different named parameters.\n'.format(len(params)))

        print('==== Embedding Layer ====\n')

        for p in params[0:5]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

        print('\n==== First Transformer ====\n')

        for p in params[5:21]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

        print('\n==== Output Layer ====\n')

        for p in params[-4:]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

        # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
        # I believe the 'W' stands for 'Weight Decay fix"
        optimizer = AdamW(model.parameters(),
                        lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                        eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                        )


        from transformers import get_linear_schedule_with_warmup

        # Number of training epochs (authors recommend between 2 and 4)
        epochs = 4

        # Total number of training steps is number of batches * number of epochs.
        total_steps = len(train_dataloader) * epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps = 0, # Default value in run_glue.py
                                                    num_training_steps = total_steps)



        # Set the seed value all over the place to make this reproducible.
        seed_val = 42

        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        # Store the average loss after each epoch so we can plot them.
        loss_values = []

        # For each epoch...
        for epoch_i in range(0, epochs):
            
            # ========================================
            #               Training
            # ========================================
            
            # Perform one full pass over the training set.

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Reset the total loss for this epoch.
            total_loss = 0

            # Put the model into training mode. Don't be mislead--the call to 
            # `train` just changes the *mode*, it doesn't *perform* the training.
            # `dropout` and `batchnorm` layers behave differently during training
            # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
            model.train()

            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):

                # Progress update every 40 batches.
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = format_time(time.time() - t0)
                    
                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

                # Unpack this training batch from our dataloader. 
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using the 
                # `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids 
                #   [1]: attention masks
                #   [2]: labels 
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                ###############Bug fix code####################
                b_input_ids = b_input_ids.type(torch.LongTensor)
                b_input_mask = b_input_mask.type(torch.LongTensor)
                b_labels = b_labels.type(torch.LongTensor)

                b_input_ids = b_input_ids.to(device)
                b_input_mask = b_input_mask.to(device)
                b_labels = b_labels.to(device)
                ############################################
                # Always clear any previously calculated gradients before performing a
                # backward pass. PyTorch doesn't do this automatically because 
                # accumulating the gradients is "convenient while training RNNs". 
                # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
                model.zero_grad()        

                # Perform a forward pass (evaluate the model on this training batch).
                # This will return the loss (rather than the model output) because we
                # have provided the `labels`.
                # The documentation for this `model` function is here: 
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask, 
                            labels=b_labels)
                
                # The call to `model` always returns a tuple, so we need to pull the 
                # loss value out of the tuple.
                loss = outputs[0]

                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end. `loss` is a Tensor containing a
                # single value; the `.item()` function just returns the Python value 
                # from the tensor.
                total_loss += loss.item()

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient.
                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.
                optimizer.step()

                # Update the learning rate.
                scheduler.step()

            # Calculate the average loss over the training data.
            avg_train_loss = total_loss / len(train_dataloader)            
            
            # Store the loss value for plotting the learning curve.
            loss_values.append(avg_train_loss)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
                
            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            print("")
            print("Running Validation...")

            t0 = time.time()

            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            model.eval()

            # Tracking variables 
            f1_validation_micro = 0 
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            # Evaluate data for one epoch
            for batch in validation_dataloader:
                
                # Add batch to GPU
                batch = tuple(t.to(device) for t in batch)
                
                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, b_labels = batch
                
                # Telling the model not to compute or store gradients, saving memory and
                # speeding up validation
                with torch.no_grad():        

                    # Forward pass, calculate logit predictions.
                    # This will return the logits rather than the loss because we have
                    # not provided labels.
                    # token_type_ids is the same as the "segment ids", which 
                    # differentiates sentence 1 and 2 in 2-sentence tasks.
                    # The documentation for this `model` function is here: 
                    # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                    outputs = model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask)
                
                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.
                logits = outputs[0]

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                
            #     # Calculate the accuracy for this batch of test sentences.
            #     tmp_eval_accuracy = flat_accuracy(logits, label_ids)
                
            #     # Accumulate the total accuracy.
            #     eval_accuracy += tmp_eval_accuracy

            #     # Track the number of batches
            #     nb_eval_steps += 1

            # # Report the final accuracy for this validation run.
            # print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
            # print("  Validation took: {:}".format(format_time(time.time() - t0)))

                # Calculate the f1 score for this batch of test sentences.
                temp_f1_validation_micro, temp_f1_validation_each = flat_f1score(logits, label_ids)
                
                # Accumulate the total f1 score.
                f1_validation_micro += temp_f1_validation_micro
                
                # Track the number of batche
                nb_eval_steps += 1
            
            f1_validation_micro = f1_validation_micro/nb_eval_steps
            f1_final_validation_micro = f1_validation_micro
            print("f1_micro: {}".format(f1_validation_micro))
            # print("  f1_meach: {}".format(f1_validation_each))
            print("  Validation took: {:}".format(format_time(time.time() - t0)))
        f1_validation_micro_list.append(f1_final_validation_micro)
        print("")
        print("Training complete!")
    f1_cv_avearge_micro = 0
    for score in f1_validation_micro_list:
        f1_cv_avearge_micro += score/k
    print("Then {}-fold crossvalidation f1 micro score is: {}".format(k,f1_cv_avearge_micro))
    ######################### predict #####################################

    #     y_train_pred = clf.predict(x_train)
    #     f1_train = f1_score(y_train,y_train_pred,average=None)
    #     f1_train_micro = f1_score(y_train,y_train_pred,average='micro')
    #     f1_train_list.append(f1_train)
    #     f1_train_micro_list.append(f1_train_micro)
    #     y_dev_pred = clf.predict(x_dev) 
    #     f1_dev = f1_score(y_dev,y_dev_pred,average=None)
    #     f1_dev_micro = f1_score(y_dev,y_dev_pred,average='micro')
    #     f1_dev_list.append(f1_dev)
    #     f1_dev_micro_list.append(f1_dev_micro)

    # print("Cross validation is done. Here are the scores:")
    # f1_train_average = [0]*len(f1_train_list[0])
    # for i in range(len(f1_train_list)):
    #     for j in range(len(f1_train_list[i])):
    #         f1_train_average[j] += f1_train_list[i][j]/k

    # f1_dev_average = [0]*len(f1_dev_list[0])
    # for i in range(len(f1_dev_list)):
    #     for j in range(len(f1_dev_list[i])):
    #         f1_dev_average[j] += f1_dev_list[i][j]/k

    # f1_train_micro_average = 0
    # for i in range(len(f1_train_micro_list)):
    #     f1_train_micro_average += f1_train_micro_list[i]/k


    # f1_dev_micro_average = 0
    # for i in range(len(f1_dev_micro_list)):
    #     f1_dev_micro_average += f1_dev_micro_list[i]/k

    # f1_scores_dict = {"f1_train_average":f1_train_average,"f1_dev_average":f1_dev_average,"f1_train_micro_average":f1_train_micro_average,"f1_dev_micro_average":f1_dev_micro_average}
    # return f1_scores_dict
    # # print("training score:\n micro average:{}\nfor each class:{}".format(f1_train_micro_average,f1_train_average))
    # # print("validation score:\n micro average:{}\nfor each class:{}".format(f1_dev_micro_average,f1_dev_average))

myBERT_crossvalidation()