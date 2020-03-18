!pip install pytorch-transformers
!pip install transformers
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


from pytorch_transformers import XLNetModel, XLNetTokenizer, XLNetForSequenceClassification
from pytorch_transformers import AdamW

import pandas as pd
import glob
import os
import os.path
import sys
import random
import codecs
import time
import nltk
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

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

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

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
dev_articles = read_articles_from_file_list(dev_folder)
dev_article_ids, dev_span_starts, dev_span_ends, dev_labels = read_predictions_from_file(dev_template_labels_file)
x_dev = []
for i in range(len(dev_article_ids)):
    article = dev_articles[dev_article_ids[i]]
    x_piece = article[int(dev_span_starts[i]):int(dev_span_ends[i])]
    x_dev.append(x_piece)

x_train = [fragment + "[SEP] [CLS]" for fragment in x_train]
x_dev = [fragment + "[SEP] [CLS]" for fragment in x_dev]

label_to_idx_dict, idx_to_label_dict = get_label_dict(y_train)
y_train = label_to_idx(y_train,label_to_idx_dict)

## tokenize
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)

train_tokenized_texts = [tokenizer.tokenize(sent) for sent in x_train]
print ("Tokenize the first sentence:")
print (train_tokenized_texts[0])

MAX_LEN = 64

# Use the XLNet tokenizer to convert the tokens to their index numbers in the XLNet vocabulary
train_input_ids = [tokenizer.convert_tokens_to_ids(x) for x in train_tokenized_texts]

# Pad our input tokens
train_input_ids = pad_sequences(train_input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

print('\nDone.')

# Create attention masks
train_attention_masks = []


# Create a mask of 1s for each token followed by 0s for padding
for seq in train_input_ids:
  seq_mask = [float(i>0) for i in seq]
  train_attention_masks.append(seq_mask)


## Repeat the same process to validation set
validation_tokenized_texts = [tokenizer.tokenize(sent) for sent in x_dev]
print ("Tokenize the first sentence:")
print (validation_tokenized_texts[0])

MAX_LEN = 64

# Use the XLNet tokenizer to convert the tokens to their index numbers in the XLNet vocabulary
validation_input_ids = [tokenizer.convert_tokens_to_ids(x) for x in validation_tokenized_texts]

# Pad our input tokens
validation_input_ids = pad_sequences(validation_input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

print('\nDone.')

# Create attention masks
validation_attention_masks = []


# Create a mask of 1s for each token followed by 0s for padding
for seq in validation_input_ids:
  seq_mask = [float(i>0) for i in seq]
  validation_attention_masks.append(seq_mask)

train_labels = y_train
validation_labels = [0]*len(validation_input_ids)

train_inputs = torch.tensor(train_input_ids)
validation_inputs = torch.tensor(validation_input_ids)

train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)

train_masks = torch.tensor(train_attention_masks)
validation_masks = torch.tensor(validation_attention_masks)

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# The DataLoader needs to know our batch size for training, so we specify it 
# here.
# For fine-tuning BERT on a specific task, the authors recommend a batch size of
# 16 or 32.

batch_size = 12

# Create the DataLoader for our training set.
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Create the DataLoader for our validation set.
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

# Load BertForSequenceClassification, the pretrained XLNet model with a single 
# linear classification layer on top. 
model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=14)


# Tell pytorch to run this model on the GPU.
model.cuda()

# Get all of the model's parameters as a list of tuples.
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
# I believe the 'W' stands for 'Weight Decay fix"
# This variable contains all of the hyperparemeter information our training loop needs
optimizer = AdamW(optimizer_grouped_parameters,
                     lr=2e-5)


from transformers import get_linear_schedule_with_warmup

# Number of training epochs (authors recommend between 2 and 4)
epochs = 4

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

import datetime



# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Store the average loss after each epoch so we can plot them.
loss_values = []
y_dev_pred = []
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
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    total_pred = []
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
        logits = logits.detach().cpu().numpy()
        dev_pred = np.argmax(logits, axis=1).flatten()
        dev_pred = list(dev_pred)
        total_pred += dev_pred
    y_dev_pred = total_pred

########################## Output ######################################

y_dev_pred = idx_to_label(y_dev_pred,idx_to_label_dict)
print(len(y_dev_pred))
print(y_dev_pred)
with open(task_TC_output_file, "w") as fout:
    for article_id, prediction, span_start, span_end in zip(dev_article_ids, y_dev_pred, dev_span_starts, dev_span_ends):
        fout.write("%s\t%s\t%s\t%s\n" % (article_id, prediction, span_start, span_end))
print("Predictions written to file " + task_TC_output_file)





print("")
print("Training complete!")