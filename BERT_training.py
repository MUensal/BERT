import sys
import time

import torch as torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np

print('Python version     : ' + str(sys.version))
print('Torch version   : ' + str(torch.__version__))


# instantiating from the pre-trained model gbert for German language
config = BertConfig.from_pretrained("deepset/gbert-base", finetuning_task='binary', num_labels=2)
tokenizer = BertTokenizer.from_pretrained("deepset/gbert-base")
model = BertForSequenceClassification.from_pretrained("deepset/gbert-base")

# loading prepared data
df = pd.read_csv(r'Prepared_data.csv', engine='python', sep='delimiter', encoding='ISO-8859-1',
                 delimiter=';', header=0)

# lowercase tokens (do i need it?)
df['text'] = df['text'].str.lower()

# renaming the column hof_OR_none to labels, inplace updates original object
df.rename(columns={'hof_OR_none': 'labels'}, inplace=True)

# print(df.head(5))


# converting labels into numerical values, in order to save it into a tensor
df['labels'] = df['labels'].replace(['HOF', 'NOT'], [1, 0])

# check new data type of column labels: should be int now
# print(df.labels.dtypes)

# check data: print(df['labels'].sample(15))

# DATA SPLIT TO TRAINING AND TEST
# size 20 to 80
X_train, X_test, Y_train, Y_test = train_test_split(df['text'],
                                                    df['labels'],
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=df['labels']
                                                    )

# TOKENIZATION

# tokenize training set
X_train_tokens = []
for row in X_train:
    X_train_tokens.append(tokenizer.encode(row,
                                           add_special_tokens=True,
                                           max_length=50,
                                           truncation=True,
                                           padding='max_length'
                                           ))

# creating the attention mask for training data
att_mask_train = [[float(id > 0) for id in seq] for seq in X_train_tokens]

# tokenize test set
X_test_tokens = []
for row in X_test:
    X_test_tokens.append(tokenizer.encode(row,
                                          add_special_tokens=True,
                                          max_length=50,
                                          truncation=True,
                                          padding='max_length',
                                          # return_tensors='pt'  # returns tensors
                                          ))

# creating the attention mask for test data
att_mask_test = [[float(id > 0) for id in seq] for seq in X_test_tokens]

# CURRENT DATA TYPES
print(att_mask_train[0])
print(type(X_train_tokens))
print(type(att_mask_train))
print(type(Y_train))
# print(len(X_train_tokens[2]))

# Create Tensors for training set, attention mask, and training labels
input_ids_train = torch.LongTensor(X_train_tokens)
input_mask_train = torch.FloatTensor(att_mask_train)
label_ids_train = torch.tensor(Y_train.values)

# Test Data Tensors
ids_test_data = torch.LongTensor(X_test_tokens)
input_mask_test = torch.FloatTensor(att_mask_test)
label_ids_test = torch.tensor(Y_test.values)


# shapes of this tensors
print('\n------------------------')
print(input_ids_train.shape)
print(input_mask_train.shape)
print(label_ids_train.shape)
print('------------------------')

# input example
print('sample input tokens: ' + str(input_ids_train[1]))
print('attention mask: ' + str(input_mask_train[1]))
print('label: ' + str(label_ids_train[1]))

# Concatenate tensors into one tensor
training_data = TensorDataset(input_ids_train, input_mask_train, label_ids_train)
test_data = TensorDataset(ids_test_data, input_mask_test, label_ids_test)

# MODEL TRAINING
print('-----------------------')

# split observations into batches, train for 2 epochs
batch_size = 64
num_train_epochs = 2

train_sampler = RandomSampler(training_data)

# Dataloader is used to iterate over batches
train_dataloader = DataLoader(training_data,
                              sampler=train_sampler,
                              batch_size=batch_size)

# //: divide and discard remainder (Division ohne Rest)
t_total = len(train_dataloader) // num_train_epochs

# this is used in the huggingface example
# num_training_steps = train_epochs * len(train_dataloader)
# print(num_training_steps)

# Learning variables
print(len(training_data))
print(num_train_epochs)
print(batch_size)
print(t_total)

# set learning parameters
learning_rate = 1e-4
adam_epsilon = 1e-8
warmup_steps = 0

# for parameter adjustment
optimizer = AdamW(model.parameters(), lr=learning_rate, eps=adam_epsilon)

#  define a learning rate scheduler
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=warmup_steps,
                                            num_training_steps=t_total)

# if available, the gpu will be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# output is too long
# print(model)

# tqdm is for tracking the training progress
progress_bar = tqdm(range(t_total))

# Put model in 'train'
model.train()

print("\nTraining Model...")
print('--- Start Time ' + str(time.strftime('%c')))

# wrap epoch and batches in loops
for epoch in range(num_train_epochs):
    print('\nStart of epoch %d' % (epoch,))
    # iterate over data/batches
    for step, batch in enumerate(train_dataloader):
        # all gradients reset at start of every iteration
        # model.zero_grad()

        # print(device)
        batch = tuple(t.to(device) for t in batch)

        # set inputs of the model
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2]}

        # forward to model
        outputs = model(**inputs)

        # deviation (loss)
        # loss = outputs[0]
        loss = outputs.loss
        print("\r%f" % loss, end='')

        # Backpropagation
        loss.backward()

        # limit gradients to 1.0 to prevent exploding gradients -->  is deprecated
        # torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)

        # update parameters and learning rate
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

print('--- End Time ' + str(time.strftime('%c')))

# save model
model.save_pretrained('BERT_model_1')

# EVALUATE MODEL

test_batch_size = 64

# this time samples are fed sequentially, not random
test_sampler = SequentialSampler(test_data)

# dataloader for the test data
test_dataloader = DataLoader(test_data,
                             sampler=test_sampler,
                             batch_size=test_batch_size)

# load model if needed, by
# model = model.from_pretrained('BERT_model_1')

# Init for prediction and labels
preds = None
out_labels_ids = None


# evaluation mode
model.eval()

for batch in tqdm(test_dataloader, desc="Evaluating"):

    model.to(device)
    batch = tuple(t.to(device) for t in batch)

    # no gradients tracking because testing
    with torch.no_grad():
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2]}

        outputs = model(**inputs)

        # get loss
        tmp_eval_loss, logits = outputs[:2]

        # batch items check
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids,
                                      inputs['labels'].detach().cpu().numpy(),
                                      axis=0)

# Get prediction and accuracy
preds = np.argmax(preds, axis=1)
acc_score = accuracy_score(preds, out_label_ids)
print('\nAccuracy Score on Test data ', acc_score)






