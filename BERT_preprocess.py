from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import regex as re

# instantiating fromm the pre-trained model gbert for German language
config = BertConfig.from_pretrained("deepset/gbert-base", finetuning_task='binary')
tokenizer = BertTokenizer.from_pretrained("deepset/gbert-base")
model = BertForSequenceClassification.from_pretrained("deepset/gbert-base")

# loading prepared data
df = pd.read_csv(r'Prepared_data.csv', engine='python', sep='delimiter', encoding='ISO-8859-1',
                 delimiter=';', header=0)

print('------------TOKENIZER PIPELINE-------------\n\n')

# lowercase tokens?
df['text'] = df['text'].str.lower()

# max seq len according to model architecture of gbert-base

# instead i will be using truncation=True, automatically to models max_seq_len
# MAX_LEN = 32

# encode tokens into ids
input_ids = []
for row in df['text']:
    input_ids.append(tokenizer.encode(row,
                                      add_special_tokens=True,
                                      max_length=32,
                                      padding='max_length',
                                      ))

# print(input_ids)

# check if the # of rows is the same
print()
print(len(df['text']))
print(len(input_ids))

# Attention mask:  1 is for all input tokens and 0 is for all padding tokens
attention_masks = [[float(id > 0) for id in seq] for seq in input_ids]

# convert ids to tokens - doesnt work like this because i have a list, not just a sentence
# input_tokens = tokenizer.convert_ids_to_tokens(input_ids)

# convert just a row into token representation
input_tokens3 = tokenizer.convert_ids_to_tokens(input_ids[1])


# or like this
# input_tokens = tokenizer.decode(input_ids[2])
# TO DO: find out host to do this for the whole list

print('\n##################  Sample row 3  ################\n')

print('\nBefore tokenization:  ' + df['text'].iloc[1] + '\n')
print('After tokenization:  ' + str(input_ids[1]) + '\n')
print('Tokens representation:  ' + str(input_tokens3) + '\n')
print('Attention mask:  ' + str(attention_masks[1]) + '\n')

# print(df.sample(5))

# Split data 20 to 80, test and train size
X_train, X_test, Y_train, Y_test = train_test_split(df['text'],
                                                    df['hof_OR_none'],
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=df['hof_OR_none']
                                                    )
# tokenize training data
X_train_tokens = []
for row in X_train:
    X_train_tokens.append(tokenizer.encode(row,
                                           add_special_tokens=True,
                                           max_length=32,
                                           padding='max_length'
                                           ))

# tokenize test data
X_test_tokens = []
for row in X_test:
    X_test_tokens.append(tokenizer.encode(row,
                                          add_special_tokens=True,
                                          max_length=32,
                                          padding='max_length'
                                          ))

# print(X_train_tokens)
# print(X_test_tokens)

print('\n############  tensors  #################\n')

# save datafile into new csv file
# df.to_csv('Bert_Data.csv', sep=';', encoding='ISO-8859-1', index=None)

