import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import regex as re

print('I use Pandas version: ' + pd.__version__ + '\n')

print('--- Loading data. ---')

# loading prepared data
df = pd.read_csv(r'Prepared_data.csv', engine='python', sep='delimiter', encoding='ISO-8859-1',
                 delimiter=';', header=0)

# create new column for tokenized text
df['text_tokens'] = df['text'].str.lower()

print('\n\n--------------Tokenization methods.-----------------')

# per default splits around whitespace
text_tok_1 = (df['text_tokens'].str.split())

print('\n1. Tokenizing using str.split(): \n')
print(text_tok_1.str.join('|'))
print()

# using str.findall and str.join with regular expressions
print('2. Tokenizing using str.findall() and regular expressions: \n')

# for unicode character ranges look at unicode list
text_tok_2 = (df['text_tokens'].str.findall(r'[\w-]*[A-Za-z0-9 \-_.\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u00FF][\w-]*'))
# type of text_split_2 is still pandas series
# print('\nType of tokenized text column is' + str(type(text_split_2)))

# separate tokens with a straight line symbol
print(text_tok_2.str.join('|'))
print()

# nltk.download() here again for load 'punkt-module' in Models

# tokenize using nltk word tokenize method
print('3. Tokenizing using nltk word_tokenize method: \n')
df['text_tokens'] = df['text_tokens'].apply(word_tokenize)
print(df['text_tokens'].str.join('|'))

print('\n\n----------Removing stopwords and non alphanumeric characters.------------\n')

# to download module stopwords use nltk.download() one time
# instead of 'from nltk.corpus import stopwords', here u can see and choose packages

# import german stopwords list from the nltk library as a 'set'
stopwords = set(nltk.corpus.stopwords.words('german'))

# print('\nLength of stop_word_list: ' + str(len(stopwords)))
# print('\nGerman stopwords saved as type set:')
# print(type(stopwords))
# print (stopwords)

# save stopwords into file
with open('nltk_german_stop_words.txt', 'w', encoding='ISO-8859-1') as f:
    for word in stopwords:
        f.write('%s\n' % word)


# removing stop words with a
# list comprehension = short syntax for: from an existing list, make a new list based on condition
# basically return all tokens in text, that are not in stopwords
# def remove_stopwords(tokens):
# [t for t in tokens if t not in stopwords]

# function to remove stop words from a list
# df['text_tokens'] = df['text_tokens'].apply(lambda x: [t for t in x if t not in stopwords])
def remove_stopwords(tokenized_text):
    return tokenized_text.apply(lambda x: [t for t in x if t not in stopwords])


df['text_tokens'] = remove_stopwords(df['text_tokens'])


# \w matches any word character (equivalent to [a-zA-Z0-9_])
def remove_non_alphanumerics(tokenized_text):
    return tokenized_text.apply(lambda x: [t for t in x if re.match(r'[\w]', t)])


df['text_tokens'] = remove_non_alphanumerics(df['text_tokens'])

print([df['text_tokens'].str.join('|')])

# optional: add or exclude stopwords
# include_stopwords = {'some word', ' '}
# exclude_stopwords = {' ', ''}
# or use stopwords.append('some word')

# | union/or , - difference
# stopwords |= include_stopwords
# stopwords -= exclude stopwords

# save datafile into new csv file
df.to_csv('Pre_processed_Data.csv', sep=';', encoding='ISO-8859-1', index=None)

# To Save with tab- delimiter
# df.to_csv('Pre_processed_Data2.tsv', sep='\t', encoding='utf-8', index=None)
