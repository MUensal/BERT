import sys
import transformers
import pandas as pd

print('\n--------------SCRIPT START-----------------------------')
print('Transformers version: ' + transformers.__version__)
print('Python version: ' + sys.version)
print('-------------------------------------------------------\n')

# Loading datafile and display information   # encoding="ISO-8859-1" instead of "utf-8"
df = pd.read_csv(r'Training_data.csv', engine='python', sep='delimiter', encoding='ISO-8859-1',
                 delimiter=';', header=0, )

print('-----------Dataset Info-----------------\n')
print('Data Shape: ' + str(df.shape))
print(df.columns)
print(df.head(3) + '\n')
print(df[['text_id', 'text', 'task_1', 'task_2']].describe(include='O').T)

# Dataset contains 407 HOF( Hate OR Offensive comments
hof_data = df[df['task_1'].str.contains('HOF')]['task_1'].value_counts()
hate_data = df[df['task_2'].str.contains('HATE')]['task_2'].value_counts()
not_data = df[df['task_1'].str.contains('NOT')]['task_1'].value_counts()

print(type(df.values))
print(type(df.index))

print('Number of Hate, Offensive, or None comments: ')
print(hof_data)
print(hate_data)
print(not_data)

print('\n-----------Dataset modification-----------------')

# deleting urls of tweets
df['text'] = df['text'].str.split('https').str[0]
print(df.head(5))
print(df)

# add a column length to dataset
df['length'] = df['text'].str.len()
print(df.head(3))

# sort data according to text length
# df1 = df['text'].str.len().sort_values()
# print(df1)

# delete rows with text less than 30 characters
for le in df.index:
    if df.loc[le, 'length'] < 30:
        df.drop(le, inplace=True)

# new shape leaves 3609 entries /header
print(df.shape)

# a dictionary to select and rename columns
column_mapping = {
    'text_id': None,
    'text': 'text',
    'task_1': 'hof_OR_none',
    'task_2': None,
    'length': None
}
# select columns that stay
columns = [c for c in column_mapping.keys()
           if column_mapping[c] is not None]

# rename columns with dictionary
df = df[columns].rename(columns=column_mapping)

# show random row transposed
print(df.sample(1).T)

# remove th @ in mentions and replies \@ removes all @, | is delimiter
df['text'] = df['text'].str.replace(r'\@|\#', '')

# save datafile into new csv file
df.to_csv('Prepared_data.csv', sep=';', encoding='ISO-8859-1', index=None)

df_new = pd.read_csv('Prepared_data.csv', engine='python', sep='delimiter',
                     delimiter=';', header=0)

print(df_new.shape)

print('\n-------------------------------------')
