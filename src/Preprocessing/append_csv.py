import numpy as np
import pandas as pd
import re
import nltk
import spacy
import string

dflist = []
df2 = pd.DataFrame()
df = pd.read_csv('Altro/covid19_en.csv', sep=",")
df2['text'] = df['text']
df2['date'] = df['date']
dflist.append(df2)
del df
del df2
print("uno")

df2 = pd.DataFrame()
dc = pd.read_csv('Altro/corona_tweets.csv', sep=",")
df2['text'] = dc['text']
df2['date'] = dc['date']
dflist.append(df2)
del dc
del df2
print("due")

df2 = pd.DataFrame()
dk = pd.read_csv('Input/covid19.csv', sep=";")
df2['text'] = dk['text']
df2['date'] = dk['date']
dflist.append(df2)
del dk
del df2
print("tre")

df2 = pd.DataFrame()
dx = pd.read_csv('Altro/us_tweets.csv', sep=",")
df2['text'] = dx['text']
df2['date'] = dx['date']
dflist.append(df2)
del dx
del df2
print("quattro")

df1 = pd.DataFrame()
df1 = pd.concat(dflist, axis=0)
df1.to_csv('Altro/final_dataset.csv',index=False, encoding='utf-8',columns=['date', 'text'])
print("fine")
