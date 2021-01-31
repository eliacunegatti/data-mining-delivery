from numpy.core.records import record
import pandas as pd
import re
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
import sys
pd.options.mode.chained_assignment = None


#function to show the progress bar during the execution of the program
def progress(count, total, status):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('\r[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

#tokenize the tweets
def tokenization(text):
    text = re.split('\W+', text)
    return text


#function to decide which dataset take as input
rseba = True
while(rseba):
    dataset = str(input("Do you want to use the database you provided --> execution time 10/20s or the one I used for the project --> execution time 400/450s?\nA - dataset provided by you\nB - project dataset\n")).lower()
    if (dataset == 'a' or dataset =='b'):
        if(dataset == 'b'):
            df = pd.read_pickle('data/pickle/project_dataset.pkl')
        else:
            df = pd.read_pickle('data/pickle/covid19.pkl')
        rseba=False


progress(0, 100, status='Reading the dataset')

df1= pd.DataFrame()

progress(10, 100, status='Group dataset by day')

#remove symbol as [] '' , from the pre-processed dataset
df['text'] = df['text'].str.replace('[^\w\s]','')

#tokenize each tweet
df['text'] = df['text'].apply(lambda x: tokenization(x.lower()))

#groupby tweets by day
groups = df.groupby('date')['text'].apply(list)
df1 = groups.reset_index(name='text')

total = len(df1)*2
ii = 0
i=0
progress(len(df1)/2, total, status=str(len(df1)))
dflist = []
dflist1 = []

ii=20
#computed the frequent itemsets and association rules according to the days of tweets
while(i< len(df1)):
    records = []
    records = df1['text'].iloc[i]

    #TransactionEncoder to prepare the element in order to give as input to the frequent_itemsets   
    te = TransactionEncoder()

    te_ary = te.fit(records).transform(records)
    df2 = pd.DataFrame()
    df2 = pd.DataFrame(te_ary, columns=te.columns_)

    #compute the frequent_itemsets with FP-Growth algorithm
    frequent_itemsets = fpgrowth(df2, min_support=0.003, use_colnames=True, max_len=2)
    oks = pd.DataFrame()
    oks = frequent_itemsets
    
    #keeping only the itemsets of exactly two words
    oks['length'] = oks['itemsets'].apply(lambda x: len(x))
    oks = oks[(oks['length'] == 2)]

    #append the frequent itemsets of this day to the list
    dflist1.append(oks)
    del oks

    #compute the association rules of the frequent itemsets of the day
    rules = association_rules(frequent_itemsets,metric="confidence", min_threshold=0.5) 
    rules = rules[(rules['lift'] > 10)] 
    prova = pd.DataFrame()
    
    #remove repetaed items e.g (corona-virus), (virus-corona) are kept only once randomly in one of the form
    prova['itemsets'] = [x | y for x, y in zip(rules['antecedents'], rules['consequents'])]
    prova = prova.drop_duplicates(subset=['itemsets'])

    #append the interesting items of this day to the list
    dflist.append(prova) 

    i = i+1

    del df2
    del frequent_itemsets
    ii = ii+1
    progress(ii, total, status='Processing the frequent consistent topics')


#group the total frequent itemsets and counts the occurency
dfx = pd.DataFrame()
dfx = pd.concat(dflist1, axis=0)
dfx = dfx.groupby(['itemsets']).size().reset_index(name='counts')
dfx=dfx.sort_values(["counts"], ascending=False)

progress(50, total, status='Doing very long job')

#group the interesting items and counts the occurency
df3 = pd.DataFrame()
df3 = pd.concat(dflist, axis=0)
ok = pd.DataFrame()
ok = df3.groupby(['itemsets']).size().reset_index(name='counts')
del df3
progress(ii+len(df1)/2, total, status='Doing very long job')
ok=ok.sort_values(["counts"], ascending=False)

#intersection of the two list
int_df = pd.merge(ok, dfx, how='inner', on=['itemsets']) 

#set the threshold
thresh = len(df1)*0.15
thresh1 = len(df1)*0.75
prova = pd.DataFrame()

#keeping element according to the threshold set
prova = int_df[(int_df['counts_y']> thresh) & (int_df['counts_y']< thresh1)]
prova = prova[['itemsets']]

#transfor fronzenset to tuple in order to better visualize the output
i=0
while(i< len(prova)):
    prova['itemsets'].iloc[i] = tuple(prova['itemsets'].iloc[i])
    i= i+1   

#save the result in the main folder
prova.to_csv('results.csv', encoding='utf-8', index=False, header=None)
progress(total, total, status='Finish')

