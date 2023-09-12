#!/usr/bin/env python
# coding: utf-8

# In[31]:


import csv
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from collections import defaultdict

df2 = pd.read_csv('BERT_PREDICTIONS.csv')

df3 = pd.read_csv('df_with_moral_embeddings.csv')

#making sure that the columns for analysis from previous analyses are in the correct format/have the correct labels.
# This function recodes any of the foundation which could be labelled as 2,3,5 (the categories which are not being assess due to rareity are removed)
def recode_moral(value):
    if value in [2, 3, 5]:
        return 6
    else:
        return value

# Apply the custom function to create the 'recoded' column
df2['recoded'] = df2['moral_mf_single'].apply(recode_moral)
#df2.head()

df3['recoded'] = df3['moral_mf_single'].apply(recode_moral)
#df3.head(20)


# In[32]:


#here I am creating the foreign secretaries subset for df3 (the dataset which has the word embedding assigned labels contrained within it)

#this is the list of all foreign secretaries contained in the dataset. 
mps_to_filter = [
    'James Cleverly', 'Elizabeth Truss', 'Dominic Raab', 'Jeremy Hunt', 'Boris Johnson',
    'Philip Hammond', 'William Hague', 'David Miliband', 'Margaret Beckett', 'Jack Straw',
    'Robin Cook', 'Malcolm Rifkind', 'Douglas Hurd', 'John Major', 'Geoffrey Howe'
]

filtered_df = df3[df3['speaker2'].isin(mps_to_filter)]

# filtering for references said by foreign secretaries based of off the year that they they were foreign secretary in
#Dictionary of foreign secretaries and their tenures which was gained using wikipedia.
foreign_secretary_tenures = {
    'James Cleverly': (2022, 2023),
    'Elizabeth Truss': (2021, 2022),
    'Dominic Raab': (2019, 2021),
    'Jeremy Hunt': (2018, 2019),
    'Boris Johnson': (2016, 2018),
    'Philip Hammond': (2014, 2016),
    'William Hague': (2010, 2014),
    'David Miliband': (2007, 2010),
    'Margaret Beckett': (2006, 2007),
    'Jack Straw': (2001, 2006),
    'Robin Cook': (1997, 2001),
    'Malcolm Rifkind': (1995, 1997),
    'Douglas Hurd': (1989, 1995),
    'John Major': (1989, 1989), 
    'Geoffrey Howe': (1983, 1989)
}

filtered_sentences_df1 = pd.DataFrame()

# This block of code iterates over each of the foriegn secretaries based on their year range.  
for mp, year_range in foreign_secretary_tenures.items():
    start_year, end_year = year_range[0], year_range[1]
    mp_sentences = filtered_df[(filtered_df['speaker2'] == mp) & (filtered_df['Year'] >= start_year) & (filtered_df['Year'] <= end_year)]
    filtered_sentences_df1 = pd.concat([filtered_sentences_df1, mp_sentences])

#counting the references per foreign secretaries (it should be references not sentences in the name - however I have carried this typo 
#through all the code in this section, hence it remains).
sentences_per_foreign_secretary = filtered_sentences_df1.groupby('speaker2').size().reset_index(name='Number_of_references')

filtered_sentences_df1.head()


# NOTE: much of the following code is essentially, the same initial block of code repeated again and again to calculate proportions for different models for both moral sentiment and moral foundations. So, the dataframe names change, the names of the columns and the labels themsevles namely either 0,1,2 or 0,1,4,6 but the structure of the code is identicial. 

# In[38]:


#THIS CODE IS FOR CALCULATING THE PROPORTIONS FOR THE DISTILEBRT MODEL FOR MORAL FOUNDATION CATEGORIES. 
#IT COMPARES THE PROPRTIONS FOR ALL MPS TO ALL FOREIGN SECRETARIES.

# Calculating proportions for each foundation 
proportions_conservative = {}
proportions_labour = {}

for foundation in [0, 1, 4, 6]:
    foundation_texts_all = df2[df2['predicted_labels'] == foundation]
    foundation_texts_fs = filtered_sentences_df1[filtered_sentences_df1['predicted_labels'] == foundation]

    proportion_con = len(foundation_texts_all) / len(df2) if len(df2) > 0 else 0.0
    proportion_lab = len(foundation_texts_fs) / len(filtered_sentences_df1) if len(filtered_sentences_df1) > 0 else 0.0

    proportions_conservative[foundation] = proportion_con
    proportions_labour[foundation] = proportion_lab

#setting a level for alpha to compare the p-values to although I also looked at the values of alpha 0.01 and 0.1.
alpha = 0.05

for foundation in [0, 1, 4, 6]:
    print(f"Moral Foundation {foundation}")
    print(f"Proportion of All MPs Texts: {proportions_conservative[foundation]}")
    print(f"Proportion of foreign sec Texts: {proportions_labour[foundation]}")

    #these lines of code are used to calculate the upper and lower bounds for 95% confident intervales
    se_con = np.sqrt(proportions_conservative[foundation] * (1 - proportions_conservative[foundation]) / len(df3))
    se_lab = np.sqrt(proportions_labour[foundation] * (1 - proportions_labour[foundation]) / len(filtered_sentences_df1))

    ci_con = (proportions_conservative[foundation] - 1.96 * se_con, proportions_conservative[foundation] + 1.96 * se_con)
    ci_lab = (proportions_labour[foundation] - 1.96 * se_lab, proportions_labour[foundation] + 1.96 * se_lab)

    print(f"Confidence Interval: {ci_con}")
    print(f"Confidence Interval: {ci_lab}")
    print()
#this code is setting the calculating for the pooled proportions and the z-scores, and then finally the p-values.
    p1 = proportions_labour[foundation]
    p2 = proportions_conservative[foundation]
    n1 = len(filtered_sentences_df1)
    n2 = len(df2)

    pooled_proportion = (p1 * n1 + p2 * n2) / (n1 + n2)
    standard_error = np.sqrt(pooled_proportion * (1 - pooled_proportion) * (1/n1 + 1/n2))
    z_score = (p1 - p2) / standard_error
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score))) #this was calcualted manually throughout the code as I had issues
    #with getting proprotions that summed up correctly/were dividing by the right thing when using automatic stats api calculations
    #for the confidence intervals ans the proportions.

    print(f"Z-Score: {z_score}")
    print(f"P-Value: {p_value}")
#this code determines whether the p-value is significant or not.
    if p_value <= alpha:
        print("Difference is statistically significant.")
    else:
        print("Difference is not statistically significant.")

    if z_score > 0:
        print("Foreign secs have a higher proportion.")
    else:
        print("All MPs have a higher proportion.")

    print()


# In[37]:


import numpy as np
import scipy.stats as stats

#THIS CODE IS FOR CALCULATING THE PROPORTIONS FOR THE CUSTOMISED DICTIONARY MODEL FOR MORAL FOUNDATION CATEGORIES. 
#IT COMPARES THE PROPRTIONS FOR ALL MPS TO ALL FOREIGN SECRETARIES.

# Calculating proportions for each foundation 
proportions_conservative = {}
proportions_labour = {}

for foundation in [0, 1, 4, 6]:
    foundation_texts_all = df2[df2['recoded'] == foundation]
    foundation_texts_fs = filtered_sentences_df1[filtered_sentences_df1['recoded'] == foundation]

    proportion_con = len(foundation_texts_all) / len(df2) if len(df2) > 0 else 0.0
    proportion_lab = len(foundation_texts_fs) / len(filtered_sentences_df1) if len(filtered_sentences_df1) > 0 else 0.0

    proportions_conservative[foundation] = proportion_con
    proportions_labour[foundation] = proportion_lab

#setting a level for alpha to compare the p-values to although I also looked at the values of alpha 0.01 and 0.1.
alpha = 0.05

for foundation in [0, 1, 4, 6]:
    print(f"Moral Foundation {foundation}")
    print(f"Proportion of All MPs Texts: {proportions_conservative[foundation]}")
    print(f"Proportion of foreign sec Texts: {proportions_labour[foundation]}")
    
    #I have to do this calculation more manuallythat with some stats tools (same calculation repeated throughout) do to 
    #having to make sure that the calculation was dividing by the right  length/the right dataframe. the calculation
    #is based off of this code: https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
    
    
    #these lines of code are used to calculate the upper and lower bounds for 95% confident intervales
    se_con = np.sqrt(proportions_conservative[foundation] * (1 - proportions_conservative[foundation]) / len(df3))
    se_lab = np.sqrt(proportions_labour[foundation] * (1 - proportions_labour[foundation]) / len(filtered_sentences_df1))

    ci_con = (proportions_conservative[foundation] - 1.96 * se_con, proportions_conservative[foundation] + 1.96 * se_con)
    ci_lab = (proportions_labour[foundation] - 1.96 * se_lab, proportions_labour[foundation] + 1.96 * se_lab)

    print(f"Confidence Interval: {ci_con}")
    print(f"Confidence Interval: {ci_lab}")
    print()
#this code is setting the calculating for the pooled proportions and the z-scores, and then finally the p-values.
    p1 = proportions_labour[foundation]
    p2 = proportions_conservative[foundation]
    n1 = len(filtered_sentences_df1)
    n2 = len(df2)

    pooled_proportion = (p1 * n1 + p2 * n2) / (n1 + n2)
    standard_error = np.sqrt(pooled_proportion * (1 - pooled_proportion) * (1/n1 + 1/n2))
    z_score = (p1 - p2) / standard_error
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score))) 

    print(f"Z-Score: {z_score}")
    print(f"P-Value: {p_value}")
#this code determines whether the p-value is significant or not.
    if p_value <= alpha:
        print("Difference is statistically significant.")
    else:
        print("Difference is not statistically significant.")

    if z_score > 0:
        print("Foreign secs have a higher proportion.")
    else:
        print("All MPs have a higher proportion.")

    print()


# In[3]:


import numpy as np
import scipy.stats as stats

#THIS CODE IS FOR CALCULATING THE PROPORTIONS FOR THE WORD2VEC MODEL FOR MORAL FOUNDATION CATEGORIES. 
#IT COMPARES THE PROPRTIONS FOR ALL MPS TO ALL FOREIGN SECRETARIES.

# Calculating proportions for each foundation
proportions_conservative = {}
proportions_labour = {}

for foundation in [0, 1, 4, 6]:
    foundation_texts_all = df3[df3['moral_embeddings2'] == foundation]
    foundation_texts_fs = filtered_sentences_df1[filtered_sentences_df1['moral_embeddings2'] == foundation]

    proportion_con = len(foundation_texts_all) / len(df3) if len(df3) > 0 else 0.0
    proportion_lab = len(foundation_texts_fs) / len(filtered_sentences_df1) if len(filtered_sentences_df1) > 0 else 0.0

    proportions_conservative[foundation] = proportion_con
    proportions_labour[foundation] = proportion_lab

#setting a level for alpha to compare the p-values to although I also looked at the values of alpha 0.01 and 0.1.
alpha = 0.05

for foundation in [0, 1, 4, 6]:
    print(f"Moral Foundation {foundation}")
    print(f"Proportion of All MPs Texts: {proportions_conservative[foundation]}")
    print(f"Proportion of foreign sec Texts: {proportions_labour[foundation]}")

    #these lines of code are used to calculate the upper and lower bounds for 95% confident intervales
    se_con = np.sqrt(proportions_conservative[foundation] * (1 - proportions_conservative[foundation]) / len(df3))
    se_lab = np.sqrt(proportions_labour[foundation] * (1 - proportions_labour[foundation]) / len(filtered_sentences_df1))

    ci_con = (proportions_conservative[foundation] - 1.96 * se_con, proportions_conservative[foundation] + 1.96 * se_con)
    ci_lab = (proportions_labour[foundation] - 1.96 * se_lab, proportions_labour[foundation] + 1.96 * se_lab)

    print(f"Confidence Interval: {ci_con}")
    print(f"Confidence Interval: {ci_lab}")
    print()
#this code is setting the calculating for the pooled proportions and the z-scores, and then finally the p-values.
    p1 = proportions_labour[foundation]
    p2 = proportions_conservative[foundation]
    n1 = len(filtered_sentences_df1)
    n2 = len(df3)

    pooled_proportion = (p1 * n1 + p2 * n2) / (n1 + n2)
    standard_error = np.sqrt(pooled_proportion * (1 - pooled_proportion) * (1/n1 + 1/n2))
    z_score = (p1 - p2) / standard_error
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

    print(f"Z-Score: {z_score}")
    print(f"P-Value: {p_value}")
#this code determines whether the p-value is significant or not.
    if p_value <= alpha:
        print("Difference is statistically significant.")
    else:
        print("Difference is not statistically significant.")

    if z_score > 0:
        print("Foreign secs have a higher proportion.")
    else:
        print("All MPs have a higher proportion.")

    print()


# In[7]:


##THIS IS THE CODE FOR COMPARING ALL FOREIGN SECERTARIES AND ALL MPS FOR THE WORD2VEC MODEL BUT FOR MORAL SENTIMENT. 

#Here I am calculating the proportions of each category in the moral embeddings column for the word2vec model 
proportions = len(df3['moral_embeddings'])

proportions2 = len(filtered_sentences_df1['moral_embeddings'])

proportions_conservative = {0: 0.0, 1: 0.0, 2: 0.0}
proportions_labour = {0: 0.0, 1: 0.0, 2: 0.0}

for foundation in [0, 1, 2]:
    foundation_texts_all = df3[(df3['moral_embeddings'] ==foundation)]
    foundation_texts_fs = filtered_sentences_df1[(filtered_sentences_df1['moral_embeddings'] == foundation)]

    proportion_con = len(foundation_texts_all) / proportions if proportions > 0 else 0.0
    proportion_lab = len(foundation_texts_fs) / proportions2 if proportions2 > 0 else 0.0

    proportions_conservative[foundation] = proportion_con
    proportions_labour[foundation] = proportion_lab

# this code will show the proportions and confidence intervals for moral sent. for foreign secs and all mps
for foundation in [0, 1, 2]:
    print(f"Moral Foundation {foundation}")
    print(f"Proportion of All MPs Texts: {proportions_conservative[foundation]}")
    print(f"Proportion of foreign sec Texts: {proportions_labour[foundation]}")

    #  standard errors 
    se_con = np.sqrt(proportions_conservative[foundation] * (1 - proportions_conservative[foundation]) /proportions)
    se_lab = np.sqrt(proportions_labour[foundation] * (1 - proportions_labour[foundation]) / proportions2)

    #  confidence intervals
    ci_con = (proportions_conservative[foundation] - 1.96 * se_con, proportions_conservative[foundation] + 1.96 * se_con)
    ci_lab = (proportions_labour[foundation] - 1.96 * se_lab, proportions_labour[foundation] + 1.96 * se_lab)

    print(f"Confidence Interval: {ci_con}")
    print(f"Confidence Interval: {ci_lab}")
    print()
    
    
import scipy.stats as stats

# Setting the significance level
alpha = 0.05

# This code performs the z-test for proportions for each moral sent category
for foundation in [0, 1, 2]:
    p1 = proportions_labour[foundation]
    p2 = proportions_conservative[foundation]
    n1 = proportions2
    n2 = proportions
    
    pooled_proportion = (p1 * n1 + p2 * n2) / (n1 + n2)
    standard_error = np.sqrt(pooled_proportion * (1 - pooled_proportion) * (1/n1 + 1/n2))
    z_score = (p1 - p2) / standard_error
    
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  #here I have selected the two-tailed test. 
    
    print(f"Moral Foundation {foundation}")
    print(f"Z-Score: {z_score}")
    print(f"P-Value: {p_value}")
    
    if p_value < alpha:
        print("Difference is statistically significant.")
    else:
        print("Difference is not statistically significant.")
    
    if z_score > 0:
        print("foreign secs have a higher proportion.")
    else:
        print("All Mps a higher proportion.")
    
    print()


# In[40]:


#THIS IS THE CODE FOR COMPARING ALL FOREIGN SECERTARIES AND ALL MPS FOR THE CUSTOMISED DICTIONARY MODEL FOR MORAL SENTIMENT. 


#Here I am calculating the proportions of each category in the moral sentiment column for the dictionary model 
proportions = len(df2['moral_sentiment'])

proportions2 = len(filtered_sentences_df1['moral_sentiment'])

proportions_conservative = {0: 0.0, 1: 0.0, 2: 0.0}
proportions_labour = {0: 0.0, 1: 0.0, 2: 0.0}

for foundation in [0, 1, 2]:
    foundation_texts_all = df2[(df2['moral_sentiment'] ==foundation)]
    foundation_texts_fs = filtered_sentences_df1[(filtered_sentences_df1['moral_sentiment'] == foundation)]

    proportion_con = len(foundation_texts_all) / proportions if proportions > 0 else 0.0
    proportion_lab = len(foundation_texts_fs) / proportions2 if proportions2 > 0 else 0.0

    proportions_conservative[foundation] = proportion_con
    proportions_labour[foundation] = proportion_lab

# proportions and confidence intervals for each sent category are calculated here 
for foundation in [0, 1, 2]:
    print(f"Moral Foundation {foundation}")
    print(f"Proportion of All MPs Texts: {proportions_conservative[foundation]}")
    print(f"Proportion of foreign sec Texts: {proportions_labour[foundation]}")

    # standard errors 
    se_con = np.sqrt(proportions_conservative[foundation] * (1 - proportions_conservative[foundation]) /proportions)
    se_lab = np.sqrt(proportions_labour[foundation] * (1 - proportions_labour[foundation]) / proportions2)

    # confidence intervals
    ci_con = (proportions_conservative[foundation] - 1.96 * se_con, proportions_conservative[foundation] + 1.96 * se_con)
    ci_lab = (proportions_labour[foundation] - 1.96 * se_lab, proportions_labour[foundation] + 1.96 * se_lab)

    print(f"Confidence Interval: {ci_con}")
    print(f"Confidence Interval: {ci_lab}")
    print()

# Setting  the significance level
alpha = 0.05

# here I am performing the  z-test for proportions for each moral sentiment category 
for foundation in [0, 1, 2]:
    p1 = proportions_labour[foundation]
    p2 = proportions_conservative[foundation]
    n1 = proportions2
    n2 = proportions
    
    pooled_proportion = (p1 * n1 + p2 * n2) / (n1 + n2)
    standard_error = np.sqrt(pooled_proportion * (1 - pooled_proportion) * (1/n1 + 1/n2))
    z_score = (p1 - p2) / standard_error
    
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # here I have selected the two-tailed test
    
    print(f"Moral Foundation {foundation}")
    print(f"Z-Score: {z_score}")
    print(f"P-Value: {p_value}")
    
    if p_value < alpha:
        print("Difference is statistically significant.")
    else:
        print("Difference is not statistically significant.")
    
    if z_score > 0:
        print("foreign secs have a higher proportion.")
    else:
        print("All Mps a higher proportion.")
    
    print()


# In[42]:


#THIS IS THE CODE FOR COMPARING ALL FOREIGN SECERTARIES AND ALL MPS FOR THE BERT MODEL FOR MORAL SENTIMENT. 

# here I am calculating the proportions of each category in the 'predicted_label2' column for moral sentiment
proportions = len(df2['predicted_labels2'])

proportions2 = len(filtered_sentences_df1['predicted_labels2'])

proportions_conservative = {0: 0.0, 1: 0.0, 2: 0.0}
proportions_labour = {0: 0.0, 1: 0.0, 2: 0.0}

for foundation in [0, 1, 2]:
    foundation_texts_all = df2[(df2['predicted_labels2'] ==foundation)]
    foundation_texts_fs = filtered_sentences_df1[(filtered_sentences_df1['predicted_labels2'] == foundation)]

    proportion_con = len(foundation_texts_all) / proportions if proportions > 0 else 0.0
    proportion_lab = len(foundation_texts_fs) / proportions2 if proportions2 > 0 else 0.0

    proportions_conservative[foundation] = proportion_con
    proportions_labour[foundation] = proportion_lab

# This code compute the proportions and confidence intervals for each moral sent categroy 
for foundation in [0, 1, 2]:
    print(f"Moral Foundation {foundation}")
    print(f"Proportion of All MPs Texts: {proportions_conservative[foundation]}")
    print(f"Proportion of foreign sec Texts: {proportions_labour[foundation]}")

    # standard errors 
    se_con = np.sqrt(proportions_conservative[foundation] * (1 - proportions_conservative[foundation]) /proportions)
    se_lab = np.sqrt(proportions_labour[foundation] * (1 - proportions_labour[foundation]) / proportions2)

    # confidence intervals
    ci_con = (proportions_conservative[foundation] - 1.96 * se_con, proportions_conservative[foundation] + 1.96 * se_con)
    ci_lab = (proportions_labour[foundation] - 1.96 * se_lab, proportions_labour[foundation] + 1.96 * se_lab)

    print(f"Confidence Interval: {ci_con}")
    print(f"Confidence Interval: {ci_lab}")
    print()
    
import scipy.stats as stats

alpha = 0.05

# here I am performing the z-test for proportions for each moral sentiment category
for foundation in [0, 1, 2]:
    p1 = proportions_labour[foundation]
    p2 = proportions_conservative[foundation]
    n1 = proportions2
    n2 = proportions
    
    pooled_proportion = (p1 * n1 + p2 * n2) / (n1 + n2)
    standard_error = np.sqrt(pooled_proportion * (1 - pooled_proportion) * (1/n1 + 1/n2))
    z_score = (p1 - p2) / standard_error
    
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Here I have selected the two-tailed test
    
    print(f"Moral Foundation {foundation}")
    print(f"Z-Score: {z_score}")
    print(f"P-Value: {p_value}")
    
    if p_value < alpha:
        print("Difference is statistically significant.")
    else:
        print("Difference is not statistically significant.")
    
    if z_score > 0:
        print("foreign secs have a higher proportion.")
    else:
        print("All Mps a higher proportion.")
    
    print()


# In[8]:


##THIS CODE IS FOR COMPARING LABOUR AND CONSERVATIVE FORIEGN SECRETARIES FOR THE WORD2VEC MODEL FOR MORAL SENTIMENT CATEGORIES

# Calculate the total number of texts for conservative and labour fs
total_texts_con = len(filtered_sentences_df1[filtered_sentences_df1['party2'] == 'Con'])
total_texts_lab = len(filtered_sentences_df1[filtered_sentences_df1['party2'] == 'Lab'])

proportions_conservative = {0: 0.0, 1: 0.0, 2: 0.0}
proportions_labour = {0: 0.0, 1: 0.0, 2: 0.0}

for foundation in [0, 1, 2]:
    foundation_texts_con = filtered_sentences_df1[(filtered_sentences_df1['moral_embeddings'] == foundation) & (filtered_sentences_df1['party2'] == 'Con')]
    foundation_texts_lab = filtered_sentences_df1[(filtered_sentences_df1['moral_embeddings'] == foundation) & (filtered_sentences_df1['party2'] == 'Lab')]

    proportion_con = len(foundation_texts_con) / total_texts_con if total_texts_con > 0 else 0.0
    proportion_lab = len(foundation_texts_lab) / total_texts_lab if total_texts_lab > 0 else 0.0

    proportions_conservative[foundation] = proportion_con
    proportions_labour[foundation] = proportion_lab

# here i am printing the proportions and confidence intervals for each moral sent category for conservative and labour foreign secs
for foundation in [0, 1, 2]:
    print(f"Moral Foundation {foundation}")
    print(f"Proportion of Conservative Texts: {proportions_conservative[foundation]}")
    print(f"Proportion of Labour Texts: {proportions_labour[foundation]}")

    # standard errors for the proportions
    se_con = np.sqrt(proportions_conservative[foundation] * (1 - proportions_conservative[foundation]) / total_texts_con)
    se_lab = np.sqrt(proportions_labour[foundation] * (1 - proportions_labour[foundation]) / total_texts_lab)

    # confidence intervals
    ci_con = (proportions_conservative[foundation] - 1.96 * se_con, proportions_conservative[foundation] + 1.96 * se_con)
    ci_lab = (proportions_labour[foundation] - 1.96 * se_lab, proportions_labour[foundation] + 1.96 * se_lab)

    print(f"Confidence Interval (Conservative): {ci_con}")
    print(f"Confidence Interval (Labour): {ci_lab}")
    print()
    
    
import scipy.stats as stats

# Setting the significance level
alpha = 0.05

# Here I am performing the z-test for proportions for each moral sentiment category
for foundation in [0, 1, 2]:
    p1 = proportions_labour[foundation]
    p2 = proportions_conservative[foundation]
    n1 = total_texts_lab
    n2 = total_texts_con
    
    pooled_proportion = (p1 * n1 + p2 * n2) / (n1 + n2)
    standard_error = np.sqrt(pooled_proportion * (1 - pooled_proportion) * (1/n1 + 1/n2))
    z_score = (p1 - p2) / standard_error
    
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed test has been selected here.
    
    print(f"Moral Foundation {foundation}")
    print(f"Z-Score: {z_score}")
    print(f"P-Value: {p_value}")
    
    if p_value < alpha:
        print("Difference is statistically significant.")
    else:
        print("Difference is not statistically significant.")
    
    if z_score > 0:
        print("Labour has a higher proportion.")
    else:
        print("Conservative has a higher proportion.")
    
    print()


# In[44]:


##THIS CODE IS FOR COMPARING LABOUR AND CONSERVATIVE FORIEGN SECRETARIES FOR THE CUSTOMISED DIC MODEL FOR MORAL SENTIMENT CATEGORIES

# Here I am calculating the total number of texts for conservative and labour fs
total_texts_con = len(filtered_sentences_df1[filtered_sentences_df1['party2'] == 'Con'])
total_texts_lab = len(filtered_sentences_df1[filtered_sentences_df1['party2'] == 'Lab'])

proportions_conservative = {0: 0.0, 1: 0.0, 2: 0.0}
proportions_labour = {0: 0.0, 1: 0.0, 2: 0.0}

for foundation in [0, 1, 2]:
    foundation_texts_con = filtered_sentences_df1[(filtered_sentences_df1['moral_sentiment'] == foundation) & (filtered_sentences_df1['party2'] == 'Con')]
    foundation_texts_lab = filtered_sentences_df1[(filtered_sentences_df1['moral_sentiment'] == foundation) & (filtered_sentences_df1['party2'] == 'Lab')]

    proportion_con = len(foundation_texts_con) / total_texts_con if total_texts_con > 0 else 0.0
    proportion_lab = len(foundation_texts_lab) / total_texts_lab if total_texts_lab > 0 else 0.0

    proportions_conservative[foundation] = proportion_con
    proportions_labour[foundation] = proportion_lab

# This code calculates the proportions and confidence intervals for each moral sent category for conservative and labour foreign secs
for foundation in [0, 1, 2]:
    print(f"Moral Foundation {foundation}")
    print(f"Proportion of Conservative Texts: {proportions_conservative[foundation]}")
    print(f"Proportion of Labour Texts: {proportions_labour[foundation]}")

    # Calculate standard errors for the proportions
    se_con = np.sqrt(proportions_conservative[foundation] * (1 - proportions_conservative[foundation]) / total_texts_con)
    se_lab = np.sqrt(proportions_labour[foundation] * (1 - proportions_labour[foundation]) / total_texts_lab)

    # Calculate and print confidence intervals
    ci_con = (proportions_conservative[foundation] - 1.96 * se_con, proportions_conservative[foundation] + 1.96 * se_con)
    ci_lab = (proportions_labour[foundation] - 1.96 * se_lab, proportions_labour[foundation] + 1.96 * se_lab)

    print(f"Confidence Interval (Conservative): {ci_con}")
    print(f"Confidence Interval (Labour): {ci_lab}")
    print()
    
    
import scipy.stats as stats

# Setting the significance level
alpha = 0.05

# Here I am performing the z-test for proportions for each moral sentiment category
for foundation in [0, 1, 2]:
    p1 = proportions_labour[foundation]
    p2 = proportions_conservative[foundation]
    n1 = total_texts_lab
    n2 = total_texts_con
    
    pooled_proportion = (p1 * n1 + p2 * n2) / (n1 + n2)
    standard_error = np.sqrt(pooled_proportion * (1 - pooled_proportion) * (1/n1 + 1/n2))
    z_score = (p1 - p2) / standard_error
    
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed test is selected here.
    
    print(f"Moral Foundation {foundation}")
    print(f"Z-Score: {z_score}")
    print(f"P-Value: {p_value}")
    
    if p_value < alpha:
        print("Difference is statistically significant.")
    else:
        print("Difference is not statistically significant.")
    
    if z_score > 0:
        print("Labour has a higher proportion.")
    else:
        print("Conservative has a higher proportion.")
    


# In[45]:


##THIS CODE IS FOR COMPARING LABOUR AND CONSERVATIVE FORIEGN SECRETARIES FOR THE BERT MODEL FOR MORAL SENTIMENT CATEGORIES

# here i am finding out the total number of texts for conservative and labour fs
total_texts_con = len(filtered_sentences_df1[filtered_sentences_df1['party2'] == 'Con'])
total_texts_lab = len(filtered_sentences_df1[filtered_sentences_df1['party2'] == 'Lab'])

proportions_conservative = {0: 0.0, 1: 0.0, 2: 0.0}
proportions_labour = {0: 0.0, 1: 0.0, 2: 0.0}

for foundation in [0, 1, 2]:
    foundation_texts_con = filtered_sentences_df1[(filtered_sentences_df1['predicted_labels2'] == foundation) & (filtered_sentences_df1['party2'] == 'Con')]
    foundation_texts_lab = filtered_sentences_df1[(filtered_sentences_df1['predicted_labels2'] == foundation) & (filtered_sentences_df1['party2'] == 'Lab')]

    proportion_con = len(foundation_texts_con) / total_texts_con if total_texts_con > 0 else 0.0
    proportion_lab = len(foundation_texts_lab) / total_texts_lab if total_texts_lab > 0 else 0.0

    proportions_conservative[foundation] = proportion_con
    proportions_labour[foundation] = proportion_lab

#  proportions and confidence intervals for each moral sent category for conservative and labour foreign secs calculated here.
for foundation in [0, 1, 2]:
    print(f"Moral Foundation {foundation}")
    print(f"Proportion of Conservative Texts: {proportions_conservative[foundation]}")
    print(f"Proportion of Labour Texts: {proportions_labour[foundation]}")

    # standard errors
    se_con = np.sqrt(proportions_conservative[foundation] * (1 - proportions_conservative[foundation]) / total_texts_con)
    se_lab = np.sqrt(proportions_labour[foundation] * (1 - proportions_labour[foundation]) / total_texts_lab)

    # confidence intervals
    ci_con = (proportions_conservative[foundation] - 1.96 * se_con, proportions_conservative[foundation] + 1.96 * se_con)
    ci_lab = (proportions_labour[foundation] - 1.96 * se_lab, proportions_labour[foundation] + 1.96 * se_lab)

    print(f"Confidence Interval (Conservative): {ci_con}")
    print(f"Confidence Interval (Labour): {ci_lab}")
    print()
    
    
import scipy.stats as stats

# Setting the significance level
alpha = 0.05

# Here I am performing the z-test for proportions for each moral sentiment category
for foundation in [0, 1, 2]:
    p1 = proportions_labour[foundation]
    p2 = proportions_conservative[foundation]
    n1 = total_texts_lab
    n2 = total_texts_con
    
    pooled_proportion = (p1 * n1 + p2 * n2) / (n1 + n2)
    standard_error = np.sqrt(pooled_proportion * (1 - pooled_proportion) * (1/n1 + 1/n2))
    z_score = (p1 - p2) / standard_error
    
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed test is selected here. 
    
    print(f"Moral Foundation {foundation}")
    print(f"Z-Score: {z_score}")
    print(f"P-Value: {p_value}")
    
    if p_value < alpha:
        print("Difference is statistically significant.")
    else:
        print("Difference is not statistically significant.")
    
    if z_score > 0:
        print("Labour has a higher proportion.")
    else:
        print("Conservative has a higher proportion.")
    


# In[46]:


##THIS CODE IS FOR COMPARING LABOUR AND CONSERVATIVE FORIEGN SECRETARIES FOR THE BERT MODEL FOR MORAL FOUNDATION CATEGORIES

# Here I am findong the total number of texts for conservative and labour fs
total_texts_con = len(filtered_sentences_df1[filtered_sentences_df1['party2'] == 'Con'])
total_texts_lab = len(filtered_sentences_df1[filtered_sentences_df1['party2'] == 'Lab'])

proportions_conservative = { }
proportions_labour = { }

for foundation in [0, 1, 4, 6]:
    foundation_texts_con = filtered_sentences_df1[(filtered_sentences_df1['predicted_labels'] == foundation) & (filtered_sentences_df1['party2'] == 'Con')]
    foundation_texts_lab = filtered_sentences_df1[(filtered_sentences_df1['predicted_labels'] == foundation) & (filtered_sentences_df1['party2'] == 'Lab')]

    proportion_con = len(foundation_texts_con) / total_texts_con if total_texts_con > 0 else 0.0
    proportion_lab = len(foundation_texts_lab) / total_texts_lab if total_texts_lab > 0 else 0.0

    proportions_conservative[foundation] = proportion_con
    proportions_labour[foundation] = proportion_lab

# Here I am printing the proportions and confidence intervals for each moral foundation category for conservative and labour foreign secs
for foundation in [0, 1, 4,6]:
    print(f"Moral Foundation {foundation}")
    print(f"Proportion of Conservative Texts: {proportions_conservative[foundation]}")
    print(f"Proportion of Labour Texts: {proportions_labour[foundation]}")

    # standard errors 
    se_con = np.sqrt(proportions_conservative[foundation] * (1 - proportions_conservative[foundation]) / total_texts_con)
    se_lab = np.sqrt(proportions_labour[foundation] * (1 - proportions_labour[foundation]) / total_texts_lab)

    # confidence intervals
    ci_con = (proportions_conservative[foundation] - 1.96 * se_con, proportions_conservative[foundation] + 1.96 * se_con)
    ci_lab = (proportions_labour[foundation] - 1.96 * se_lab, proportions_labour[foundation] + 1.96 * se_lab)

    print(f"Confidence Interval (Conservative): {ci_con}")
    print(f"Confidence Interval (Labour): {ci_lab}")
    print()

# Setting the significance level
alpha = 0.05

# Here I am performing the z-test for proportions for each moral foundation category
for foundation in [0, 1, 4, 6]:
    p1 = proportions_labour[foundation]
    p2 = proportions_conservative[foundation]
    n1 = total_texts_lab
    n2 = total_texts_con
    
    pooled_proportion = (p1 * n1 + p2 * n2) / (n1 + n2)
    standard_error = np.sqrt(pooled_proportion * (1 - pooled_proportion) * (1/n1 + 1/n2))
    z_score = (p1 - p2) / standard_error
    
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed test is selected here.
    
    print(f"Moral Foundation {foundation}")
    print(f"Z-Score: {z_score}")
    print(f"P-Value: {p_value}")
    
    if p_value < alpha:
        print("Difference is statistically significant.")
    else:
        print("Difference is not statistically significant.")
    
    if z_score > 0:
        print("Labour has a higher proportion.")
    else:
        print("Conservative has a higher proportion.")


# In[47]:


##THIS CODE IS FOR COMPARING LABOUR AND CONSERVATIVE FORIEGN SECRETARIES FOR THE Dictionary MODEL FOR MORAL FOUNDATION CATEGORIES

# Here I am finding out the total number of texts for conservative and labour for. secs.
total_texts_con = len(filtered_sentences_df1[filtered_sentences_df1['party2'] == 'Con'])
total_texts_lab = len(filtered_sentences_df1[filtered_sentences_df1['party2'] == 'Lab'])

proportions_conservative = { }
proportions_labour = { }

for foundation in [0, 1, 4, 6]:
    foundation_texts_con = filtered_sentences_df1[(filtered_sentences_df1['recoded'] == foundation) & (filtered_sentences_df1['party2'] == 'Con')]
    foundation_texts_lab = filtered_sentences_df1[(filtered_sentences_df1['recoded'] == foundation) & (filtered_sentences_df1['party2'] == 'Lab')]

    proportion_con = len(foundation_texts_con) / total_texts_con if total_texts_con > 0 else 0.0
    proportion_lab = len(foundation_texts_lab) / total_texts_lab if total_texts_lab > 0 else 0.0

    proportions_conservative[foundation] = proportion_con
    proportions_labour[foundation] = proportion_lab

# Here I am finding and printing the proportions and confidence intervals for each moral foundation category for conservative and labour foreign secs
for foundation in [0, 1, 4,6]:
    print(f"Moral Foundation {foundation}")
    print(f"Proportion of Conservative Texts: {proportions_conservative[foundation]}")
    print(f"Proportion of Labour Texts: {proportions_labour[foundation]}")

    # standard error
    se_con = np.sqrt(proportions_conservative[foundation] * (1 - proportions_conservative[foundation]) / total_texts_con)
    se_lab = np.sqrt(proportions_labour[foundation] * (1 - proportions_labour[foundation]) / total_texts_lab)

    # confidence intervals
    ci_con = (proportions_conservative[foundation] - 1.96 * se_con, proportions_conservative[foundation] + 1.96 * se_con)
    ci_lab = (proportions_labour[foundation] - 1.96 * se_lab, proportions_labour[foundation] + 1.96 * se_lab)

    print(f"Confidence Interval (Conservative): {ci_con}")
    print(f"Confidence Interval (Labour): {ci_lab}")
    print()
    

# Setting the significance level
alpha = 0.05

# Here I am performing the z-test for proportions for each moral foundation category
for foundation in [0, 1, 4, 6]:
    p1 = proportions_labour[foundation]
    p2 = proportions_conservative[foundation]
    n1 = total_texts_lab
    n2 = total_texts_con
    
    pooled_proportion = (p1 * n1 + p2 * n2) / (n1 + n2)
    standard_error = np.sqrt(pooled_proportion * (1 - pooled_proportion) * (1/n1 + 1/n2))
    z_score = (p1 - p2) / standard_error
    
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed test
    
    print(f"Moral Foundation {foundation}")
    print(f"Z-Score: {z_score}")
    print(f"P-Value: {p_value}")
    
    if p_value < alpha:
        print("Difference is statistically significant.")
    else:
        print("Difference is not statistically significant.")
    
    if z_score > 0:
        print("Labour has a higher proportion.")
    else:
        print("Conservative has a higher proportion.")


# In[49]:


#THIS CODE IS FOR COMPARING LABOUR AND CONSERVATIVE FORIEGN SECRETARIES FOR THE WORD2VEC MODEL FOR MORAL FOUNDATION CATEGORIES

# Here I am finding the total number of texts for conservative and labour for secs.
total_texts_con = len(filtered_sentences_df1[filtered_sentences_df1['party2'] == 'Con'])
total_texts_lab = len(filtered_sentences_df1[filtered_sentences_df1['party2'] == 'Lab'])

proportions_conservative = { }
proportions_labour = { }

for foundation in [0, 1, 4, 6]:
    foundation_texts_con = filtered_sentences_df1[(filtered_sentences_df1['moral_embeddings2'] == foundation) & (filtered_sentences_df1['party2'] == 'Con')]
    foundation_texts_lab = filtered_sentences_df1[(filtered_sentences_df1['moral_embeddings2'] == foundation) & (filtered_sentences_df1['party2'] == 'Lab')]

    proportion_con = len(foundation_texts_con) / total_texts_con if total_texts_con > 0 else 0.0
    proportion_lab = len(foundation_texts_lab) / total_texts_lab if total_texts_lab > 0 else 0.0

    proportions_conservative[foundation] = proportion_con
    proportions_labour[foundation] = proportion_lab

# here I am finding the proportions and confidence intervals for each moral foundation category for conservative and labour foreign secs
for foundation in [0, 1, 4,6]:
    print(f"Moral Foundation {foundation}")
    print(f"Proportion of Conservative Texts: {proportions_conservative[foundation]}")
    print(f"Proportion of Labour Texts: {proportions_labour[foundation]}")

    # standard errors
    se_con = np.sqrt(proportions_conservative[foundation] * (1 - proportions_conservative[foundation]) / total_texts_con)
    se_lab = np.sqrt(proportions_labour[foundation] * (1 - proportions_labour[foundation]) / total_texts_lab)

    # confidence intervals
    ci_con = (proportions_conservative[foundation] - 1.96 * se_con, proportions_conservative[foundation] + 1.96 * se_con)
    ci_lab = (proportions_labour[foundation] - 1.96 * se_lab, proportions_labour[foundation] + 1.96 * se_lab)

    print(f"Confidence Interval (Conservative): {ci_con}")
    print(f"Confidence Interval (Labour): {ci_lab}")
    print()
    

alpha = 0.05

# Here i am performing the z-test for proportions for each moral foundation category
for foundation in [0, 1, 4, 6]:
    p1 = proportions_labour[foundation]
    p2 = proportions_conservative[foundation]
    n1 = total_texts_lab
    n2 = total_texts_con
    
    pooled_proportion = (p1 * n1 + p2 * n2) / (n1 + n2)
    standard_error = np.sqrt(pooled_proportion * (1 - pooled_proportion) * (1/n1 + 1/n2))
    z_score = (p1 - p2) / standard_error
    
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed test is selected here. 
    
    print(f"Moral Foundation {foundation}")
    print(f"Z-Score: {z_score}")
    print(f"P-Value: {p_value}")
    
    if p_value < alpha:
        print("Difference is statistically significant.")
    else:
        print("Difference is not statistically significant.")
    
    if z_score > 0:
        print("Labour has a higher proportion.")
    else:
        print("Conservative has a higher proportion.")


# In[9]:


#CODE FOR COMPARING lAB AND CON MPS ACROSS THE FULL DATASET FOR MORAL SENTIMENT FOR THE DISTILBERT MODEL 

# Here I am calculating the total number of texts for conservative and labour MPs
total_texts_con = len(df2[df2['party2'] == 'Con'])
total_texts_lab = len(df2[df2['party2'] == 'Lab'])

proportions_conservative = {0: 0.0, 1: 0.0, 2: 0.0}
proportions_labour = {0: 0.0, 1: 0.0, 2: 0.0}

for foundation in [0, 1, 2]:
    foundation_texts_con = df2[(df2['predicted_labels2'] == foundation) & (df2['party2'] == 'Con')]
    foundation_texts_lab = df2[(df2['predicted_labels2'] == foundation) & (df2['party2'] == 'Lab')]

    proportion_con = len(foundation_texts_con) / total_texts_con if total_texts_con > 0 else 0.0
    proportion_lab = len(foundation_texts_lab) / total_texts_lab if total_texts_lab > 0 else 0.0

    proportions_conservative[foundation] = proportion_con
    proportions_labour[foundation] = proportion_lab

# Printing the proportions and confidence intervals for each moral sent for conservative and labour MPs
for foundation in [0, 1, 2]:
    print(f"Moral Foundation {foundation}")
    print(f"Proportion of Conservative Texts: {proportions_conservative[foundation]}")
    print(f"Proportion of Labour Texts: {proportions_labour[foundation]}")

    # standard errors 
    se_con = np.sqrt(proportions_conservative[foundation] * (1 - proportions_conservative[foundation]) / total_texts_con)
    se_lab = np.sqrt(proportions_labour[foundation] * (1 - proportions_labour[foundation]) / total_texts_lab)

    # Confidence intervals
    ci_con = (proportions_conservative[foundation] - 1.96 * se_con, proportions_conservative[foundation] + 1.96 * se_con)
    ci_lab = (proportions_labour[foundation] - 1.96 * se_lab, proportions_labour[foundation] + 1.96 * se_lab)

    print(f"Confidence Interval (Conservative): {ci_con}")
    print(f"Confidence Interval (Labour): {ci_lab}")
    print()
    
    
import scipy.stats as stats

# Setting the significance level
alpha = 0.05

# Here I am performing z-test for proportions for each moral sent.
for foundation in [0, 1, 2]:
    p1 = proportions_labour[foundation]
    p2 = proportions_conservative[foundation]
    n1 = total_texts_lab
    n2 = total_texts_con
    
    pooled_proportion = (p1 * n1 + p2 * n2) / (n1 + n2)
    standard_error = np.sqrt(pooled_proportion * (1 - pooled_proportion) * (1/n1 + 1/n2))
    z_score = (p1 - p2) / standard_error
    
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Here I am using a Two-tailed test
    
    print(f"Moral Foundation {foundation}")
    print(f"Z-Score: {z_score}")
    print(f"P-Value: {p_value}")
    
    if p_value < alpha:
        print("Difference is statistically significant.")
    else:
        print("Difference is not statistically significant.")
    
    if z_score > 0:
        print("Labour has a higher proportion.")
    else:
        print("Conservative has a higher proportion.")
    
    print()


# In[51]:


#CODE FOR COMPARING lAB AND CON MPS ACROSS THE FULL DATASET FOR MORAL SENTIMENT FOR THE dictionary MODEL 

# Here I am calculating the total number of texts for conservative and labour MPs
total_texts_con = len(df2[df2['party2'] == 'Con'])
total_texts_lab = len(df2[df2['party2'] == 'Lab'])

proportions_conservative = {0: 0.0, 1: 0.0, 2: 0.0}
proportions_labour = {0: 0.0, 1: 0.0, 2: 0.0}

for foundation in [0, 1, 2]:
    foundation_texts_con = df2[(df2['moral_sentiment'] == foundation) & (df2['party2'] == 'Con')]
    foundation_texts_lab = df2[(df2['moral_sentiment'] == foundation) & (df2['party2'] == 'Lab')]

    proportion_con = len(foundation_texts_con) / total_texts_con if total_texts_con > 0 else 0.0
    proportion_lab = len(foundation_texts_lab) / total_texts_lab if total_texts_lab > 0 else 0.0

    proportions_conservative[foundation] = proportion_con
    proportions_labour[foundation] = proportion_lab

# Finding the proportions and confidence intervals for each moral sent for conservative and labour MPs
for foundation in [0, 1, 2]:
    print(f"Moral Foundation {foundation}")
    print(f"Proportion of Conservative Texts: {proportions_conservative[foundation]}")
    print(f"Proportion of Labour Texts: {proportions_labour[foundation]}")

    # standard errors 
    se_con = np.sqrt(proportions_conservative[foundation] * (1 - proportions_conservative[foundation]) / total_texts_con)
    se_lab = np.sqrt(proportions_labour[foundation] * (1 - proportions_labour[foundation]) / total_texts_lab)

    # confidence intervals
    ci_con = (proportions_conservative[foundation] - 1.96 * se_con, proportions_conservative[foundation] + 1.96 * se_con)
    ci_lab = (proportions_labour[foundation] - 1.96 * se_lab, proportions_labour[foundation] + 1.96 * se_lab)

    print(f"Confidence Interval (Conservative): {ci_con}")
    print(f"Confidence Interval (Labour): {ci_lab}")
    print()
    

# Setting the significance level
alpha = 0.05

# Performing z-test for proportions for each moral sent.
for foundation in [0, 1, 2]:
    p1 = proportions_labour[foundation]
    p2 = proportions_conservative[foundation]
    n1 = total_texts_lab
    n2 = total_texts_con
    
    pooled_proportion = (p1 * n1 + p2 * n2) / (n1 + n2)
    standard_error = np.sqrt(pooled_proportion * (1 - pooled_proportion) * (1/n1 + 1/n2))
    z_score = (p1 - p2) / standard_error
    
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Here I am using a Two-tailed test
    
    print(f"Moral Foundation {foundation}")
    print(f"Z-Score: {z_score}")
    print(f"P-Value: {p_value}")
    
    if p_value < alpha:
        print("Difference is statistically significant.")
    else:
        print("Difference is not statistically significant.")
    
    if z_score > 0:
        print("Labour has a higher proportion.")
    else:
        print("Conservative has a higher proportion.")
    
    print()


# In[53]:


#CODE FOR COMPARING lAB AND CON MPS ACROSS THE FULL DATASET FOR MORAL SENTIMENT FOR THE word2vec MODEL 

# Here I am calculating the total number of texts for conservative and labour MPs
total_texts_con = len(df2[df2['party2'] == 'Con'])
total_texts_lab = len(df2[df2['party2'] == 'Lab'])

proportions_conservative = {0: 0.0, 1: 0.0, 2: 0.0}
proportions_labour = {0: 0.0, 1: 0.0, 2: 0.0}

for foundation in [0, 1, 2]:
    foundation_texts_con = df3[(df3['moral_embeddings'] == foundation) & (df2['party2'] == 'Con')]
    foundation_texts_lab = df3[(df3['moral_embeddings'] == foundation) & (df2['party2'] == 'Lab')]

    proportion_con = len(foundation_texts_con) / total_texts_con if total_texts_con > 0 else 0.0
    proportion_lab = len(foundation_texts_lab) / total_texts_lab if total_texts_lab > 0 else 0.0

    proportions_conservative[foundation] = proportion_con
    proportions_labour[foundation] = proportion_lab

# here I am finding the proportions and confidence intervals for each moral sent for conservative and labour MPs
for foundation in [0, 1, 2]:
    print(f"Moral Foundation {foundation}")
    print(f"Proportion of Conservative Texts: {proportions_conservative[foundation]}")
    print(f"Proportion of Labour Texts: {proportions_labour[foundation]}")

    # standard errors 
    se_con = np.sqrt(proportions_conservative[foundation] * (1 - proportions_conservative[foundation]) / total_texts_con)
    se_lab = np.sqrt(proportions_labour[foundation] * (1 - proportions_labour[foundation]) / total_texts_lab)

    # confidence intervals
    ci_con = (proportions_conservative[foundation] - 1.96 * se_con, proportions_conservative[foundation] + 1.96 * se_con)
    ci_lab = (proportions_labour[foundation] - 1.96 * se_lab, proportions_labour[foundation] + 1.96 * se_lab)

    print(f"Confidence Interval (Conservative): {ci_con}")
    print(f"Confidence Interval (Labour): {ci_lab}")
    print()

# Setting the significance level
alpha = 0.05

# Performing z-test for proportions for each moral sent.
for foundation in [0, 1, 2]:
    p1 = proportions_labour[foundation]
    p2 = proportions_conservative[foundation]
    n1 = total_texts_lab
    n2 = total_texts_con
    
    pooled_proportion = (p1 * n1 + p2 * n2) / (n1 + n2)
    standard_error = np.sqrt(pooled_proportion * (1 - pooled_proportion) * (1/n1 + 1/n2))
    z_score = (p1 - p2) / standard_error
    
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Here I am using a Two-tailed test
    
    print(f"Moral Foundation {foundation}")
    print(f"Z-Score: {z_score}")
    print(f"P-Value: {p_value}")
    
    if p_value < alpha:
        print("Difference is statistically significant.")
    else:
        print("Difference is not statistically significant.")
    
    if z_score > 0:
        print("Labour has a higher proportion.")
    else:
        print("Conservative has a higher proportion.")
    
    print()


# In[13]:


#CODE FOR COMPARING LAB AND CON MPS ACROSS THE FULL DATASET FOR MORAL FOUNDATION FOR THE BERT MODEL

# Here I am calculating the total number of texts for conservative and labour MPs
total_texts_con = len(df2[df2['party2'] == 'Con'])
total_texts_lab = len(df2[df2['party2'] == 'Lab'])

proportions_conservative = {}
proportions_labour = {}

for foundation in [0, 1, 4,6]:
    foundation_texts_con = df2[(df2['predicted_labels'] == foundation) & (df2['party2'] == 'Con')]
    foundation_texts_lab = df2[(df2['predicted_labels'] == foundation) & (df2['party2'] == 'Lab')]

    proportion_con = len(foundation_texts_con) / total_texts_con if total_texts_con > 0 else 0.0
    proportion_lab = len(foundation_texts_lab) / total_texts_lab if total_texts_lab > 0 else 0.0

    proportions_conservative[foundation] = proportion_con
    proportions_labour[foundation] = proportion_lab

# Here I am finding the proportions and confidence intervals for each moral foundation for conservative and labour parties
for foundation in [0, 1, 4,6]:
    print(f"Moral Foundation {foundation}")
    print(f"Proportion of Conservative Texts: {proportions_conservative[foundation]}")
    print(f"Proportion of Labour Texts: {proportions_labour[foundation]}")

    # standard errors 
    se_con = np.sqrt(proportions_conservative[foundation] * (1 - proportions_conservative[foundation]) / total_texts_con)
    se_lab = np.sqrt(proportions_labour[foundation] * (1 - proportions_labour[foundation]) / total_texts_lab)

    # confidence intervals
    ci_con = (proportions_conservative[foundation] - 1.96 * se_con, proportions_conservative[foundation] + 1.96 * se_con)
    ci_lab = (proportions_labour[foundation] - 1.96 * se_lab, proportions_labour[foundation] + 1.96 * se_lab)

    print(f"Confidence Interval (Conservative): {ci_con}")
    print(f"Confidence Interval (Labour): {ci_lab}")
    print()


# Setting the significance level
alpha = 0.05

# Here I am performing the z-test for proportions for each moral sent caterory
for foundation in [0, 1, 4, 6]:
    p1 = proportions_labour[foundation]
    p2 = proportions_conservative[foundation]
    n1 = total_texts_lab
    n2 = total_texts_con
    
    pooled_proportion = (p1 * n1 + p2 * n2) / (n1 + n2)
    standard_error = np.sqrt(pooled_proportion * (1 - pooled_proportion) * (1/n1 + 1/n2))
    z_score = (p1 - p2) / standard_error
    
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Here I am using a two-tailed test
    
    print(f"Moral Foundation {foundation}")
    print(f"Z-Score: {z_score}")
    print(f"P-Value: {p_value}")
    
    if p_value < alpha:
        print("Difference is statistically significant.")
    else:
        print("Difference is not statistically significant.")
    
    if z_score > 0:
        print("Labour has a higher proportion.")
    else:
        print("Conservative has a higher proportion.")
    
    print()


# In[54]:


#CODE FOR COMPARING LAB AND CON MPS ACROSS THE FULL DATASET FOR MORAL foundations FOR THE dictionary MODEL

# Here i am calculating the total number of texts for conservative and labour MPs
total_texts_con = len(df2[df2['party2'] == 'Con'])
total_texts_lab = len(df2[df2['party2'] == 'Lab'])

proportions_conservative = {}
proportions_labour = {}

for foundation in [0, 1, 4,6]:
    foundation_texts_con = df2[(df2['recoded'] == foundation) & (df2['party2'] == 'Con')]
    foundation_texts_lab = df2[(df2['recoded'] == foundation) & (df2['party2'] == 'Lab')]

    proportion_con = len(foundation_texts_con) / total_texts_con if total_texts_con > 0 else 0.0
    proportion_lab = len(foundation_texts_lab) / total_texts_lab if total_texts_lab > 0 else 0.0

    proportions_conservative[foundation] = proportion_con
    proportions_labour[foundation] = proportion_lab

# Here i am finding the proportions and confidence intervals for each moral foundation for conservative and labour parties
for foundation in [0, 1, 4,6]:
    print(f"Moral Foundation {foundation}")
    print(f"Proportion of Conservative Texts: {proportions_conservative[foundation]}")
    print(f"Proportion of Labour Texts: {proportions_labour[foundation]}")

    # Calculating standard errors 
    se_con = np.sqrt(proportions_conservative[foundation] * (1 - proportions_conservative[foundation]) / total_texts_con)
    se_lab = np.sqrt(proportions_labour[foundation] * (1 - proportions_labour[foundation]) / total_texts_lab)

    # Calculating confidence intervals
    ci_con = (proportions_conservative[foundation] - 1.96 * se_con, proportions_conservative[foundation] + 1.96 * se_con)
    ci_lab = (proportions_labour[foundation] - 1.96 * se_lab, proportions_labour[foundation] + 1.96 * se_lab)

    print(f"Confidence Interval (Conservative): {ci_con}")
    print(f"Confidence Interval (Labour): {ci_lab}")
    print()


# Setting the significance level
alpha = 0.05

# Here i am performing the z-test for proportions for each moral sent caterory
for foundation in [0, 1, 4, 6]:
    p1 = proportions_labour[foundation]
    p2 = proportions_conservative[foundation]
    n1 = total_texts_lab
    n2 = total_texts_con
    
    pooled_proportion = (p1 * n1 + p2 * n2) / (n1 + n2)
    standard_error = np.sqrt(pooled_proportion * (1 - pooled_proportion) * (1/n1 + 1/n2))
    z_score = (p1 - p2) / standard_error
    
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Here I am using a two-tailed test
    
    print(f"Moral Foundation {foundation}")
    print(f"Z-Score: {z_score}")
    print(f"P-Value: {p_value}")
    
    if p_value < alpha:
        print("Difference is statistically significant.")
    else:
        print("Difference is not statistically significant.")
    
    if z_score > 0:
        print("Labour has a higher proportion.")
    else:
        print("Conservative has a higher proportion.")
    
    print()


# In[55]:


#CODE FOR COMPARING LAB AND CON MPS ACROSS THE FULL DATASET FOR MORAL FOUNDATION FOR THE word2vec MODEL

# Here I am calculating the total number of texts for conservative and labour MPs
total_texts_con = len(df2[df2['party2'] == 'Con'])
total_texts_lab = len(df2[df2['party2'] == 'Lab'])

proportions_conservative = {}
proportions_labour = {}

for foundation in [0, 1, 4,6]:
    foundation_texts_con = df3[(df3['moral_embeddings2'] == foundation) & (df2['party2'] == 'Con')]
    foundation_texts_lab = df3[(df3['moral_embeddings2'] == foundation) & (df2['party2'] == 'Lab')]

    proportion_con = len(foundation_texts_con) / total_texts_con if total_texts_con > 0 else 0.0
    proportion_lab = len(foundation_texts_lab) / total_texts_lab if total_texts_lab > 0 else 0.0

    proportions_conservative[foundation] = proportion_con
    proportions_labour[foundation] = proportion_lab

# here I am finding the proportions and confidence intervals for each moral foundation for conservative and labour parties
for foundation in [0, 1, 4,6]:
    print(f"Moral Foundation {foundation}")
    print(f"Proportion of Conservative Texts: {proportions_conservative[foundation]}")
    print(f"Proportion of Labour Texts: {proportions_labour[foundation]}")

    # standard errors 
    se_con = np.sqrt(proportions_conservative[foundation] * (1 - proportions_conservative[foundation]) / total_texts_con)
    se_lab = np.sqrt(proportions_labour[foundation] * (1 - proportions_labour[foundation]) / total_texts_lab)

    # confidence intervals
    ci_con = (proportions_conservative[foundation] - 1.96 * se_con, proportions_conservative[foundation] + 1.96 * se_con)
    ci_lab = (proportions_labour[foundation] - 1.96 * se_lab, proportions_labour[foundation] + 1.96 * se_lab)

    print(f"Confidence Interval (Conservative): {ci_con}")
    print(f"Confidence Interval (Labour): {ci_lab}")
    print()


# Setting the significance level
alpha = 0.05

# Here I am performing the z-test for proportions for each moral sent caterory
for foundation in [0, 1, 4, 6]:
    p1 = proportions_labour[foundation]
    p2 = proportions_conservative[foundation]
    n1 = total_texts_lab
    n2 = total_texts_con
    
    pooled_proportion = (p1 * n1 + p2 * n2) / (n1 + n2)
    standard_error = np.sqrt(pooled_proportion * (1 - pooled_proportion) * (1/n1 + 1/n2))
    z_score = (p1 - p2) / standard_error
    
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Here I am using a two-tailed test
    
    print(f"Moral Foundation {foundation}")
    print(f"Z-Score: {z_score}")
    print(f"P-Value: {p_value}")
    
    if p_value < alpha:
        print("Difference is statistically significant.")
    else:
        print("Difference is not statistically significant.")
    
    if z_score > 0:
        print("Labour has a higher proportion.")
    else:
        print("Conservative has a higher proportion.")
    
    print()


# code below is for creating a bar chart for all foreign secs, all mps and lab and con foreign secrtaries for the word2vec modle to create the same bar chart but for different models change the dataframe name and the labels. 
# #for DistilBERT moral sentiment change df3 to df2 and change moral embeddings to predicted_labels2
# #for customised dicitonary model sentiment change df3 to df2 and change moral embedding to moral_sentiment

# In[59]:


# Here I am finding the total number of referencnes in the DataFrame - using sentences is an error carried through from when I thought I might use sentences however the name of the variables does not impact the analysis.
total_sentences = len(df3)

# Here I am finding the proportions of each moral sentiment categrory 
proportions = df3['moral_embeddings'].value_counts(normalize=True).sort_index()

# Here I am finding the proportions for Lab and Con separately
lab_df = filtered_sentences_df1[filtered_sentences_df1['party2'] == 'Lab']
con_df = filtered_sentences_df1[filtered_sentences_df1['party2'] == 'Con']

proportions_lab = lab_df['moral_embeddings'].value_counts(normalize=True).sort_index()
proportions_con = con_df['moral_embeddings'].value_counts(normalize=True).sort_index()
proportions2 = filtered_sentences_df1['moral_embeddings'].value_counts(normalize=True).sort_index()


# This will plto the bar graph for all proportions
plt.figure(figsize=(12, 6))

x = np.arange(len(proportions.index))
width = 0.2 
plt.bar(x - width, proportions, width=width, color='green', label='All MPs')
plt.bar(x, proportions_lab, width=width, color='red', alpha=0.9, label='Lab foreign secs')
plt.bar(x + width, proportions_con, width=width, color='blue', alpha=0.9, label='Con foreign secs')
plt.bar(x + 2*width, proportions2, width=width, color='orange', label='All foreign secs')
plt.xlabel('Predicted Labels')
plt.ylabel('Proportion')
plt.title('Proportions of Predicted Labels by Party- word2vec')
plt.xticks(ticks=x + width/2, labels=proportions.index, rotation=45) 
plt.legend()
plt.tight_layout()

plt.show()

# This code for the plotting the bar chart of the BERT model for moral sentiment categories.

import matplotlib.pyplot as plt
# Here I am calcuating the total number of references in the DataFrame
total_sentences = len(df2)

# Here I am finding the proportions of each moral sentiment category
proportions = df2['predicted_labels2'].value_counts(normalize=True).sort_index()

# here i am finding the proportions for Lab and Con separately
lab_df = filtered_sentences_df1[filtered_sentences_df1['party2'] == 'Lab']
con_df = filtered_sentences_df1[filtered_sentences_df1['party2'] == 'Con']

proportions_lab = lab_df['predicted_labels2'].value_counts(normalize=True).sort_index()
proportions_con = con_df['predicted_labels2'].value_counts(normalize=True).sort_index()
proportions2 = filtered_sentences_df1['predicted_labels2'].value_counts(normalize=True).sort_index()


# The follow section of code plots the the bar graph for all proportions
plt.figure(figsize=(12, 6))

x = np.arange(len(proportions.index))
width = 0.2  
plt.bar(x - width, proportions, width=width, color='green', label='All MPs')
plt.bar(x, proportions_lab, width=width, color='red', alpha=0.9, label='Lab foreign secs')
plt.bar(x + width, proportions_con, width=width, color='blue', alpha=0.9, label='Con foreign secs')
plt.bar(x + 2*width, proportions2, width=width, color='orange', label='All foreign secs')

plt.xlabel('Predicted Labels')
plt.ylabel('Proportion')
plt.title('Proportions of Predicted Labels by Party - distil bert')
plt.xticks(ticks=x + width/2, labels=proportions.index, rotation=45)  
plt.legend()
plt.tight_layout()

plt.show()

#This last bar graph for this section plots the proportions for moral sentiment for the customised dictionary 
total_sentences = len(df2)

#Here I am calculate the proportions of each moral sentiment category. 
proportions = df2['moral_sentiment'].value_counts(normalize=True).sort_index()

# Here I am calculating the proportions for Lab and Con separately
lab_df = filtered_sentences_df1[filtered_sentences_df1['party2'] == 'Lab']
con_df = filtered_sentences_df1[filtered_sentences_df1['party2'] == 'Con']

proportions_lab = lab_df['moral_sentiment'].value_counts(normalize=True).sort_index()
proportions_con = con_df['moral_sentiment'].value_counts(normalize=True).sort_index()
proportions2 = filtered_sentences_df1['moral_sentiment'].value_counts(normalize=True).sort_index()


#The folowing code plots the bar graph - in the same manner as the previous two bar graphs.
plt.figure(figsize=(12, 6))

x = np.arange(len(proportions.index))
width = 0.2  
plt.bar(x - width, proportions, width=width, color='green', label='All MPs')
plt.bar(x, proportions_lab, width=width, color='red', alpha=0.9, label='Lab foreign secs')
plt.bar(x + width, proportions_con, width=width, color='blue', alpha=0.9, label='Con foreign secs')
plt.bar(x + 2*width, proportions2, width=width, color='orange', label='All foreign secs')

plt.xlabel('Predicted Labels')
plt.ylabel('Proportion')
plt.title('Proportions of Predicted Labels by Party -dictionary')
plt.xticks(ticks=x + width/2, labels=proportions.index, rotation=45) 
plt.legend()
plt.tight_layout()

plt.show()


# In[64]:


#code for the ploting the bar chart of the customised dictionary, bert and word2vec for moral foundations.

import matplotlib.pyplot as plt
# HEre I am calculating the total number of references in the dataframe
total_sentences = len(df2)

# Here I am finding the proportions of each moral foundation category for the customised dictionary 
proportions = df2['recoded'].value_counts(normalize=True).sort_index()

# here I am calculating the proportions for Lab and Con separately
lab_df = filtered_sentences_df1[filtered_sentences_df1['party2'] == 'Lab']
con_df = filtered_sentences_df1[filtered_sentences_df1['party2'] == 'Con']

proportions_lab = lab_df['recoded'].value_counts(normalize=True).sort_index()
proportions_con = con_df['recoded'].value_counts(normalize=True).sort_index()
proportions2 = filtered_sentences_df1['recoded'].value_counts(normalize=True).sort_index()


# This code plots the bar graph of proportions for each category. 
plt.figure(figsize=(12, 6))

x = np.arange(len(proportions.index))
width = 0.2  
plt.bar(x - width, proportions, width=width, color='green', label='All MPs')
plt.bar(x, proportions_lab, width=width, color='red', alpha=0.9, label='Lab foreign secs')
plt.bar(x + width, proportions_con, width=width, color='blue', alpha=0.9, label='Con foreign secs')
plt.bar(x + 2*width, proportions2, width=width, color='orange', label='All foreign secs')

plt.xlabel('Predicted Labels')
plt.ylabel('Proportion')
plt.title('Proportions of Predicted Labels by Party -dictionary ')
plt.xticks(ticks=x + width/2, labels=proportions.index, rotation=45)
plt.legend()
plt.tight_layout()

plt.show()

# Here I am calculating the proportions of each category in the 'predicted_label' column for BERT model and moral foundations.
proportions = df2['predicted_labels'].value_counts(normalize=True).sort_index()

# here I am the proportions for Lab and Con separately
lab_df = filtered_sentences_df1[filtered_sentences_df1['party2'] == 'Lab']
con_df = filtered_sentences_df1[filtered_sentences_df1['party2'] == 'Con']

proportions_lab = lab_df['predicted_labels'].value_counts(normalize=True).sort_index()
proportions_con = con_df['predicted_labels'].value_counts(normalize=True).sort_index()
proportions2 = filtered_sentences_df1['predicted_labels'].value_counts(normalize=True).sort_index()


# This code plots the graph for the BERT model.
plt.figure(figsize=(12, 6))

x = np.arange(len(proportions.index))
width = 0.2  
plt.bar(x - width, proportions, width=width, color='green', label='All MPs')
plt.bar(x, proportions_lab, width=width, color='red', alpha=0.9, label='Lab foreign secs')
plt.bar(x + width, proportions_con, width=width, color='blue', alpha=0.9, label='Con foreign secs')
plt.bar(x + 2*width, proportions2, width=width, color='orange', label='All foreign secs')

plt.xlabel('Predicted Labels')
plt.ylabel('Proportion')
plt.title('Proportions of Predicted Labels by Party - distilbert')
plt.xticks(ticks=x + width/2, labels=proportions.index, rotation=45) 
plt.legend()
plt.tight_layout()

plt.show()

proportions3 = df3['moral_embeddings2'].value_counts(normalize=True).sort_index()
# Here Iam finding the proportions for Lab and Con separately
lab_df = filtered_sentences_df1[filtered_sentences_df1['party2'] == 'Lab']
con_df = filtered_sentences_df1[filtered_sentences_df1['party2'] == 'Con']

proportions_lab = lab_df['moral_embeddings2'].value_counts(normalize=True).sort_index()
proportions_con = con_df['moral_embeddings2'].value_counts(normalize=True).sort_index()
proportions4 = filtered_sentences_df1['moral_embeddings2'].value_counts(normalize=True).sort_index()

# As the word embedding model use a different dataframe this is calculated differently, and the 
#labels for the bars need to be reassigned and indexed.
combined_proportions = pd.DataFrame({
    'All MPs': proportions3.reindex(proportions4.index, fill_value=0),
    'Lab foreign secs': proportions_lab.reindex(proportions4.index, fill_value=0),
    'Con foreign secs': proportions_con.reindex(proportions4.index, fill_value=0),
    'All foreign secs': proportions4
})

#now the same code as above can be re-used again and the proprotions for each moral foundation and group are plotted.
plt.figure(figsize=(12, 6))

x = np.arange(len(proportions4.index))
width = 0.2 
for i, (col_name, data) in enumerate(combined_proportions.items()):
    plt.bar(x + i * width, data, width=width, label=col_name)

plt.xlabel('Predicted Labels')
plt.ylabel('Proportion')
plt.title('Proportions of Predicted Labels by Party - word2vec')
plt.xticks(ticks=x + width/2, labels=proportions4.index, rotation=45) 
plt.legend()
plt.tight_layout()

plt.show()


# This section of the code plots all the bar chart for comparing the labout and conservative Mps across the whole dataset, again much of the above code is simply recycled, and a few dataframes and labels changed each time. 

# In[83]:


#This code is for plotting the bar chanrt for the word2vec moral for moral sentiment.

# Here i am finding the total number of texts for conservative and labour MPs
total_texts_con = len(df3[df3['party2'] == 'Con'])
total_texts_lab = len(df3[df3['party2'] == 'Lab'])

proportions_conservative = {0: 0.0, 1: 0.0, 2: 0.0}
proportions_labour = {0: 0.0, 1: 0.0, 2: 0.0}

for foundation in [0, 1, 2]:
    foundation_texts_con = df3[(df3['moral_embeddings'] == foundation) & (df3['party2'] == 'Con')]
    foundation_texts_lab = df3[(df3['moral_embeddings'] == foundation) & (df3['party2'] == 'Lab')]

    proportion_con = len(foundation_texts_con) / total_texts_con if total_texts_con > 0 else 0.0
    proportion_lab = len(foundation_texts_lab) / total_texts_lab if total_texts_lab > 0 else 0.0

    proportions_conservative[foundation] = proportion_con
    proportions_labour[foundation] = proportion_lab

# Here I am printing the proportions for each moral foundation separately for conservative and labour Mps
for foundation in [0, 1, 2]:
    print(f"Moral Foundation {foundation}")
    print(f"Proportion of Conservative Texts: {proportions_conservative[foundation]}")
    print(f"Proportion of Labour Texts: {proportions_labour[foundation]}")

relevant_columns = ['recoded', 'party2']

#Herw I am creating the labels for the bars of the graph
foundation_labels = {
    0: 'non moral',
    1: 'postive',
    2: 'negative',
  
}

# Here I am retreiving the Proportions for 'Lab' and 'Con'
proportions_lab = [proportions_labour[f] for f in [0, 1, 2]]
proportions_con = [proportions_conservative[f] for f in [0, 1, 2]]

# This code plots the bar graph.
fig, ax = plt.subplots(figsize=(10, 6))
width = 0.4
color_lab = 'orange'
color_con = 'blue'

ax.bar([x - width/2 for x in range(len(proportions_lab))], proportions_lab, width, label='Lab', color=color_lab)
ax.bar([x + width/2 for x in range(len(proportions_con))], proportions_con, width, label='Con', color=color_con)
ax.set_ylabel('Proportion')
ax.set_title('Proportions of Moral Sentiment by Party - word2vec')
ax.set_xticks(range(len(foundation_labels)))
ax.set_xticklabels([foundation_labels[f] for f in [0, 1, 2]])
ax.legend()

plt.tight_layout()
plt.show()


# In[82]:


#this code plots the bar graph for moral sentiment for the BERT model. - for comparing all Labour and con MPs


# Here i am finding the total number of texts for conservative and labour MPs
total_texts_con = len(df2[df2['party2'] == 'Con'])
total_texts_lab = len(df2[df2['party2'] == 'Lab'])

proportions_conservative = {0: 0.0, 1: 0.0, 2: 0.0}
proportions_labour = {0: 0.0, 1: 0.0, 2: 0.0}

for foundation in [0, 1, 2]:
    foundation_texts_con = df2[(df2['predicted_labels2'] == foundation) & (df2['party2'] == 'Con')]
    foundation_texts_lab = df2[(df2['predicted_labels2'] == foundation) & (df2['party2'] == 'Lab')]

    proportion_con = len(foundation_texts_con) / total_texts_con if total_texts_con > 0 else 0.0
    proportion_lab = len(foundation_texts_lab) / total_texts_lab if total_texts_lab > 0 else 0.0

    proportions_conservative[foundation] = proportion_con
    proportions_labour[foundation] = proportion_lab

# Here i am finding the proportions for each moral foundation separately for conservative and labour Mps
for foundation in [0, 1, 2]:
    print(f"Moral Foundation {foundation}")
    print(f"Proportion of Conservative Texts: {proportions_conservative[foundation]}")
    print(f"Proportion of Labour Texts: {proportions_labour[foundation]}")

relevant_columns = ['recoded', 'party2']

#Here i am assinging each of the categories a label for the bar chart.
foundation_labels = {
    0: 'non moral',
    1: 'postive',
    2: 'negative',
  
}

# here i am finding/retevieing the proportions for 'Lab' and 'Con' for the bar chart
proportions_lab = [proportions_labour[f] for f in [0, 1, 2]]
proportions_con = [proportions_conservative[f] for f in [0, 1, 2]]

# This code plots the bar graph.
fig, ax = plt.subplots(figsize=(10, 6))
width = 0.4

color_lab = 'orange'
color_con = 'blue'

ax.bar([x - width/2 for x in range(len(proportions_lab))], proportions_lab, width, label='Lab', color=color_lab)
ax.bar([x + width/2 for x in range(len(proportions_con))], proportions_con, width, label='Con', color=color_con)
ax.set_ylabel('Proportion')
ax.set_title('Proportions of Moral Sentiment by Party - BERT')
ax.set_xticks(range(len(foundation_labels)))
ax.set_xticklabels([foundation_labels[f] for f in [0, 1, 2]])
ax.legend()

plt.tight_layout()
plt.show()


# In[65]:


#this code plots the bar graph for moral sentiment for the customised dictionary model. - for comparing all Labour and con MPs


# here I am finding the total number of texts for conservative and labour MPs
total_texts_con = len(df2[df2['party2'] == 'Con'])
total_texts_lab = len(df2[df2['party2'] == 'Lab'])


# this section here sets up the empty proportions for the conservative and labour moral sentiment categories
proportions_conservative = {0: 0.0, 1: 0.0, 2: 0.0}
proportions_labour = {0: 0.0, 1: 0.0, 2: 0.0}


#this loops through each of the proportion labels and calculates the proportions by party.
for foundation in [0, 1, 2]:
    foundation_texts_con = df2[(df2['moral_sentiment'] == foundation) & (df2['party2'] == 'Con')]
    foundation_texts_lab = df2[(df2['moral_sentiment'] == foundation) & (df2['party2'] == 'Lab')]

    proportion_con = len(foundation_texts_con) / total_texts_con if total_texts_con > 0 else 0.0
    proportion_lab = len(foundation_texts_lab) / total_texts_lab if total_texts_lab > 0 else 0.0

    proportions_conservative[foundation] = proportion_con
    proportions_labour[foundation] = proportion_lab

 #Here i am finding the proportions for each moral foundation separately for conservative and labour Mps
for foundation in [0, 1, 2]:
    print(f"Moral Foundation {foundation}")
    print(f"Proportion of Conservative Texts: {proportions_conservative[foundation]}")
    print(f"Proportion of Labour Texts: {proportions_labour[foundation]}")

relevant_columns = ['recoded', 'party2']


foundation_labels = {
    0: 'non moral',
    1: 'postive',
    2: 'negative',
  
}

# here i am finding/retevieing the proportions for 'Lab' and 'Con' for the bar chart
proportions_lab = [proportions_labour[f] for f in [0, 1, 2]]
proportions_con = [proportions_conservative[f] for f in [0, 1, 2]]

#This code plots the bar chart for the moral sentiment categories
fig, ax = plt.subplots(figsize=(10, 6))
width = 0.4
color_lab = 'orange'
color_con = 'blue'

ax.bar([x - width/2 for x in range(len(proportions_lab))], proportions_lab, width, label='Lab', color=color_lab)
ax.bar([x + width/2 for x in range(len(proportions_con))], proportions_con, width, label='Con', color=color_con)


ax.set_ylabel('Proportion')
ax.set_title('Proportions of Moral Sentiment by Party - customised dictionary')
ax.set_xticks(range(len(foundation_labels)))
ax.set_xticklabels([foundation_labels[f] for f in [0, 1, 2]])
ax.legend()

plt.tight_layout()
plt.show()


# In[74]:


#this code plots the bar chat for the customised dictionary model for moral foundations

# Here i am finding the total number of texts for conservative and labour MPs
total_texts_con = len(df2[df2['party2'] == 'Con'])
total_texts_lab = len(df2[df2['party2'] == 'Lab'])

for foundation in [0, 1, 4, 6]:
    foundation_texts_con = df2[(df2['recoded'] == foundation) & (df2['party2'] == 'Con')]
    foundation_texts_lab = df2[(df2['recoded'] == foundation) & (df2['party2'] == 'Lab')]

    proportion_con = len(foundation_texts_con) / total_texts_con if total_texts_con > 0 else 0.0
    proportion_lab = len(foundation_texts_lab) / total_texts_lab if total_texts_lab > 0 else 0.0

    proportions_conservative[foundation] = proportion_con
    proportions_labour[foundation] = proportion_lab

# Here i am finding the proportions for each moral foundation categrory for lab and con MPs
for foundation in [0, 1, 4, 6]:
    print(f"Moral Foundation {foundation}")
    print(f"Proportion of Conservative Texts: {proportions_conservative[foundation]}")
    print(f"Proportion of Labour Texts: {proportions_labour[foundation]}")
    
# here i am assigning labels to the bars for each moral foundation column
foundation_labels = {
    0: 'non moral',
    1: 'care/harm',
    4: 'Authority/Respect',
    6: 'moral other'
}

proportions_lab = [proportions_labour[f] for f in [0, 1, 4, 6]]
proportions_con = [proportions_conservative[f] for f in [0, 1, 4, 6]]

# This code plots the graph.
fig, ax = plt.subplots(figsize=(10, 6))
width = 0.4

color_lab = 'orange'
color_con = 'blue'

ax.bar([x - width/2 for x in range(len(proportions_lab))], proportions_lab, width, label='Lab', color=color_lab)
ax.bar([x + width/2 for x in range(len(proportions_con))], proportions_con, width, label='Con', color=color_con)
ax.set_ylabel('Proportion')
ax.set_title('Proportions of Moral Foundations by Party- customised dictionary')
ax.set_xticks(range(len(foundation_labels)))
ax.set_xticklabels([foundation_labels[f] for f in [0, 1, 4, 6]])
ax.legend()

plt.tight_layout()
plt.show()


# In[87]:


#this code plots the bar chat for the BERT model for moral foundations

# Here  iam finding the total number of texts for conservative and labour MPs
total_texts_con = len(df2[df2['party2'] == 'Con'])
total_texts_lab = len(df2[df2['party2'] == 'Lab'])

for foundation in [0, 1, 4, 6]:
    foundation_texts_con = df2[(df2['predicted_labels'] == foundation) & (df2['party2'] == 'Con')]
    foundation_texts_lab = df2[(df2['predicted_labels'] == foundation) & (df2['party2'] == 'Lab')]

    proportion_con = len(foundation_texts_con) / total_texts_con if total_texts_con > 0 else 0.0
    proportion_lab = len(foundation_texts_lab) / total_texts_lab if total_texts_lab > 0 else 0.0

    proportions_conservative[foundation] = proportion_con
    proportions_labour[foundation] = proportion_lab

# Here i am finding the proportions for each moral foundation separately for conservative and labour Mps
for foundation in [0, 1, 4, 6]:
    print(f"Moral Foundation {foundation}")
    print(f"Proportion of Conservative Texts: {proportions_conservative[foundation]}")
    print(f"Proportion of Labour Texts: {proportions_labour[foundation]}")
    
#Here I am assigning the moral foundations labels for the bars of the bar chart
foundation_labels = {
    0: 'non moral',
    1: 'care/harm',
    4: 'Authority/Respect',
    6: 'moral other'
}

proportions_lab = [proportions_labour[f] for f in [0, 1, 4, 6]]
proportions_con = [proportions_conservative[f] for f in [0, 1, 4, 6]]

#This code plots the bar graph.
fig, ax = plt.subplots(figsize=(10, 6))
width = 0.4
color_lab = 'orange'
color_con = 'blue'

ax.bar([x - width/2 for x in range(len(proportions_lab))], proportions_lab, width, label='Lab', color=color_lab)
ax.bar([x + width/2 for x in range(len(proportions_con))], proportions_con, width, label='Con', color=color_con)
ax.set_ylabel('Proportion')
ax.set_title('Proportions of Moral Foundations by Party- BERT')
ax.set_xticks(range(len(foundation_labels)))
ax.set_xticklabels([foundation_labels[f] for f in [0, 1, 4, 6]])
ax.legend()

plt.tight_layout()
plt.show()


# In[76]:


#this code plots the bar chat for the wrod2vec model for moral foundations

# Here i am calculating the total number of texts for conservative and labour MPs
total_texts_con = len(df3[df3['party2'] == 'Con'])
total_texts_lab = len(df3[df3['party2'] == 'Lab'])

for foundation in [0, 1, 4, 6]:
    foundation_texts_con = df3[(df3['moral_embeddings2'] == foundation) & (df3['party2'] == 'Con')]
    foundation_texts_lab = df3[(df3['moral_embeddings2'] == foundation) & (df3['party2'] == 'Lab')]

    proportion_con = len(foundation_texts_con) / total_texts_con if total_texts_con > 0 else 0.0
    proportion_lab = len(foundation_texts_lab) / total_texts_lab if total_texts_lab > 0 else 0.0

    proportions_conservative[foundation] = proportion_con
    proportions_labour[foundation] = proportion_lab

# Here i am finding the proportions for each moral foundation separately for conservative and labour Mps
for foundation in [0, 1, 4, 6]:
    print(f"Moral Foundation {foundation}")
    print(f"Proportion of Conservative Texts: {proportions_conservative[foundation]}")
    print(f"Proportion of Labour Texts: {proportions_labour[foundation]}")
    
# here I am assigning the bar labels for each moral foundation
foundation_labels = {
    0: 'non moral',
    1: 'care/harm',
    4: 'Authority/Respect',
    6: 'moral other'
}

proportions_lab = [proportions_labour[f] for f in [0, 1, 4, 6]]
proportions_con = [proportions_conservative[f] for f in [0, 1, 4, 6]]

#This code plots the bar graph.
fig, ax = plt.subplots(figsize=(10, 6))
width = 0.4
color_lab = 'orange'
color_con = 'blue'

ax.bar([x - width/2 for x in range(len(proportions_lab))], proportions_lab, width, label='Lab', color=color_lab)
ax.bar([x + width/2 for x in range(len(proportions_con))], proportions_con, width, label='Con', color=color_con)
ax.set_ylabel('Proportion')
ax.set_title('Proportions of Moral Foundations by Party- customised dictionary')
ax.set_xticks(range(len(foundation_labels)))
ax.set_xticklabels([foundation_labels[f] for f in [0, 1, 4, 6]])
ax.legend()

plt.tight_layout()
plt.show()


# GETTING THE SUMMARY STATS FOR THE FOREIGN SECRETARIES DATASET using this code: https://pythonnumericalmethods.berkeley.edu/notebooks/chapter05.01-For-Loops.html and this code: https://stackoverflow.com/questions/38530130/group-by-count-and-calculate-proportions-in-pandas

# In[86]:


for mp, year_range in foreign_secretary_tenures.items():
    start_year, end_year = year_range[0], year_range[1]
    mp_sentences = filtered_df[(filtered_df['speaker2'] == mp) & (filtered_df['Year'] >= start_year) & (filtered_df['Year'] <= end_year)]
    filtered_sentences_df1 = pd.concat([filtered_sentences_df1, mp_sentences])

sentences_per_foreign_secretary = filtered_sentences_df1.groupby('speaker2').size().reset_index(name='Number_of_references')

grouped_by_speaker = filtered_sentences_df1.groupby('speaker2')
average_words_per_reference = grouped_by_speaker['Reference to China'].apply(lambda x: np.mean([len(reference.split()) for reference in x]))

summary_df = pd.DataFrame({
    'Foreign_Secretary': average_words_per_reference.index,
    'Average_Words_per_Reference': average_words_per_reference.values
})

print("Summary Statistics for Average Words in References to China:")
print(summary_df)

overall_average_words = average_words_per_reference.mean()
print(f"Overall Average Words per Reference to China: {overall_average_words:.2f}")
labour_subset = filtered_sentences_df1[filtered_sentences_df1['party2'] == 'Lab']
conservative_subset = filtered_sentences_df1[filtered_sentences_df1['party2'] == 'Con']
labour_average_words = labour_subset['Reference to China'].apply(lambda x: len(x.split())).mean()
conservative_average_words = conservative_subset['Reference to China'].apply(lambda x: len(x.split())).mean()

print(f"Average Words per Reference to China for Labour: {labour_average_words:.2f}")
print(f"Average Words per Reference to China for Conservative: {conservative_average_words:.2f}")

john_major_sentences = filtered_df[(filtered_df['speaker2'] == 'John Major') & (filtered_df['Year'] == 1989)]
john_major_count = len(john_major_sentences)
print("Number of sentences spoken by John Major in 1989:", john_major_count)


# In[ ]:




