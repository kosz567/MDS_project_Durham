# -*- coding: utf-8 -*-
"""trendsovertimeattempt

Automatically generated by Colaboratory.
"""

#this shows an attempt and looking at the trends over time in the dataset, with more research time this could have been fully carried out.
from google.colab import files
uploaded_files = files.upload()

import pandas as pd
file_path = "BERT_PREDICTIONS.csv"
df_analysis = pd.read_csv(file_path)

df_analysis.head()

X = df_analysis['predicted_labels2'].value_counts()
print(X)
y = df_analysis['predicted_labels'].value_counts()
print(y)
X = df_analysis['predicted_labels2'].value_counts()
X_percentage = (X / X.sum()) * 100
y = df_analysis['predicted_labels'].value_counts()
y_percentage = (y / y.sum()) * 100
print(X_percentage)
print(y_percentage)

"""Here  am performing a linear regression on the categories for moral sentiment for labour and conservative Mps using the predicted labels from the DistiBERT model."""

import numpy as np
import statsmodels.api as sm
# setting up the list of moral categories for sentiment to be regressed
categories = [0, 1, 2]

# Here I am separating the data out into Lab and con mps to be assessed
df_lab = df_analysis[(df_analysis['party2'] == 'Lab') & (df_analysis['Year'] != 2023)]
df_con = df_analysis[(df_analysis['party2'] == 'Con') & (df_analysis['Year'] != 2023)]
grouped_data_lab = df_lab[df_lab['predicted_labels2'].isin(categories)].groupby(['Year', 'predicted_labels2']).size().unstack().fillna(0) # these two lines group the data by year and also by the predicted labels
grouped_data_con = df_con[df_con['predicted_labels2'].isin(categories)].groupby(['Year', 'predicted_labels2']).size().unstack().fillna(0)

#this is a check to makes sure that the dataframes for lab and con are the same length
unique_years_lab = df_lab['Year'].unique()
unique_years_con = df_con['Year'].unique()
unique_years = np.intersect1d(unique_years_lab, unique_years_con)
df_lab_filtered = df_lab[df_lab['Year'].isin(unique_years)]
df_con_filtered = df_con[df_con['Year'].isin(unique_years)]

# getting the total number of references so that proportions can be calculated for lab and con.
total_per_year_lab = df_lab_filtered.groupby('Year').size()
total_per_year_con = df_con_filtered.groupby('Year').size()
proportions_lab= grouped_data_lab / total_per_year_lab[:, np.newaxis]
proportions_con= grouped_data_con / total_per_year_con[:, np.newaxis]
#creating on dataframe with both lab and com labels in it and party labels.
proportions_df = pd.concat([proportions_lab, proportions_con], axis=1)
proportions_df.columns = ['Lab0', 'Lab_1', 'Lab_2', 'Con_0', 'Con_1', 'Con_2']
proportions_df.reset_index(inplace=True)

#the last section of the code performs the standard regression analysis.
X = proportions_df[['Year']]
X = sm.add_constant(X)
y = proportions_df[['Lab0', 'Lab_1', 'Lab_2', 'Con_0', 'Con_1', 'Con_2']]
#this makes sure the results are print out in a table format.
results = []
for column in y.columns:
    model = sm.OLS(y[column], X).fit()
    results.append({'Party_Foundation': column, 'Intercept': model.params['const'], 'Coefficient': model.params['Year'], 'R-squared': model.rsquared, 'P-value': model.pvalues['Year']})
regression_results = pd.DataFrame(results)
print(regression_results)