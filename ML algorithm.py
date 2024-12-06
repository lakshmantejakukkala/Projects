# import the relevant libraries
import pandas as pd
import numpy as np

# load the preprocessed CSV data
data_preprocessed = pd.read_csv('Absenteeism_preprocessed.csv')

# create targets for our logistic regression
# for instance, we could have assigned some arbitrary value as a cut-off line, instead of the median
# note that what line does is to assign 1 to anyone who has been absent 4 hours or more (more than 3 hours)
# that is the equivalent of taking half a day off
targets = np.where(data_preprocessed['Absenteeism Time in Hours'] >
                   data_preprocessed['Absenteeism Time in Hours'].median(), 1, 0)

# create a Series in the original data frame that will contain the targets for the regression
data_preprocessed['Excessive Absenteeism'] = targets

# create a checkpoint by dropping the unnecessary variables
data_with_targets = data_preprocessed.drop(['Absenteeism Time in Hours'],axis=1)

# create a variable that will contain the inputs (everything without the targets)
unscaled_inputs = data_with_targets.iloc[:,:-1]

# standardize the inputs

# standardization is one of the most common preprocessing tools
from sklearn.preprocessing import StandardScaler

# define scaler as an object
absenteeism_scaler = StandardScaler()

# fit the unscaled_inputs
absenteeism_scaler.fit(unscaled_inputs)

# transform the unscaled inputs
scaled_inputs = absenteeism_scaler.transform(unscaled_inputs)

# import train_test_split so we can split our data into train and test
from sklearn.model_selection import train_test_split

# declare 4 variables for the split
x_train, x_test, y_train, y_test = train_test_split(scaled_inputs, targets, #train_size = 0.8,
                                                                            test_size = 0.2, random_state = 20)

# import the LogReg model from sklearn
from sklearn.linear_model import LogisticRegression

# import the 'metrics' module, which includes important metrics we may want to use
from sklearn import metrics

# create a logistic regression object
reg = LogisticRegression()

# fit our train inputs
reg.fit(x_train,y_train)

# save the names of the columns in an ad-hoc variable
feature_name = unscaled_inputs.columns.values

# creating a summary table
summary_table = pd.DataFrame (columns=['Feature name'], data = feature_name)

# add the coefficient values to the summary table
summary_table['Coefficient'] = np.transpose(reg.coef_)

# move all indices by 1
summary_table.index = summary_table.index + 1

# add the intercept at index 0
summary_table.loc[0] = ['Intercept', reg.intercept_[0]]

# sort the df by index
summary_table = summary_table.sort_index()

# create a new Series called: 'Odds ratio' which will show the.. odds ratio of each feature
summary_table['Odds_ratio'] = np.exp(summary_table.Coefficient)

# sort the table according to odds ratio
summary_table.sort_values('Odds_ratio', ascending=False)
