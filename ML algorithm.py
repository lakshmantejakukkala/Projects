# import the relevant libraries
import pandas as pd
import numpy as np

# load the preprocessed CSV data
data_preprocessed = pd.read_csv('Absenteeism_preprocessed.csv')

# create targets for our logistic regression
# they have to be categories and we must find a way to say if someone is 'being absent too much' or not
# what we've decided to do is to take the median of the dataset as a cut-off line
# in this way the dataset will be balanced (there will be roughly equal number of 0s and 1s for the logistic regression)
# as balancing is a great problem for ML, this will work great for us
# alternatively, if we had more data, we could have found other ways to deal with the issue
# for instance, we could have assigned some arbitrary value as a cut-off line, instead of the median

# note that what line does is to assign 1 to anyone who has been absent 4 hours or more (more than 3 hours)
# that is the equivalent of taking half a day off

# initial code from the lecture
# targets = np.where(data_preprocessed['Absenteeism Time in Hours'] > 3, 1, 0)

# parameterized code
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
# since data of different magnitude (scale) can be biased towards high values,
# we want all inputs to be of similar magnitude
# this is a peculiarity of machine learning in general - most (but not all) algorithms do badly with unscaled data

# a very useful module we can use is StandardScaler
# it has much more capabilities than the straightforward 'preprocessing' method
from sklearn.preprocessing import StandardScaler


# we will create a variable that will contain the scaling information for this particular dataset
# here's the full documentation: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

# define scaler as an object
absenteeism_scaler = StandardScaler()

# fit the unscaled_inputs
# this basically calculates the mean and standard deviation of each feature
absenteeism_scaler.fit(unscaled_inputs)

# transform the unscaled inputs
# this is the scaling itself - we subtract the mean and divide by the standard deviation
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
# that is basically the whole training part of the machine learning
reg.fit(x_train,y_train)

# save the names of the columns in an ad-hoc variable
feature_name = unscaled_inputs.columns.values

# use the coefficients from this table (they will be exported later and will be used in Tableau)
# transpose the model coefficients (model.coef_) and throws them into a df (a vertical organization, so that they can be
# multiplied by certain matrices later)
summary_table = pd.DataFrame (columns=['Feature name'], data = feature_name)

# add the coefficient values to the summary table
summary_table['Coefficient'] = np.transpose(reg.coef_)

# do a little Python trick to move the intercept to the top of the summary table
# move all indices by 1
summary_table.index = summary_table.index + 1

# add the intercept at index 0
summary_table.loc[0] = ['Intercept', reg.intercept_[0]]

# sort the df by index
summary_table = summary_table.sort_index()

# create a new Series called: 'Odds ratio' which will show the.. odds ratio of each feature
summary_table['Odds_ratio'] = np.exp(summary_table.Coefficient)

# sort the table according to odds ratio
# note that by default, the sort_values method sorts values by 'ascending'
summary_table.sort_values('Odds_ratio', ascending=False)
