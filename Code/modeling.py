#===========================================================#
# METRO RIDERSHIP ANALYSIS - MODELING
#
# NOTE: Project for GA Data Science class.
# I'm interested in looking at the factors that affect
# Metro rail ridership and modeling their relationships
#===========================================================#
# CREATED BY: Lena Nguyen - April 4, 2015
#===========================================================#

import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import cross_val_score
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import RFE

# Set current working directory
os.chdir("/Users/Zelda/Data Science/GA/Project/Data")

#=============#
# IMPORT DATA #
#=============#

data = pd.read_csv('model_data.csv')

# Split data into weekday and weekends (Temporary)
weekday = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']

wkday = data[data.Weekday.isin(weekday)]
wkend = data[~(data.Weekday.isin(weekday))]

# Alternatively, create ridership per car variable
data['RidersPC'] = data['Riders']/data['Cars']
data.head()  # Check it worked

cols = data.columns.values
feats = cols[5:-2]

# ! Temporary
data['Registered'].fillna(value=0, inplace=True)
data['Casual'].fillna(value=0, inplace=True)

X = data[feats]
Y = data['RidersPC']

#=======================================================#
# STANDARDIZE DATA
#=======================================================#

std_X = scale(X)
std_Y = scale(Y)

# Change into dataframe for easier visualizing
std_X = pd.DataFrame(data=std_X, columns=feats)
std_Y = pd.DataFrame(data=std_Y, columns=['RidersPC'])

#=======================================================#
# SPLIT DATASET
#=======================================================#

# Split into train and test datasets
# Test is 30% of complete dataset
X_train, X_test, Y_train, Y_test = train_test_split(std_X, std_Y, test_size=0.3, random_state=1)

# Convert test/train data to data fame
X_train = pd.DataFrame(data=X_train, columns=feats)
X_test = pd.DataFrame(data=X_test, columns=feats)
Y_train = pd.DataFrame(data=Y_train, columns=['RidersPC'])
Y_test = pd.DataFrame(data=Y_test, columns=['RidersPC'])

#=======================================================#
# FEATURE SELECTION
#=======================================================#

# Find features that are significant
ftest = f_regression(X,Y)
feat_sel = pd.DataFrame(data=ftest[0], index=feats, columns=['F_score'])  # makes data frame with p values
feat_sel['p_values'] = ftest[1]  # I want to see the relationship between F-score and p values

feat_sel
#-- Let's put the threshold at F_score of 40.
#-- From this, it appears the significant features are:
#-- SNWD, TMAX, TMIN, Gas_Price, Labor Force, Employment
#-- Registered Cabi Riders, Holiday

# Try out recurse feature elimination
lm = LinearRegression()
rfe = RFE(lm, 8)  # selects top 8 features
rfe = rfe.fit(X_train,Y_train)

print(rfe.support_)
print(rfe.ranking_)
rfe.ranking_

feat_sel['Ranking'] = rfe.ranking_

feat_sel

#=======================================================#
# LINEAR REGRESSION MODEL
#=======================================================#

mfeats = ['SNWD', 'TMAX', 'TMIN', 'Labor Force', 'Employment', 'Registered', 'Holiday']

m_X = std_X[mfeats]
m_Y = data['RidersPC']

mX_train, mX_test, mY_train, mY_test = train_test_split(m_X, m_Y, test_size=0.3, random_state=1)

# Convert test/train data to data fame
mX_train = pd.DataFrame(data=mX_train, columns=mfeats)
mX_test = pd.DataFrame(data=mX_test, columns=mfeats)
mY_train = pd.DataFrame(data=mY_train, columns=['RidersPC'])
mY_test = pd.DataFrame(data=mY_test, columns=['RidersPC'])

# Simple Linear Regression model
plm = LinearRegression()
plm.fit(mX_train, mY_train)

plm.intercept_
plm.coef_

#============#
# MODEL EVAL #
#============#

# Evaluate the fit of the model based off of the training set
assert not np.any(np.isnan(mX_test) | np.isinf(mX_test))  # Just to be safe
preds = plm.predict(mX_test)
np.sqrt(mean_squared_error(mY_test,preds))
# RMSE = 1265.869

# Plot the residuals across the range of predicted values
preds = np.ravel(preds)  # flatten ndarray

resid = preds - mY_test['RidersPC']
plt.scatter(preds, resid, alpha=0.7)
plt.xlabel("Predicted Ridership per Car")
plt.ylabel("Residuals")
plt.show()
# Looking at the graph, the residuals are better for most cases than the first model
# But the cases that are off are way off so are driving the RMSE up

# Evaluate the model fit based off of cross validation
scores = cross_val_score(plm, X, Y, cv=10, scoring='mean_squared_error')
np.mean(np.sqrt(-scores))
# 1239.7653
