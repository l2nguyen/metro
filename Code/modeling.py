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
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import cross_val_score

# Set current working directory
os.chdir("/Users/Zelda/Data Science/GA/Project/Data")

#=============#
# IMPORT DATA #
#=============#

data = pd.read_csv('model_data.csv')

# Alternatively, create ridership per car variable
data['RidersPC'] = data['Riders']/data['Cars']
data.head()  # Check it worked

#=======================================================#
# DEAL WITH OUTLIERS
#=======================================================#

# Make z score for Riders Per Car data to flag outlier
data['RidersPC_z'] = (data['RidersPC'] - data['RidersPC'].mean()) / data['RidersPC'].std()

# Look at obs +/- 2.5 SD away
data[(abs(data['RidersPC_z']) >= 2.5)]
#- 53 observations out of 4018 (~1% of obs)
#- The majority of outliers are on the minus side.
#- Also, the majority of them are holidays.
#- Will try making two different models for holiday and regular days

# Trim outliers from dataset
trim_data = data[(abs(data['RidersPC_z']) < 2.5)]

# Identify feature variabbles
feats = data.columns.values[5:-3]

# Bikeshare did not exist before 2010 so will fill NaN values with 0
# for all the other models
trim_data['Registered'].fillna(value=0, inplace=True)
trim_data['Casual'].fillna(value=0, inplace=True)
trim_data.isnull().sum()  # check it worked

X = trim_data[feats]
Y = trim_data['RidersPC']

#=======================================================#
# FEATURE SELECTION (TRIMMED DATASET)
#=======================================================#


def feat_sel(feats, resp):  # Function to run feature selection
    std_feats = scale(feats)  # Standardize features
    std_resp = scale(resp)  # Standardize response variable
    # Run f-test for feature selection
    ftest = f_regression(std_feats, std_resp)
    feat_sel = pd.DataFrame(data=ftest[0], index=feats.columns.values, columns=['F_score'])  # makes data frame with p values
    feat_sel['p_values'] = ftest[1]  # Make p-values column
    return feat_sel  # print out feature selection

feat_sel(X,Y)  # Run feature selection on trimmed dataset
#-- Let's put the threshold at F_score of 40.
#-- From this, it appears the significant features are:
#-- TMAX, TMIN, Gas_Price, Labor Force, Employment
#-- Registered CaBi Riders, Casual CaBi Riders, Holiday

# Use only feats that have an F-score over the threshold
new_feats = ['TMAX', 'TMIN', 'Labor Force', 'Employment', 'Registered', 'Casual', 'Holiday']

X = trim_data[new_feats]
Y = trim_data['RidersPC']

#=======================================================#
# SPLIT INTO TRAIN/TEST (TRIMMED DATASET)
#=======================================================#

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

# Change into data frames for easier visualization
X_train = pd.DataFrame(data=X_train, columns=X.columns.values)
X_test = pd.DataFrame(data=X_test, columns=X.columns.values)
Y_train = pd.DataFrame(data=Y_train, columns=['RidersPC'])
Y_test = pd.DataFrame(data=Y_test, columns=['RidersPC'])

#=======================================================#
# LINEAR REGRESSION MODEL (TRIMMED DATASET)
#=======================================================#

# Linear regression model
plm = LinearRegression()
plm.fit(X_train, Y_train)

plm.intercept_
plm.coef_

#============#
# MODEL EVAL #
#============#

# Evaluate the fit of the model based off of the training set
assert not np.any(np.isnan(X_test) | np.isinf(X_test))  # Just to be safe
preds = plm.predict(X_test)
np.sqrt(mean_squared_error(Y_test,preds))
# RMSE = 1137.2
# Not bad. It's about equivalent to the standard deviation of the Riders per car variable

# Plot the residuals across the range of predicted values
preds = np.ravel(preds)  # flatten ndarray

resid = preds - Y_test['RidersPC']
plt.scatter(preds, resid, alpha=0.7)
plt.xlabel("Predicted Ridership per Car")
plt.ylabel("Residuals")
plt.show()
# Looking at the graph, it appears that the residuals fall more on the negative side
# This model seems to lean towards predicting inaccurately on the lesser side than the higher side

# Evaluate the model fit based off of cross validation
scores = cross_val_score(plm, X, Y, cv=20, scoring='mean_squared_error')
np.mean(np.sqrt(-scores))
# RMSE from cross validation = 1175.67

#=======================================================#
# RANDOM FOREST MODEL
#=======================================================#

rtr = ensemble.RandomForestRegressor()
rtr.fit(X_train,Y_train)
rtr.score(X_train,Y_train)
# R squared is 0.862 - OK

rtr.score(X_test,Y_test)
# really terrible - it does not appear this current model is generalizable

rtr_select = pd.DataFrame(data=rtr.feature_importances_, index=X.columns.values, columns=['importance'])

preds = rtr.predict(X)
mean_squared_error(Y, preds)
# 554945 - really terrible

resid = preds - Y
plt.scatter(preds, resid, alpha=0.7)
plt.xlabel("Predicted Riders per Car")
plt.ylabel("Residuals")
plt.show()
#-- The residuals are all over the place
#-- Probably will not use random forest model

##########################################################################

#=======================================================#
# DATASET SPLIT FOR WEEKDAY/WEEKEND MODELS
#=======================================================#

# NOTE: Using dataset where the outliers have been trimmed

# WEEKDAY/WEEKEND SPLIT -------------------------------
weekday = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']

# Feature dataset
X_wkday = trim_data[feats][data.Weekday.isin(weekday)]
X_wkend = trim_data[feats][~(data.Weekday.isin(weekday))]

# Respnse dataset
Y_wkday = trim_data['RidersPC'][data.Weekday.isin(weekday)]
Y_wkend = trim_data['RidersPC'][~(data.Weekday.isin(weekday))]

#=======================================================#
# FEATURE SELECTION (WEEKDAY/WEEKEND)
#=======================================================#

# ? Wonder if weekday/weekends will have different feature importance

feat_sel(X_wkday, Y_wkday)
# About the same as the full dataset but CaBi seems to have no importance

feat_sel(X_wkend, Y_wkend)
# Employment variables less important. CaBi more important
# Holiday has very little effect on weekend ridership

# Weekday features (thresdhold: F-score>40)
wkday_feats = ['SNWD', 'TMAX', 'TMIN', 'Gas_Price', 'Labor Force', 'Employment', 'Unemployment', 'Holiday']
wkend_feats = ['TMAX', 'TMIN', 'Gas_Price', 'Labor Force', 'Employment', 'Casual']

# Replace old datasets with new one with selected features
X_wkday = X_wkday[wkday_feats]
X_wkend = X_wkend[wkend_feats]

#=======================================================#
# SPLIT INTO TRAIN/TEST (TRIMMED DATASET)
#=======================================================#

# WEEKDAY DATASET
Xwkday_train, Xwkday_test, Ywkday_train, Ywkday_test = train_test_split(X_wkday, Y_wkday, test_size=0.3, random_state=1)

# WEEKEND DATASET
Xwkend_train, Xwkend_test, Ywkend_train, Ywkend_test = train_test_split(X_wkend, Y_wkend, test_size=0.3, random_state=1)

#=======================================================#
# LINEAR REGRESSION MODEL (WEEKDAY/WEEKEND MODELS)
#=======================================================#

# WEEKDAY MODEL
wlm = LinearRegression()
wlm.fit(Xwkday_train, Ywkday_train)

wlm.intercept_
wlm.coef_

# WEEKDAY MODEL
wnlm = LinearRegression()
wnlm.fit(Xwkend_train, Ywkend_train)

wnlm.intercept_
wnlm.coef_

#=====================================#
# MODEL EVAL (WEEKDAY/WEEKEND MODELS) #
#=====================================#


#===============#
# WEEKDAY MODEL
#===============#

# Evaluate the fit of the model based off of the training set
assert not np.any(np.isnan(Xwkday_test) | np.isinf(Xwkday_test))  # Just to be safe
preds = wlm.predict(Xwkday_test)
np.sqrt(mean_squared_error(Ywkday_test,preds))
# RMSE = 648.1
# Performs better than the model of the full dataset

# Plot the residuals across the range of predicted values
preds = np.ravel(preds)  # flatten ndarray

resid = preds - Ywkday_test
plt.scatter(preds, resid, alpha=0.7)
plt.xlabel("Predicted Ridership per Car (Weekdays)")
plt.ylabel("Residuals")
plt.show()
# Got most of the points right but some points are pretty off driving up the RMSE

# Evaluate the model fit based off of cross validation
scores = cross_val_score(plm, X_wkday, Y_wkday, cv=20, scoring='mean_squared_error')
np.mean(np.sqrt(-scores))
# RMSE from cross validation = 646.72

#===============#
# WEEKEND MODEL
#===============#

# Evaluate the fit of the model based off of the training set
assert not np.any(np.isnan(Xwkend_test) | np.isinf(Xwkend_test))  # Just to be safe
preds = wnlm.predict(Xwkend_test)
np.sqrt(mean_squared_error(Ywkend_test,preds))
# RMSE = 629.24
# Much worse at predicting weekend ridership

# Plot the residuals across the range of predicted values
preds = np.ravel(preds)  # flatten ndarray

resid = preds - Ywkend_test
plt.scatter(preds, resid, alpha=0.7)
plt.xlabel("Predicted Ridership per Car (Weekends)")
plt.ylabel("Residuals")
plt.show()
# Got most of the points right but some points are pretty off driving up the RMSE

# Evaluate the model fit based off of cross validation
scores = cross_val_score(wnlm, X_wkend, Y_wkend, cv=20, scoring='mean_squared_error')
np.mean(np.sqrt(-scores))
# RMSE from cross validation = 687.03

#-- This double model performs better than the full dataset model
#-- However, the lower sample size for weekend will cause issues with how
#-- generalizable this model will be




# SPLIT OFF DATASET WHERE CAPITAL BIKESHARE EXISTED
dcabi = data[data['Registered'].isnull() == False]
dcabi.describe()

# Make sure the min year is 2010 and max year is 2014
assert min(dcabi.Year) == 2010
assert max(dcabi.Year) == 2014
dcabi.isnull().sum()

X_cabi = dcabi[feats]
Y_cabi = dcabi['RidersPC']




# HOLIDAY/REGULAR DAY SPLIT ----------------------------
# Feature dataset
X_regular = data[feats][data['Holiday'] == 0]
X_holiday = data[feats][data['Holiday'] == 1]

# Response dataset
Y_regular = data['RidersPC'][data['Holiday'] == 0]
Y_holiday = data['RidersPC'][data['Holiday'] == 1]
