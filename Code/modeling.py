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
data.columns.values

# Create ridership per train variable
data['RidersPC'] = data['Riders']/data['Cars']
data.head()  # Check it worked

# Bikeshare did not exist before 2010 so will fill NaN values with 0
# for all the models
data['Registered'].fillna(value=0, inplace=True)
data['Casual'].fillna(value=0, inplace=True)
data.isnull().sum()  # check it worked

#=======================================================#
# DEAL WITH OUTLIERS
#=======================================================#

# Make z score for Riders Per Car data to flag outliers in that variable
data['RidersPC_z'] = (data['RidersPC'] - data['RidersPC'].mean()) / data['RidersPC'].std()

# Look at obs +/- 3 SD away
data[(abs(data['RidersPC_z']) >= 3)]
#- 16 observations out of 4018 (<1% of obs)
#- The majority of outliers are on the minus side.
#- Also, the majority of them are holidays.
#- Will try making two different models for holiday and regular days

# Trim outliers from dataset
trim_data = data[(abs(data['RidersPC_z']) < 3)]

# Define feature and response variables
feats = data.columns.values[5:-3]
resp = ['RidersPC']

X = trim_data[feats]
Y = trim_data[resp]

#=======================================================#
# SPLIT INTO TRAIN/TEST (TRIMMED DATASET)
#=======================================================#
# NOTE: These train/test sets have all the features.
# Features selection is not needed for random forest
# and gradient boosting

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

#=======================================================#
# RANDOM FOREST MODEL (TRIMMED DATA)
#=======================================================#

rtr = ensemble.RandomForestRegressor()
rtr.fit(X_train,Y_train)
rtr.score(X_train,Y_train)
# R squared is 0.852 - Pretty good

rtr.score(X_test,Y_test)
# 0.241 - it does not appear this current model is generalizable

preds = rtr.predict(X_test)
np.sqrt(mean_squared_error(Y_test, preds))
# 1148.69 - only mildly better than the linear regression model

# Plot residualsR
resid = preds - Y.T
plt.scatter(preds, resid, alpha=0.7)
plt.xlabel("Predicted Riders per Train")
plt.ylabel("Residuals")
plt.show()
#-- The residuals are all over the place
#-- Probably will not use random forest model

#=======================================================#
# GRADIENT BOOSTING MODEL (TRIMMED DATA)
#=======================================================#

gbm = ensemble.GradientBoostingRegressor()
gbm.fit(X_train, Y_train)
gbm.score(X_train, Y_train)
# R^2 is 0.41 - worse than random forest model

gbm.score(X_test,Y_test)
# R^2 = 0.345 - better at predicting virgin data
preds = gbm.predict(X_test)
np.sqrt(mean_squared_error(Y_test, preds))
# RMSE = 1067.07 - performs better than random forest and linear regression model

# Look at feature importance in the two models
f_select = pd.DataFrame(data=rtr.feature_importances_, index=X.columns.values, columns=['RF'])
f_select['GBM'] = gbm.feature_importances_
#- In general, they agree about which features are more important.

#==========================================================#
# FEATURE SELECTION (TRIMMED DATASET) - LINEAR REGRESSION
#==========================================================#


def feat_sel(feats, resp):  # Function to run feature selection
    std_feats = scale(feats)  # Standardize features
    std_resp = scale(resp)  # Standardize response variable
    # Run f-test for feature selection
    ftest = f_regression(std_feats, std_resp)
    feat_sel = pd.DataFrame(data=ftest[0], index=feats.columns.values, columns=['F_score'])  # makes data frame with p values
    feat_sel['p_values'] = ftest[1]  # Make p-values column
    return feat_sel  # print out feature selection

feat_sel(X,Y)  # Run feature selection on trimmed dataset
#-- Let's put the threshold at F_score of 30.

# Use only feats that have an F-score over the threshold
new_feats = ['TMAX', 'TMIN', 'SNWD', 'Gas_Price', 'Labor Force', 'Employment', 'Registered', 'Casual', 'Holiday']

X_new = trim_data[new_feats]
Y_new = trim_data[resp]

#=======================================================#
# SPLIT INTO TRAIN/TEST (TRIMMED DATASET)
#=======================================================#
# NOTE: This train/test dataset does not have all the features

X_train, X_test, Y_train, Y_test = train_test_split(X_new, Y_new, test_size=0.3, random_state=1)

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
# RMSE = 1180.19
# Not bad. It's about equivalent to the standard deviation of the Riders per car variable

# Plot the residuals across the range of predicted values
resid = preds - Y_test[resp]
plt.scatter(preds, resid, alpha=0.7)
plt.xlabel("Predicted Ridership per Train")
plt.ylabel("Residuals")
plt.show()
# Looking at the graph, it appears that the residuals fall more on the negative side
# This model seems to lean towards predicting inaccurately on the lesser side than the higher side


def cvrmse(X,Y,lm):
    scores = cross_val_score(lm, X, Y, cv=20, scoring='mean_squared_error')
    return np.mean(np.sqrt(-scores))

# Evaluate the model fit based off of cross validation
cvrmse(X,Y,plm)
# RMSE = 1206.75

##########################################################################

#=======================================================#
# DATASET SPLIT FOR HOLIDAY/REGULAR DAY SPLIT
#=======================================================#

# Feature datasets
X_regular = trim_data[feats][trim_data['Holiday'] == 0]
X_holiday = trim_data[feats][trim_data['Holiday'] == 1]

# Response datasets
Y_regular = trim_data[resp][trim_data['Holiday'] == 0]
Y_holiday = trim_data[resp][trim_data['Holiday'] == 1]
#- NOTE: Holiday has only 134 obs
#- Any model made with this will probably not be very good

# Drop holiday from data frames
X_regular.drop('Holiday', axis=1, inplace=True)
X_holiday.drop('Holiday', axis=1, inplace=True)

#=======================================================#
# SPLIT INTO TRAIN/TEST (TRIMMED DATASET)
#=======================================================#

# HOLIDAY DATASET
Xh_train, Xh_test, Yh_train, Yh_test = train_test_split(X_holiday, Y_holiday, test_size=0.3, random_state=1)

# REGULAR DAY DATASET
Xr_train, Xr_test, Yr_train, Yr_test = train_test_split(X_regular, Y_regular, test_size=0.3, random_state=1)

#=======================================================#
# LINEAR REGRESSION MODEL (HOLIDAY/REGULAR MODELS)
#=======================================================#

# HOLIDAY MODEL
holm = LinearRegression()
holm.fit(Xh_train, Yh_train)

holm.intercept_
holm.coef_

# REGULAR DAY MODEL
relm = LinearRegression()
relm.fit(Xr_train, Yr_train)

relm.intercept_
relm.coef_

#=====================================#
# MODEL EVAL (HOLIDAY/REGULAR MODELS) #
#=====================================#

#===============#
# HOLIDAY MODEL
#===============#

# Evaluate the fit of the model based off of the training set
assert not np.any(np.isnan(Xh_test) | np.isinf(Xh_test))  # Just to be safe
preds = holm.predict(Xh_test)
np.sqrt(mean_squared_error(Yh_test,preds))
# RMSE = 1742.9

# Plot the residuals across the range of predicted values
resid = np.ravel(preds) - np.ravel(Yh_test)
plt.scatter(preds, resid, alpha=0.7)
plt.xlabel("Predicted Ridership per Train (Holidays)")
plt.ylabel("Residuals")
plt.show()
# All over the place. Probably because some holidays, people will travel more.
# And some holidays, people will travel less

# Evaluate the model fit based off of cross validation
cvrmse(X_holiday,Y_holiday,holm)
# RMSE from cross validation = 1624.22

#===================#
# REGULAR DAY MODEL
#===================#
# Evaluate the fit of the model based off of the training set
assert not np.any(np.isnan(Xr_test) | np.isinf(Xr_test))  # Just to be safe
preds = relm.predict(Xr_test)
np.sqrt(mean_squared_error(Yr_test, preds))
# RMSE = 1188.14

# Plot the residuals across the range of predicted values
resid = np.ravel(preds) - np.ravel(Yr_test)
plt.scatter(preds, resid, alpha=0.7)
plt.xlabel("Predicted Ridership per Train (Regular Days)")
plt.ylabel("Residuals")
plt.show()
# Model more likely to predict fewer riders than there should be
# Similar to the residual plots of the model with no split at all

# Evaluate the model fit based off of cross validation
cvrmse(X_regular,Y_regular,relm)
# RMSE from cross validation = 1180.03

#- Because there were so few holidays in the dataset, this model performed
#- just as well as the one with no split at all.

##########################################################################

#=======================================================#
# DATASET SPLIT FOR WEEKDAY/WEEKEND MODELS
#=======================================================#

# NOTE: Using dataset where the outliers have been trimmed

# WEEKDAY/WEEKEND SPLIT
weekday = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']

# Feature dataset
X_wkday = trim_data[feats][trim_data.Weekday.isin(weekday)]
X_wkend = trim_data[feats][~(trim_data.Weekday.isin(weekday))]

# Respnse dataset
Y_wkday = trim_data[resp][trim_data.Weekday.isin(weekday)]
Y_wkend = trim_data[resp][~(trim_data.Weekday.isin(weekday))]

# See number of obsevations in each split dataset
len(X_wkday)  # 2858 obs
len(X_wkend)  # 1144 obs - half of weekdays

#=======================================================#
# SPLIT INTO TRAIN/TEST (WEEKDAY/WEEKEND)
#=======================================================#

# WEEKDAY DATASET
Xwkday_train, Xwkday_test, Ywkday_train, Ywkday_test = train_test_split(X_wkday, Y_wkday, test_size=0.3, random_state=1)

# WEEKEND DATASET
Xwkend_train, Xwkend_test, Ywkend_train, Ywkend_test = train_test_split(X_wkend, Y_wkend, test_size=0.3, random_state=1)

#=======================================================#
# GRADIENT BOOSTED REGRESSOR (WEEKDAY/WEEKEND MODELS)
#=======================================================#

# WEEKDAY MODEL
wgr = ensemble.GradientBoostingRegressor()
wgr.fit(Xwkday_train, Ywkday_train)

# WEEKEND MODEL
wngr = ensemble.GradientBoostingRegressor()
wngr.fit(Xwkend_train, Ywkend_train)

#=================================================#
# GRADIENT BOOSTING EVAL (WEEKDAY/WEEKEND MODELS) #
#=================================================#

#------- WEEKDAY MODEL -------#
wgr.score(Xwkday_train, Ywkday_train)  # Train sets R^2
# R^2 = 0.621
wgr.score(Xwkday_test, Ywkday_test)  # Test sets R^2
# R^2 = 0.520
preds = wgr.predict(Xwkday_test)
np.sqrt(mean_squared_error(Ywkday_test,preds))
# RMSE = 633.93 - better than linear regression

#------ WEEKEND MODEL --------#
wngr.score(Xwkend_train, Ywkend_train)  # Train sets R^2
# R^2 = 0.615
wngr.score(Xwkend_test, Ywkend_test)  # Test sets R^2
# R^2 = 0.342
preds = wngr.predict(Xwkend_test)
np.sqrt(mean_squared_error(Ywkend_test,preds))
# RMSE = 679.22 - better than linear regression

#=======================================================#
# FEATURE SELECTION (WEEKDAY/WEEKEND)
#=======================================================#

feat_sel(X_wkday, Y_wkday)
# About the same as the full dataset but CaBi seems to have no importance

feat_sel(X_wkend, Y_wkend)
# Employment variables less important. CaBi more important
# Holiday has very little effect on weekend ridership

# Weekday features (thresdhold: F-score>40)
wkday_feats = ['SNWD', 'TMAX', 'TMIN', 'Gas_Price', 'Labor Force', 'Employment', 'Unemployment', 'Holiday', 'Registered']
wkend_feats = ['PRCP', 'SNOW', 'TMAX', 'TMIN', 'Gas_Price', 'Labor Force', 'Employment', 'Holiday', 'Casual']

# Replace old datasets with new one with selected features
X_wkday = X_wkday[wkday_feats]
X_wkend = X_wkend[wkend_feats]

#=======================================================#
# SPLIT INTO TRAIN/TEST (WEEKDAY/WEEKEND)
#=======================================================#
# NOTE: This is after the feature selection

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

# WEEKEND MODEL
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
# RMSE = 721.89
# Performs better than the model of the full dataset

# Plot the residuals across the range of predicted values
resid = np.ravel(preds) - np.ravel(Ywkday_test)
plt.scatter(preds, resid, alpha=0.7)
plt.xlabel("Predicted Ridership per Train (Weekdays)")
plt.ylabel("Residuals")
plt.show()
# Got most of the points right but some points are pretty off driving up the RMSE

# Evaluate the model fit based off of cross validation
cvrmse(X_wkday,Y_wkday,wlm)
# RMSE from cross validation = 706.39 - slightly worse than gradient boosting

#===============#
# WEEKEND MODEL
#===============#

# Evaluate the fit of the model based off of the training set
assert not np.any(np.isnan(Xwkend_test) | np.isinf(Xwkend_test))  # Just to be safe
preds = wnlm.predict(Xwkend_test)
np.sqrt(mean_squared_error(Ywkend_test, preds))
# RMSE = 754.08
# Much worse at predicting weekend ridership

# Plot the residuals across the range of predicted values
resid = np.ravel(preds) - np.ravel(Ywkend_test)
plt.scatter(preds, resid, alpha=0.7)
plt.xlabel("Predicted Ridership per Train (Weekends)")
plt.ylabel("Residuals")
plt.show()
# Got most of the points right but some points are pretty off driving up the RMSE

# Evaluate the model fit based off of cross validation
cvrmse(X_wkend, Y_wkend, wnlm)
# RMSE from cross validation = 710.43 - slightly worse than gradient boosting

#-- This double model performs better than the full dataset model
#-- However, the lower sample size for weekend will cause issues with how
#-- well that model will perform more generally

##########################################################################

#=======================================================#
# SPLIT OFF DATASET WHERE CABI EXISTED
#=======================================================#

# SPLIT OFF DATASET WHERE CAPITAL BIKESHARE EXISTED
dcabi = trim_data[trim_data['Registered'] > 0]
len(dcabi)  # 1559 obs
dcabi.describe()

# Make sure the min year is 2010 and max year is 2014
assert min(dcabi.Year) == 2010
assert max(dcabi.Year) == 2014

X_cabi = dcabi[feats]
Y_cabi = dcabi['RidersPC']

# Feature selection
feat_sel(X_cabi,Y_cabi)
# Both capital bikeshare variables have much more significance now

#=======================================================#
# SPLIT INTO TRAIN/TEST (CABI)
#=======================================================#

Xcabi_train, Xcabi_test, Ycabi_train, Ycabi_test = train_test_split(X_cabi, Y_cabi, test_size=0.3, random_state=1)


#=======================================================#
# MODELING / MODEL EVAL (CABI)
#=======================================================#

#-------- Linear regression model -------------#
cblm = LinearRegression()
cblm.fit(Xcabi_train, Ycabi_train)

cblm.intercept_
cblm.coef_

cvrmse(X_cabi,Y_cabi,cblm)
# RMSE = 1084.02 - not that much better performing than the really simple model

#-------- Gradient boosting model -------------#
cbgbm = ensemble.GradientBoostingRegressor()
cbgbm.fit(Xcabi_train, Ycabi_train)
cbgbm.score(Xcabi_train, Ycabi_train)
# R^2 = 0.836

cbgbm.score(Xcabi_test,Ycabi_test)
# R^2 = 0.679 - performs much better
preds = cbgbm.predict(Xcabi_test)
np.sqrt(mean_squared_error(Ycabi_test, preds))
# RMSE = 760.41 - That is pretty good considering it is the trim dataset
