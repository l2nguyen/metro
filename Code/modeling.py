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

#==================#
# DEFINE FUNCTIONS #
#==================#


def feat_sel(feats, resp):  # Function to run feature selection
    std_feats = scale(feats)  # Standardize features
    std_resp = scale(resp)  # Standardize response variable
    # Run f-test for feature selection
    ftest = f_regression(std_feats, std_resp)
    feat_sel = pd.DataFrame(data=ftest[0], index=feats.columns.values, columns=['F_score'])  # makes data frame with p values
    feat_sel['p_values'] = ftest[1]  # Make p-values column
    return feat_sel  # print out feature selection


def plot_residuals(actual, predictions):  # Function to plot predictions
    residuals = predictions - actual
    plt.scatter(actual, residuals, alpha=0.7)
    plt.xlabel("Predicted Ridership per Train")
    plt.ylabel("Residuals")
    plt.show()
    return plt


def cvrmse(X,Y,lm):  # Function to get rmse for cross validation
    scores = cross_val_score(lm, X, Y, cv=20, scoring='mean_squared_error')
    return np.mean(np.sqrt(-scores))


def ttcompare(feat_test, resp_test, feat_train, resp_train, lm):  # Plot train/test residuals
    plt.scatter(lm.predict(feat_train), lm.predict(feat_train) - resp_train, c='b', s=30, alpha=0.6)
    plt.scatter(lm.predict(feat_test), lm.predict(feat_test) - resp_test, c='g', s=30, alpha=0.6)
    plt.title('Residual plot using train (blue) and test (green) data')
    plt.ylabel('Residuals')
    plt.xlabel('Predicted Value')
    plt.show()

#=============#
# IMPORT DATA #
#=============#

data = pd.read_csv('model_data.csv')
data.columns.values

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
#- They all fall under the following categories:
#- * Christmas day
#- * Extreme weather conditions (Snowmageddon, Hurricane Sandy)
#- * One day from Obama's inauguration

# Trim the two data points from Hurricane Sandy and Obama's inauguration
trim_data = data[(abs(data['RidersPC_z']) < 3.5)]

# Define feature and response variables
feats = list(data.columns.values[5:-2])
feats.remove('Cars')
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

# R squared values
rtr.score(X_train,Y_train)
# R^2 = 0.868 - Pretty good
rtr.score(X_test,Y_test)
# R^2 = 0.267 - it does not appear this current model is generalizable
rtr.score(X,Y)
# R^2 = 0.696

preds = rtr.predict(X_test)
np.sqrt(mean_squared_error(Y_test, preds))
# 1146.71 - only mildly better than the linear regression model

# Plot residuals
resid = preds - np.ravel(Y_test)
plt.scatter(preds, resid, alpha=0.7)
plt.xlabel("Predicted Riders per Train")
plt.ylabel("Residuals")
plt.show()
#- The residuals seem to have a linear relationship

#=======================================================#
# GRADIENT BOOSTING MODEL (TRIMMED DATA)
#=======================================================#

gbm = ensemble.GradientBoostingRegressor()
gbm.fit(X_train, Y_train)

# R squared values
gbm.score(X_train, Y_train)
# R^2 is 0.454 - worse than random forest model
gbm.score(X_test,Y_test)
# R^2 = 0.356 - better than RF at predicting unseen data
gbm.score(X,Y)
# R^2 = 0.430

preds = gbm.predict(X_test)
np.sqrt(mean_squared_error(Y_test, preds))
# RMSE = 1074.53 - performs better than random forest and linear regression model

# Look at feature importance in the two models
f_select = pd.DataFrame(data=rtr.feature_importances_, index=X.columns.values, columns=['RF'])
f_select['GBM'] = gbm.feature_importances_
f_select
#- In general, they agree about which features are important.

#==========================================================#
# LINEAR REGRESSION MODEL (COMPLETE TRIMMED DATASET)
#==========================================================#
# NOTE: This uses all the features
alm = LinearRegression()
alm.fit(X_train, Y_train)

alm.intercept_
alm.coef_

#============#
# MODEL EVAL #
#============#

# Evaluate the fit of the model based off of the training set
assert not np.any(np.isnan(X_test) | np.isinf(X_test))  # Just to be safe
pred = alm.predict(X_test)
np.sqrt(mean_squared_error(Y_test,preds))
# RMSE = 1159.95
# Not bad. It's about equivalent to the standard deviation of the Riders per car variable

# R squared values
alm.score(X_train, Y_train)
# R^2 = 0.251
alm.score(X_test,Y_test)
# R^2 = 0.253
alm.score(X,Y)
# R^2 = 0.251

plot_residuals(Y_test,preds)
# Looking at the graph, it appears that the residuals fall more on the negative side
# This model seems to lean towards predicting inaccurately on the lesser side than the higher side

# Plot train/test residuals
ttcompare(X_test,Y_test,X_train,Y_train,alm)
# Equally wrong for both

# Evaluate the model fit based off of cross validation
cvrmse(X,Y,alm)
# RMSE = 1180.55

#==========================================================#
# FEATURE SELECTION (TRIMMED DATASET) - LINEAR REGRESSION
#==========================================================#

feat_sel(X_train,Y_train)  # Run feature selection on trimmed dataset
#-- Let's put the threshold at F_score of 40 (completely arbitrary).

# Use only feats that have an F-score over the threshold
new_feats = ['TMAX', 'TMIN', 'SNWD', 'Gas_Price', 'Labor Force', 'Employment', 'Registered', 'Holiday', 'Month_1', 'Month_4', 'Month_6', 'Month_12']

X_new = trim_data[new_feats]
Y_new = trim_data[resp]

#=======================================================#
# SPLIT INTO TRAIN/TEST (TRIMMED DATASET)
#=======================================================#
# all features
# with signficant features
Xnew_train, Xnew_test, Ynew_train, Ynew_test = train_test_split(X_new, Y_new, test_size=0.3, random_state=1)

#==========================================================#
# LINEAR REGRESSION MODEL (COMPLETE TRIMMED DATASET)
#==========================================================#
# NOTE: This model uses only significant features

# Linear regression model
fslm = LinearRegression()
fslm.fit(Xnew_train, Ynew_train)

fslm.intercept_
fslm.coef_

#============#
# MODEL EVAL #
#============#

# Evaluate the fit of the model based off of the training set
assert not np.any(np.isnan(Xnew_test) | np.isinf(Xnew_test))  # Just to be safe
preds = fslm.predict(Xnew_test)
np.sqrt(mean_squared_error(Ynew_test,preds))
# RMSE = 1248.77
# Not bad. It's about equivalent to the standard deviation of the Riders per car variable

# R squared values
fslm.score(Xnew_train, Ynew_train)
# R^2 = 0.146
fslm.score(Xnew_test,Ynew_test)
# R^2 = 0.136
fslm.score(X_new,Y_new)
# R^2 = 0.143

plot_residuals(Ynew_test,preds)
# Looking at the graph, it appears that the residuals fall more on the negative side
# This model seems to lean towards predicting inaccurately on the lesser side than the higher side

ttcompare(Xnew_test,Ynew_test,Xnew_train,Ynew_train,fslm)

# Evaluate the model fit based off of cross validation
cvrmse(X_new,Y_new,fslm)
# RMSE = 1256.35

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
# RMSE = 1285.39

# Plot the residuals across the range of predicted values
plot_residuals(Yh_test,preds)
# Predictions very scattered. Small sample size makes for a not very good model.

# Evaluate the model fit based off of cross validation
cvrmse(X_holiday,Y_holiday,holm)
# RMSE from cross validation = 1549.11

#===================#
# REGULAR DAY MODEL
#===================#
# Evaluate the fit of the model based off of the training set
assert not np.any(np.isnan(Xr_test) | np.isinf(Xr_test))  # Just to be safe
preds = relm.predict(Xr_test)
np.sqrt(mean_squared_error(Yr_test, preds))
# RMSE = 1148.61

# Plot the residuals across the range of predicted values
plot_residuals(Yr_test,preds)
# Model more likely to predict fewer riders than there should be
# Similar to the residual plots of the model with no split at all

# Evaluate the model fit based off of cross validation
cvrmse(X_regular,Y_regular,relm)
# RMSE from cross validation = 1148.39

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
# R^2 = 0.695
wgr.score(Xwkday_test, Ywkday_test)  # Test sets R^2
# R^2 = 0.546
wgr.score(X_wkday, Y_wkday)
# R^2 = 0.644

preds = wgr.predict(Xwkday_test)
np.sqrt(mean_squared_error(Ywkday_test,preds))
# RMSE = 671.17 - better than linear regression

#------ WEEKEND MODEL --------#
wngr.score(Xwkend_train, Ywkend_train)  # Train sets R^2
# R^2 = 0.708
wngr.score(Xwkend_test, Ywkend_test)  # Test sets R^2
# R^2 = 0.412
wngr.score(X_wkend, Y_wkend)
# R^2 = 0.62

preds = wngr.predict(Xwkend_test)
np.sqrt(mean_squared_error(Ywkend_test,preds))
# RMSE = 604.65 - better than linear regression

#=======================================================#
# FEATURE SELECTION (WEEKDAY/WEEKEND)
#=======================================================#

feat_sel(X_wkday, Y_wkday)
# About the same as the full dataset but CaBi seems to have no importance

feat_sel(X_wkend, Y_wkend)
# Employment variables less important. CaBi more important
# Holiday has very little effect on weekend ridership

# Weekday features (thresdhold: F-score>40)
wkday_feats = ['SNWD', 'TMAX', 'TMIN', 'Gas_Price', 'Labor Force', 'Employment', 'Unemployment', 'Holiday', 'Registered', 'Month_1', 'Month_4', 'Month_6', 'Month_7', 'Month_12']
wkend_feats = ['TMAX', 'TMIN', 'Gas_Price', 'Labor Force', 'Employment', 'Holiday', 'Casual', 'Month_1', 'Month_4', 'Month_12']

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
# RMSE = 749.19
# Performs better than the model of the full dataset

# R squared values
wlm.score(Xwkday_train, Ywkday_train)
# R^2 = 0.450
wlm.score(Xwkday_test,Ywkday_test)
# R^2 = 0.455
wlm.score(X_wkday,Y_wkday)
# R^2 = 0.452

# Plot the residuals across the range of predicted values
plot_residuals(Ywkday_test,preds)
# Got most of the points right but some points are pretty off driving up the RMSE
# Those points in the linear model down there are probably holidays

# Plot train/test residuals for weekday model
ttcompare(Xwkday_test,Ywkday_test,Xwkday_train,Ywkday_train,wlm)

# Evaluate the model fit based off of cross validation
cvrmse(X_wkday,Y_wkday,wlm)
# RMSE from cross validation = 695.08 - slightly worse than gradient boosting

#===============#
# WEEKEND MODEL
#===============#

# Evaluate the fit of the model based off of the training set
assert not np.any(np.isnan(Xwkend_test) | np.isinf(Xwkend_test))  # Just to be safe
preds = wnlm.predict(Xwkend_test)
np.sqrt(mean_squared_error(Ywkend_test, preds))
# RMSE = 618.12
# Much worse at predicting weekend ridership

# R squared values
wnlm.score(Xwkend_train, Ywkend_train)
# R^2 = 0.278
wnlm.score(Xwkend_test,Ywkend_test)
# R^2 = 0.366
wnlm.score(X_wkend,Y_wkend)
# R^2 = 0.304

# Plot the residuals across the range of predicted values
plot_residuals(Ywkend_test,preds)
# Got most of the points right but some points are pretty off driving up the RMSE

# Evaluate the model fit based off of cross validation
cvrmse(X_wkend, Y_wkend, wnlm)
# RMSE from cross validation = 662.37 - slightly worse than gradient boosting

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
# The months now have less significance

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
# RMSE = 1013.61 - not that much better performing than the really simple model

#-------- Gradient boosting model -------------#
cbgbm = ensemble.GradientBoostingRegressor()
cbgbm.fit(Xcabi_train, Ycabi_train)
cbgbm.score(Xcabi_train, Ycabi_train)
# R^2 = 0.843

cbgbm.score(Xcabi_test,Ycabi_test)
# R^2 = 0.676 - performs much better that complete dataset
preds = cbgbm.predict(Xcabi_test)
np.sqrt(mean_squared_error(Ycabi_test, preds))
# RMSE = 763.62 - That is pretty good considering this is the complete trim dataset
