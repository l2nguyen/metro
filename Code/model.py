#===========================================================#
# METRO RIDERSHIP ANALYSIS - MODELING
#
# NOTE: Project for GA Data Science class.
# I'm interested in looking at the factors that affect
# Metro rail ridership and modeling their relationships
#===========================================================#
# CREATED BY: Lena Nguyen - April 7, 2015
#===========================================================#

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import cross_val_score

#=============#
# IMPORT DATA #
#=============#

data = read.csv('../Data/model_data.csv')

#=======================================================#
# LINEAR REGRESSION MODEL (MODEL #1)
#=======================================================#
# Split data into weekday and weekends (Temporary)
# ! Might try standardizing the data based on number of cars later
weekday = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']

wkday = data[data.Weekday.isin(weekday)]
wkend = data[~(data.Weekday.isin(weekday))]

#-- 1. Run very simple model for weekday ridership with precipitation/snowfall variables
#-- FEATURES: Precipitation, Snow depths, Snow fall amount

# Plot the Precipitation against daily Ridership
plt.figure(figsize=(10,9))
plt.subplot(131)
plt.scatter(wkday.PRCP, wkday.Riders, color='b', alpha=0.8)  # Plot the raw data
plt.xlabel("Precipitation (inches)")
plt.ylabel("Ridership")

# Plot the Snow Depths against daily Ridership
plt.subplot(132)
plt.scatter(wkday.SNWD, wkday.Riders, color='g', alpha=0.8)  # Plot the raw data
plt.xlabel("Snow Depths (inches)")

# Plot the Snowfall against daily Ridership
plt.subplot(133)
plt.scatter(wkday.SNOW, wkday.Riders, color='r', alpha=0.8)  # Plot the raw data
plt.xlabel("Snowfall (inches)")


feats = ['PRCP','SNWD','SNOW']

X = wkday[feats]
Y = wkday['Riders']

# Split into train and test datasets
# Test is 30% of complete dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

# Convert test/train data to data fame
X_train = pd.DataFrame(data=X_train, columns=feats)
X_test = pd.DataFrame(data=X_test, columns=feats)
Y_train = pd.DataFrame(data=Y_train, columns=['Riders'])
Y_test = pd.DataFrame(data=Y_test, columns=['Riders'])

# Simple Linear Regression model
plm = LinearRegression()
plm.fit(X_train, Y_train)

plm.intercept_
plm.coef_

#============#
# MODEL EVAL #
#============#

# Evaluate the fit of the model based off of the training set
assert not np.any(np.isnan(X_test)|np.isinf(X_test))  # Just to be safe
preds = plm.predict(X_test)
np.sqrt(mean_squared_error(Y_test,preds))
# That is a pretty bad mean square error even for the scale we are working with

# Plot the residuals across the range of predicted values
resid = preds - Y_test['Riders']
plt.scatter(preds, resid, alpha=0.7)
plt.xlabel("Predicted Ridership")
plt.ylabel("Residuals")
#-- It appears that this model predicts ridership to be about 700,000 for the
#-- majority of cases. This is about the value of the intercept. 

# Evaluate the model fit based off of cross validation
scores = cross_val_score(plm, X, Y, cv=10, scoring='mean_squared_error')
np.mean(np.sqrt(-scores))
# RMSE is about equivalent to the standard deviation of weekday ridership

#-- CONCLUSION: Not a very good model at all. Maybe because there is not much
#-- variation in daily precipitation/snowfall. 

#=======================================================#
# LINEAR REGRESSION MODEL (MODEL #2)
#=======================================================#




