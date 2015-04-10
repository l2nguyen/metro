#===========================================================#
# METRO RIDERSHIP ANALYSIS
#
# NOTE: Project for GA Data Science class.
# I'm interested in looking at the factors that affect
# Metro rail ridership and modeling their relationships
#===========================================================#
# CREATED BY: Lena Nguyen - March 15, 2015
#===========================================================#

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import cross_val_score

#=======================================================#
# METRO DATA 
#=======================================================#

#============#
# DATA FILES #
#============#

# Data of daily ridership of metro rail from Open Data DC
# Note: Not disaggregated by station
# Source: http://www.opendatadc.org/dataset/wmata-metrorail-ridership-by-date
# Data scraped from visualization: http://planitmetro.com/ridership_cal/
metro = '../Data/metro.csv'

# Data of rail ridership from May 2013
# This data will be mainly used to see how people are moving throught the Metrorail system
# Source: http://planitmetro.com/2014/08/28/data-download-may-2013-2014-metrorail-ridership-by-origin-and-destination/
may = '../Data/Metro_May_2013_Data.csv'

#===========#
# READ DATA #
#===========#

# OPEN DATA DC METRO RIDERSHIP DATA
metro = pd.read_csv(metro, header=True, names=['Date','Riders'])

# QUICK LOOK
metro.head(10)
metro.describe()
metro.dtypes

#======================#
# CLEAN/TRANSFORM DATA #
#======================#

# CLEAN DATA
metro[metro['Riders']==0]
#-- One value of 0 for October 29, 2012. Is this correct?
#-- Google of the date shows that the value is correct
#-- Whole metrorail system was shut down due to Hurricane Sandy on that day
#-- https://www.wmata.com/about_metro/news/PressReleaseDetail.cfm?ReleaseID=5362
#-- http://www.wmata.com/about_metro/news/PressReleaseDetail.cfm?ReleaseID=5363

# TRANSFORM DATES

# NOTE: More interested in weekday versus weekend ridership
# So I need to transform each into its corresponding day of the week
metro['Date'] = pd.to_datetime(metro.Date, format='%Y-%m-%d')
metro.dtypes
metro.set_index('Date', inplace=True)

# Year
metro['Year'] = metro.index.year
# Month
metro['Month'] = metro.index.month
# Day
metro['Day'] = metro.index.day

# Label day of week
metro['Weekday'] = metro.index.weekday  # Creates integer for day of week

weekend = metro[metro.Weekday.isin([5, 6])]
weekday = metro[~(metro.Weekday.isin([5, 6]))]

# Map the integer to names
metro['Weekday'] = metro.Weekday.map({  0:'Mon', 1:'Tue', 2:'Wed',
                                        3:'Thu', 4:'Fri', 5:'Sat',
                                        6:'Sun'})
                                        
#============#
# GRAPH DATA #
#============#

# General trend of ridership over the years
metro.groupby('Year').Total.mean().plot(kind='line',
                                        color='r',
                                        linewidth=2,
                                        title='Average Metro Ridership by Year')
plt.savefig('Average Ridership by Year.png')  # save plot to file

metro.groupby('Month').Total.mean().plot(kind='line',
                                        color='c',
                                        linewidth=2,
                                        title='Average Metro Ridership by Month')
plt.savefig('Average Ridership by Month.png')  # save plot to file

# General trend of ridership by days of the week
metro.groupby(['Year','Weekday']).Total.mean().unstack(0).plot(kind='line',
                                                                title='Average Metro Ridership by Day of the Week')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)  # moves legend to the side
plt.savefig('Average Ridership by Day of Week.png')

# Box plot of the same graph as above
# Shows average ridership by day of the week
metro.boxplot(column='Total', by='Weekday', sym=' ')
plt.xlabel('Day of Week')
plt.ylabel('Average number of riders')
plt.title('Average number of riders by Day')
plt.savefig('Average Ridership by Day of Week (Boxplot).png')

# Looking at ridership by month to see if there's any seasonal variation
# Hard to look at but you get the general trend. There's gotta be a prettier way
metro.groupby(['Year', 'Month']).Total.mean().unstack(0).plot(kind='bar')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# Line graph of average metro ridership by month
metro.groupby(['Year', 'Month']).Total.mean().unstack(0).plot(kind='line',
                                                            title='Average Metro Ridership by Month')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# Mildly easier but not by much. Can see the general trend

# Effects of Snowmageddon (February 2010)
metro_snow = metro[(metro.Year.isin([2010])) & (metro.Month == 2)]


metro_snow.groupby(['Year', 'Day']).Total.sum().unstack(0).plot(kind='bar',
                                                                color='c',
                                                                figsize=(7, 9),
                                                                title='Effects of Snow Day on Metro Ridership')
plt.legend().set_visible(False)  # Hides legend
plt.savefig('Ridership for February 2010.png')
# ? How to turn the bars different lines different colors to highlight?

# Look at a regular month (July 2010)
metro_july = metro[(metro.Year.isin([2010])) & (metro.Month == 7)]

metro_july.groupby(['Year', 'Day']).Total.sum().unstack(0).plot(kind='bar',
                                                                color='r',
                                                                figsize=(7,9),
                                                                title='Daily Ridership in July 2010')
plt.legend().set_visible(False)  # Hides legend
plt.savefig('Ridership for July 2010.png')


# Graph weekend and weekday ridership on the same line
weekday.groupby('Month').Total.mean().plot(kind='line',
                                            color='b',
                                            linewidth=2)
weekend.groupby('Month').Total.mean().plot(kind='line',
                                            color='r',
                                            linewidth=2)
plt.xlabel('Month')
plt.ylabel('Average Number of Riders')
plt.title('Weekend and Weekday Ridership by Month')
plt.axis([1, 12, 100000, 800000])
plt.savefig('Weekday vs weekend Ridership.png')

#########################################################################

#=============================#
# MAY 2013 ENTRANCE/EXIT DATA #
#=============================#
#-- NOTE: This dataset will help us see how people are moving throughout 
#-- the metrorail system in different times of days
#-- For example, is everyone coming from the MD/VA suburbs in the morning to DC?
#-- Not very important for the regression but interesting data

#-- NOTE: Late night data only for Saturday night (Labeled as Sunday).
#-- The number of riders seem too small from personal experience
#-- I plan to ignore late night data completely. Does not seem reliable.

# MAY 2013 RIDERSHIP DATA
may13 = pd.read_csv(may, header=True, 
                    names=['Holiday', 'Day', 'Entrance', 'Exit', 'Period', 'Riders'])

# QUICK LOOK
may13.head(10)
may13.describe()
may13.dtypes

#======================#
# CLEAN/TRANSFORM DATA #
#======================#

# Transform Riders column from object to integer
may13['Riders'] = may13['Riders'].map(lambda x: x.replace(',', ''))  # removes the , from the numbers
may13['Riders'] = may13.Riders.astype('int')  # makes them all ints
may13['Riders'].describe()

#============#
# GRAPH DATA #
#============#

may13.groupby(['Period']).Riders.sum().plot( kind='line',
                                            color='r',
                                            figsize=(7,9),
                                            title='Total Daily Ridership in May 2013')
plt.legend().set_visible(False)  # Hides legend

# AM PEAK
ampeak=may13[may13.Period.isin(['AM Peak'])]  # Make a data frame with the AM Peak obs

# Make a stacked bar graph with the number of riders by Entrance and Exit Stations
ampeak.groupby(['Exit']).Riders.sum().plot(kind='bar', color='b')
ampeak.groupby(['Entrance']).Riders.sum().plot(kind='bar', color='r')
plt.title('Number of Riders Entering/Exiting at Station during AM Peak in May 2013')
plt.legend()

# PM PEAK
pmpeak=may13[may13.Period.isin(['PM Peak'])]  # Make a data frame with the PM Peak obs
# Make a stacked bar graph with the number of riders by Entrance and Exit Stations

pmpeak.groupby(['Exit']).Riders.sum().plot(kind='bar', color='m', label='Exit')
pmpeak.groupby(['Entrance']).Riders.sum().plot(kind='bar', color='g', label='Entrance')
plt.title('Number of Riders Entering/Exiting at Station during PM Peak in May 2013')
plt.legend()

#########################################################################

#=======================================================#
# WEATHER DATA 
#=======================================================#

# Daily weather data
# Source: NOAA, Washington Reagan National Airport Weather Station
weather = '../Data/Weather - Daily.csv'

# IMPORTANT: Variables are all measured in METRIC units. 
# PRCP/SNOW is measured in millimiters .
# TMAX/TMIN are in tenths of degrees celsius

#===========#
# READ DATA #
#===========#

weather = pd.read_csv(weather)

# QUICK LOOK
weather.columns.values
weather.head(10)
weather.dtypes
weather.describe()
#-- 411 degrees Celsius (~771 deg F/hotter than hell) seems a bit high to be a max temp.
#-- TMAX and TMIN variable have to be divided by 10 to get degrees Celsius
#-- That must be what tenths of degrees celsius in documentation means

#======================#
# CLEAN/TRANSFORM DATA #
#======================#
# Documentation says that "-9999" means it's a missing value
# Will transform those to NaN values to see how many missing values there actually are
cols = list(weather.columns.values)
for col in cols:
    weather[col][weather[col]==-9999] = np.nan

weather.isnull().sum()  # check number of missing values
#-- WT columns have a 1 for a day when a certain weather type occurs
#-- Most weather don't occur that often
#-- Surprisingly, snow (WT18) occurs only 264 days from 2004-2014. Hail (WT05) occurs more often at 364 days
#-- Most common are Fog (WT01), Rain (WT16), Mist (WT13), Haze (WT08) 

#-- Will drop some (maybe all?) WT columns
#-- For now, keep Hail (WT05), Rain (WT16), Snow (WT18), Thunder (WT03)

# REMOVE SOME COLUMNS
weather.drop(weather.columns[:2], axis=1, inplace=True)  # Removes Station/Station Name columns
weather.drop(weather.columns[6:14], axis=1, inplace=True)  # Removes WT columns

weather.columns.values  # assess the damage

weather.drop(weather.columns[7:12], axis=1, inplace=True)  # Removes more WT columns
weather.columns.values  # assess the damage
weather.drop(weather.columns[-2:], axis=1, inplace=True)  # Remove more WT columns
del weather['WT08']  # Last manual removal

weather.columns.values  # assess the damage

# CHANGE MISSING VALUES TO 0
# This will make the WT variables binary variables
cols = list(weather.columns.values)
for col in cols:
    weather[col][np.isnan(weather[col])] = 0
    
weather.isnull().sum()  # No missing values
weather.describe()

# SEPARATE DATE INTO YEAR, MONTH, DAY
weather['DATE'] = pd.to_datetime(weather.DATE, format='%Y%m%d')
weather.set_index('DATE', inplace=True)

# Year
weather['Year'] = weather.index.year
# Month
weather['Month'] = weather.index.month
# Day
weather['Day'] = weather.index.day

# Drop date column
weather.reset_index(drop=True)

# Convert from tenths of degrees Celsius to Fahrenheit for easier understanding
weather['TMAX'] = weather['TMAX'].map(lambda x:((float(9)/5)*(x/10) + 32))  # Highest temp
weather['TMIN'] = weather['TMIN'].map(lambda x:((float(9)/5)*(x/10) + 32))  # Lowest temp

# Convert from mm to inches
weather['PRCP'] = weather['PRCP'].map(lambda x:(x/float(254)))  # Precipitation
weather['SNWD'] = weather['SNWD'].map(lambda x:(x/25.4))  # Snow Depth
weather['SNOW'] = weather['SNOW'].map(lambda x:(x/25.4))  # Snowfall

weather.describe()
# Those max/min temps look much more reasonable

#============#
# GRAPH DATA #
#============#

# Look at data numerically
weather.groupby('Year')['PRCP','SNWD', 'SNOW','TMAX','TMIN'].mean()  # Average by year
weather.groupby('Year')['PRCP','SNWD', 'SNOW','TMAX','TMIN'].max()  # Average by year

# Look at data numerically
weather.groupby('Month')['PRCP','SNWD', 'SNOW','TMAX','TMIN'].mean()  # Average by month
weather.groupby('Month')['PRCP','SNWD', 'SNOW','TMAX','TMIN'].max()  # Average by year

# Scatter matrix (to help check for collinearity)
pd.scatter_matrix(weather[['PRCP','SNWD', 'SNOW']])

# Total rain fail by month
weather.groupby('Month').PRCP.sum().plot(kind='bar',color='g')
plt.xlabel('Month')
plt.ylabel('Total Precipitation Amount (Inches)')
plt.title('Total Precipitation by Month')

weather.groupby('Month').SNWD.mean().plot(kind='bar',color='g')
plt.xlabel('Month')
plt.ylabel('Total Precipitation Amount (Inches)')
plt.title('Total Precipitation by Month')

# Average low/high temperature by month
weather.groupby('Month').TMAX.mean().plot(kind='line', color='b', label='Max Temp')
weather.groupby('Month').TMIN.mean().plot(kind='line', color='r', label='Min Temp')
plt.xlabel('Month')
plt.ylabel('Average Temperature (Deg F)')
plt.axis([1, 12, 0, 100])
plt.legend()

# Record max/min temp from 2004-2014 by month
weather.groupby('Month').TMAX.max().plot(kind='line', color='b', label='Max Temp')
weather.groupby('Month').TMIN.min().plot(kind='line', color='r', label='Min Temp')
plt.xlabel('Month')
plt.ylabel('Temperature (Deg F)')
plt.axis([1, 12, 0, 120])
plt.legend()

#============#
# MERGE DATA #
#============#
data = pd.merge(metro, weather, on=['Year', 'Month', 'Day'])  # Merge metro/weather data

data.columns.values
data.head(10)

##########################################################################

#=======================================================#
# GAS PRICE DATA 
#=======================================================#

# Monthly data of gas prices in Lower Atlantic Region from EIA
# Source URL: http://www.eia.gov/dnav/pet/pet_pri_gnd_dcus_r1z_m.htm
gas = '../Data/Gas Prices.csv'

#===========#
# READ DATA #
#===========#

gas = pd.read_csv(gas)
gas.columns = ['Date', 'Gas_Price']  # Rename columns

# QUICK LOOK
gas.head(10)
gas.describe
gas.dtypes

gas.isnull().sum()
# No missing data

#======================#
# CLEAN/TRANSFORM DATA #
#======================#

# Round price column to 2 decimal places to look like dollar prices
gas['Gas_Price'] = np.round(gas['Gas_Price'], decimals=2)

# Change date into month/year column
gas['Date'] = pd.to_datetime(gas.Date, format='%m/%d/%Y')
gas.dtypes
gas.set_index('Date', inplace=True)

# Year
gas['Year'] = gas.index.year

# Month
gas['Month'] = gas.index.month
# Quarterly assignment
gas['Quarter'] = [((x-1)//3)+1 for x in gas['Month']]

#============#
# GRAPH DATA #
#============#

# General trend of gas price over the years (1993-2015)
gas.groupby('Year').Gas_Price.mean().plot( kind='line',
                                        color='b',
                                        linewidth=2,
                                        title='Average Gas Price by Year')
plt.xlabel('Year')
plt.ylabel('Average gas price (USD/Gallon)')
plt.axis([1993, 2014, 0.50, 4.00])
plt.savefig('Average Gas Price by Year.png')
#-- Graph shows continuous price increase since 1993, a short slight dip around 2009,
#-- Gas prices have been going down since Fall 2014

# See if there are seasonal differences in gas prices (Unlikely)
gas.groupby(['Quarter', 'Year']).Gas_Price.mean().unstack(0).plot(kind='bar',
                                                            figsize=(7,9),
                                                            title='Average Gas Price by Quarter')

plt.xlabel('Year')
plt.ylabel('Average Gas Price (USD/Gallon)')
plt.legend().set_visible(False)  # Hides legend
## As suspected, no seasonal differences in gas prices that occur annually

#============#
# MERGE DATA #
#============#

del gas['Quarter']  # delete quarter variable before merge
data = pd.merge(data, gas, on=['Year', 'Month'])  # Merge in gas data

data.columns.values
data.head(10)

##########################################################################

#=======================================================#
# UNEMPLOYMENT DATA 
#=======================================================#

# Monthly Unemployment data for the DC metro area, not adjusted for seasonality
# Source: http://www.bls.gov/eag/eag.dc_washington_md.htm
labor = '../Data/Unemployment.csv'

#===========#
# READ DATA #
#===========#
labor = pd.read_csv(labor)

# QUICK LOOK
labor.dtypes
labor.head(10)
labor.describe()

#======================#
# CLEAN/TRANSFORM DATA #
#======================#

# Change month abbreviation to integers for easier merge
monthDict = {'Jan':1, 'Feb':2, 'Mar':3, 
             'Apr':4, 'May':5, 'Jun':6, 
             'Jul':7, 'Aug':8, 'Sep':9, 
             'Oct':10, 'Nov':11, 'Dec':12}

labor['Month'] = labor.Month.map(monthDict)  

# Check it worked
labor.head(10)
labor.dtypes

#============#
# GRAPH DATA #
#============#

# General trend of unemployment rate over the years (2000-2015)
labor.groupby('Year').Unemp_Rate.mean().plot( kind='line',
                                                color='b',
                                                linewidth=2,
                                                title='Average Unemployment Rate by Year')
plt.xlabel('Year')
plt.ylabel('Average Unemployment Rate')
plt.savefig('Average Unemployment Rate by Year.png')
#-- This smooths out the seasonal effects. Unemployment rate doubled in 2008.
#-- This was due to the financial crisis in 2008 due to the housing bubble bursting

#============#
# MERGE DATA #
#============#
data = pd.merge(data, labor, on=['Year', 'Month'])  # Merge in unemployment data

data.columns.values
data.head(10)

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




