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

#============#
# DATA FILES #
#============#

# Data of daily ridership of metro rail from Open Data DC
# Note: Not disaggregated by station
# Source: http://www.opendatadc.org/dataset/wmata-metrorail-ridership-by-date
metro = '../Data/metro.csv'

# Data of rail ridership from May 2013
# This data will be mainly used to see how people are moving throught the Metrorail system
# Source: http://planitmetro.com/2014/08/28/data-download-may-2013-2014-metrorail-ridership-by-origin-and-destination/
may = '../Data/Metro_May_2013_Data.csv'

# Monthly data of gas prices in Lower Atlantic Region from EIA
# Source URL: http://www.eia.gov/dnav/pet/pet_pri_gnd_dcus_r1z_m.htm
gas = '../Data/Gas Prices.csv'

# Monthly Unemployment data for the DC metro area, not adjusted for seasonality
# Source: http://www.bls.gov/eag/eag.dc_washington_md.htm
labor = '../Data/Unemployment.csv'

#=======================================================#
# METRO DATA 
#=======================================================#

#===========#
# READ DATA #
#===========#

# OPEN DATA DC METRO RIDERSHIP DATA
metro = pd.read_csv(metro)

# QUICK LOOK
metro.head(10)
metro.describe()
metro.dtypes

#======================#
# CLEAN/TRANSFORM DATA #
#======================#

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

############################################

#=============================#
# MAY 2013 ENTRANCE/EXIT DATA #
#=============================#
# NOTE: This dataset will help us see how people are moving throughout 
# the metrorail system in different times of days
# For example, is everyone coming from the MD/VA suburbs in the morning to DC?
# Not very important for the regression but interesting data

# NOTE: Late night data only for Saturday night (Labeled as Sunday).
# The number of riders seem too small from personal experience
# Ignore late night data completely 

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

############################################

#=======================================================#
# GAS PRICE DATA 
#=======================================================#

#===========#
# READ DATA #
#===========#

gas = pd.read_csv(gas)
gas.columns = ['Date', 'Price']  # Rename columns

# QUICK LOOK
gas.head(10)
gas.describe
gas.dtypes

#======================#
# CLEAN/TRANSFORM DATA #
#======================#

# Round price column to 2 decimal places to look like dollar prices
gas['Price'] = np.round(gas['Price'], decimals=2)

gas['Date'] = pd.to_datetime(gas.Date, format='%m/%d/%Y')
gas.dtypes
gas.set_index('Date', inplace=True)

# Year
gas['Year'] = gas.index.year
# Month
gas['Month'] = gas.index.month
gas['Month'] = gas.Month.astype('int')
# Quarterly assignment
gas['Quarter'] = [((x-1)//3)+1 for x in gas['Month']]

gas.isnull().sum()
# No missing data

#============#
# GRAPH DATA #
#============#

# General trend of gas price over the years (1993-2015)
gas.groupby('Year').Price.mean().plot( kind='line',
                                        color='b',
                                        linewidth=2,
                                        title='Average Gas Price by Year')
plt.xlabel('Year')
plt.ylabel('Average gas price (USD/Gallon)')
plt.axis([1993, 2014, 0.50, 4.00])
plt.savefig('Average Gas Price by Year.png')
## Graph shows continuous price increase since 1993, a short slight dip around 2009,
## Gas prices have been going down since Fall 2014

# See if there are seasonal differences in gas prices (Unlikely)
gas.groupby(['Quarter', 'Year']).Price.mean().unstack(0).plot(kind='bar',
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
data = pd.merge(metro, gas, on=['Year', 'Month'])  # Merge in gas data

####################################

#=======================================================#
# UNEMPLOYMENT DATA 
#=======================================================#

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

# Change month abbreviation into integer
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
## This smooths out the seasonal effects. Unemployment rate doubled in 2008.
## This was due to the financial crisis in 2008 due to the housing bubble bursting

#============#
# MERGE DATA #
#============#

data = pd.merge(data, labor, on=['Year', 'Month'])  # Merge in unemployment data


