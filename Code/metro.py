#---------------
#
#
#---------------
# CREATED BY: Lena Nguyen - March 15, 2015
#---------------


#----------------#
# IMPORT MODULES #
#----------------#
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import datetime

#------------#
# DATA FILES #
#------------#
metro = '/Users/Zelda/Data Science/GA/Project/Data/2014 10 Year Historical Rail Ridership.csv'
metro2 = '/Users/Zelda/Data Science/GA/Project/Data/metro.csv'
gas = '/Users/Zelda/Data Science/GA/Project/Data/Gas Prices.csv'

#-------------#
# IMPORT DATA #
#-------------#
metro = pd.read_csv(metro2)

gas = pd.read_csv(gas)
gas.columns=['Date','Price'] # Rename columns

#------------#
# QUICK LOOK #
#------------#
metro.head(10)
metro.describe()
metro.dtypes

gas.head(10)
gas.describe
gas.dtypes

#----------------------------#
# CLEAN/TRANSFORM METRO DATA #
#----------------------------#

#-- TRANSFORM DATES --#

# NOTE: More interested in weekday versus weekend ridership
# So I need to transform each into its corresponding day of the week
metro['Date'] = pd.to_datetime(metro.Date, format='%Y-%m-%d')
metro.dtypes
metro.set_index('Date', inplace=True)

# Year
metro['Year']=metro.index.year
# Month
metro['Month']=metro.index.month
# Day
metro['Day'] = metro.index.day

# Label day of week
metro['Weekday']=metro.index.weekday # Creates integer for day of week
# maps the integer to names
metro['Weekend'] = metro.Weekday.isin([4,5])  

metro['Weekday'] = metro.Weekday.map({  0:'Mon', 1:'Tue', 2:'Wed', 
                                        3:'Thu', 4:'Fri', 5:'Sat', 
                                        6:'Sun'})



# Round price column to 2 decimal places to look like dollar prices
gas['Price'] = np.round(gas['Price'],decimals=2) 

gas['Date'] = pd.to_datetime(gas.Date, format='%m/%d/%Y')
gas.dtypes
gas.set_index('Date', inplace=True)

# Year
gas['Year']=gas.index.year
# Month
gas['Month']=gas.index.month

# Check for missing data in variables                                        
 metro.isnull().sum() 
#--> No missing data
#------------------#
# DATA INSPECTION  #
#------------------#                          

#-------------#
# METRO DATA  #
#-------------#

# General trend of ridership over the years                           
metro.groupby('Year').Total.mean().plot(kind='line', 
                                        color='r', 
                                        linewidth=2, 
                                        title='Average Metro Ridership by Year')
 
# General trend of ridership by days of the week                                   
metro.groupby(['Year','Weekday']).Total.mean().unstack(0).plot(kind='line', 
                                                                title='Average Metro Ridership by Day of the Week')                              
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) # moves legend to the side

## Box plot of the same graph as above
# Shows average ridership by day of the week
metro.boxplot(column='Total', by='Weekday')
plt.xlabel('Day of Week')
plt.ylabel('Average number of riders')
plt.title('Average number of riders by Day')

# Looking at ridership by month to see if there's any seasonal variation
# Hard to look at but you get the general trend. There's gotta be a prettier way
metro.groupby(['Year', 'Month']).Total.mean().unstack(0).plot( kind = 'bar')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# Maybe easier to see by line graph?
metro.groupby(['Year','Month']).Total.mean().unstack(0).plot(kind='line', 
                                                            title='Average Metro Ridership by Month') 
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#--> Mildly easier but not by much. Can see the general trend

# Effects of Snowmageddon (February 2010)
metro_snow= metro[(metro.Year.isin([2010])) & (metro.Month == 2)]

metro_snow.groupby(['Year', 'Day']).Total.sum().unstack(0).plot( kind = 'bar',
                                                                color='c',
                                                                figsize=(7,9),
                                                                title='Effects of Snow Day on Metro Ridership')
plt.legend().set_visible(False) # Hides legend
#--> ? How to turn the bars different lines different colors to highlight?

# Look at a regular month (May 2010)
metro_may= metro[(metro.Year.isin([2010])) & (metro.Month == 5)]

metro_may.groupby(['Year', 'Day']).Total.sum().unstack(0).plot( kind = 'bar',
                                                                color='r',
                                                                figsize=(7,9),
                                                                title='Daily Ridership in May 2010')
plt.legend().set_visible(False) # Hides legend

#-----------------#
# GAS PRICE DATA  #
#-----------------#

# General trend of gas price over the years (1993-2015)                         
gas.groupby('Year').Price.mean().plot(kind='line', 
                                    linewidth=2, 
                                    title='Average Gas Price by Year')
plt.xlabel('Year')
plt.ylabel('Average gas price')

