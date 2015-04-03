# Bikeshare data

import pandas as pd

data = '/Users/Zelda/Data Science/GA/Project/Data/Capital Bikeshare/2014-Q3-Trips-History-Data.csv'

data = pd.read_csv(data)

data.dtypes
data.head(20)

data['Start date'] = pd.to_datetime(data['Start date'], format="%m/%d/%Y %H:%M")
data.set_index('Start date', inplace=True)
data.sort_index(inplace=True)
## !! Error: time data '2014-10-01 00:00' does not match format '%m/%d/%Y %H:%M'

# data['Start time'] = pd.to_datetime(data['Start time'], format="%m/%d/%Y %H:%M")
# data.set_index('Start time', inplace=True)

data['Year'] = data.index.year
data['Month'] = data.index.month
data['Day'] = data.index.day

# data.groupby('Month')['Member Type'].apply(pd.value_counts)
# data.groupby('Month')['Type'].apply(pd.value_counts)
# data.groupby('Month')['Bike Key'].apply(pd.value_counts)
# data.groupby('Month')['Subscriber Type'].apply(pd.value_counts)
grouped = data.groupby(['Month','Day'])['Subscription Type'].apply(pd.value_counts)
##?? How to get a new data frame with the a column each counting the number of
##  casual and registered rider per day