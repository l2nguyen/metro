# Bikeshare data processing
import pandas as pd

# Define function to reuse to process date data
# Note: Q4 in 2014 has a different date format. It's %Y-%m-%d instead
def process_date(df,var):
    df[var] = pd.to_datetime(df[var], format='%m/%d/%Y %H:%M')
    df.set_index(var, inplace=True)
    df.sort_index(inplace=True)
    # Separate date into separate variables for month, year, day
    df['Year'] = df.index.year  # Create year variable
    df['Month'] = df.index.month  # Create month variable
    df['Day'] = df.index.day  # Create day variable

year = range(2010, 2015)
quarter = ['Q1', 'Q2', 'Q3', 'Q4']

# Make list of column names
cols = ['Duration', 'Start_date', 'End_date', 'Start_station', 'End_station', 'Bike', 'Type']

# ? Will turn all this into a loop later
data = '/Users/Zelda/Data Science/GA/Project/Data/Archives/CaBi/2011-Q1-Trips-History-Data.csv'
# read data into dataframe
data = pd.read_csv(data, header=True, names=cols)

# Quick look at data
data.dtypes
data.head(10)
data.columns.values

process_date(data,'Start_date')

data.head(10)  # check it worked

grouped = data.groupby(['Month','Day'])['Type'].apply(pd.value_counts)
##?? How to get a new data frame with the a column each counting the number of
##  casual and registered rider per day

grouped.to_csv('/Users/Zelda/Data Science/GA/Project/Data/Archives/CaBi/type.csv')
