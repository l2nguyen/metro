# Predicting Metrorail Ridership
##### Lena Nguyen - DAT 6 - May 2, 2015
---
## Project Goal
The [WMATA Metrorail](http://www.wmata.com) services the DC Metropolitan area, which include DC, Fairfax County and Loudon County in Northern Virginia, PG County and Montgomery County in Maryland. The ultimate goal of my project is to accurately forecast ridereship on the Metrorail.

## Data Overview and Sources
The majority of my time was spent thinking about factors that would affect metrorail ridership and where I would get the data I wanted. The process of gathering data involved a lot of searching around the internet for the most apporpriate and publicly available dataset to use. Gathering data for number of tourists proved to be the most challenging. All the data that were free were very aggragated. Since I suspected there is seasonal variation in tourism in DC, I decided not to use this data. I settled on using the Capital Bikeshare (CaBi) data as a proxy for tourism. However, this was also not a good proxy for tourism because it only went back to 2010 (when CaBi started in DC). 

#### Response Variable: Metrorail Riders per Train
I plan to use historic Metrorail ridership data to build and validate my model. Instead of using total riders, I used the number of riders per train as the response variable so that the response variable would be more standardized. Fewer trains run on the weekends and holidays. Conversely, more trains run on some holidays where WMATA thinks there will be more riders (ie July 4th). To construct the riders per train variable, I used data from two sources. 

I used the historic Metrorail Ridership from 2004 to 2014 available on Open Data DC also has [historic ridership data](http://www.opendatadc.org/dataset/wmata-metrorail-ridership-by-date). The number of trains per day was deduced from data about the [frequency of trains](http://www.wmata.com/rail/frequency.cfm). Adjustments in the number of trains were made based on information available on WMATA. In general, the Metrorail run on a Saturday schedule for most holidays so the number of trains were adjusted to the Saturday schedule for simplicity. The spreadsheet with the math can be found [here](http://www.wmata.com/rail/frequency.cfm)

#### Feature Variables
I used the following features in my model:
* Gas prices: When gas prices are low, people will choose to drive instead of using the metrorail. The US Energy Information Administration has [historic gas prices](http://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=EMM_EPMRR_PTE_R1Z_DPG&f=W) available in xls format for the every week from 1995 to present day for regions in the US. DC/MD/VA is part of the Lower Atlantic region.

* Weather: People will be less likely to use the metrorail during inclement weather. NOAA has historic [weather data](http://www.ncdc.noaa.gov/cdo-web/datatools) on their website. I used the weather data from the Reagan National Airport.

* DC employment figures: I hypothesize that the majority of people are using the metrorail to get to and from work so employment figures will have an impact on ridership. The US Bureau of Labor Statistics provide historical [labor force data](http://www.bls.gov/eag/eag.dc_washington_md.htm) for the Washington DC Metro area. 

* Capital Bikeshare (CaBi): The [trip history data](https://www.capitalbikeshare.com/trip-history-data) available on the Capital Bikeshare website was used to find the nubmer of Registtered and Casual riders everyday. Casual riders will serve as a proxy for tourist numbers in DC since many tourists use CaBi to get around the National Mall and other tourist areas. The major problem with using this data is that CaBi started in late 2010 so there will be 6 years worth of data where I will not have tourism data. 

* Federal holidays: Since the majority of people working in DC are employed by the federal government, closure of the federal government will have a large impact on ridership. Data for federal holidays can be found [here](https://catalog.data.gov/dataset/federal-holidays). The data was supplemented with historic status from [OPM](http://www.opm.gov/policy-data-oversight/snow-dismissal-procedures/status-archives/) to add closure days due to inclement weather and budgetary issues.

## Data Processing
All the datasets that I used were simple and already mostly clean.  I had to transform the date variable in each dataset to a standardardized format so that they could be merged into one large dataset. 

## Data Visualization
### Characteristics of Metro Ridership
#### By Month
![alt text](Graphs/Average Ridership by Day of Week (Boxplot).png)

