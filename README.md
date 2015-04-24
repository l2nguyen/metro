## Study of Metro Ridership
This is the class project for my General Assembly Data Science class.

**Goals:** 
* To visualize the DC Metro rail historical ridership data
* To determine the variables that affect ridership
* To build a model that detremines the relationship between the outcome, metrorail ridership, and the feature variables (ie gas price, weather, unemployment)

**Game Plan:**
* This is a regression problem and I plan to use a linear regression model 
* The main model evaluation tool will be RMSE
* Will make models of increasing complexity and see what works best

**Guide:**
* For a flavor of the project concept, see [Initial class presentation](Initial Project Presentation.pdf)
* Data wrangling code can be found [here](Code/wrangling.py)
* Graphs visualizing data can be found [here](Graphs)
* Modelling code can be found [here](Code/modeling.py)
* Documentation can be found [here](Documentation.pdf)

**To Do List**
* Try random forests with the data
* Find a way to deal with outliers
* Feature selection: figure out which features have a significant relationship with ridership numbers
* Parameter tuning

**Wish List**
* Make interactive data visualizations using javascript/d3. Maybe something like [this](http://mbtaviz.github.io/)

**Data Sources:**
* Metro Ridership Data from [Open Data DC](http://www.opendatadc.org/dataset/wmata-metrorail-ridership-by-date)
* Gas price data from [US Energy Information Administration](http://www.eia.gov/dnav/pet/pet_pri_gnd_dcus_r1z_m.htm)
* Unemployment data from [US Bureau of Labor Stats](http://www.bls.gov/eag/eag.dc_washington_md.htm)
(Note: Data is for Washington DC Metro area which includes DC and other areas such as NoVA, Montgomery County, PG County, and parts of WV)
* Weather data from [NOAA](http://www.ncdc.noaa.gov/cdo-web/datatools)
(Note: I used data from the weather station at Reagan National Airport)
* Capital Bikeshare (CaBi) data used as a proxy for tourism. Additionally, the increased use of bikeshare will mean fewer people will be taking the metro. CaBi trip history data found [here](https://www.capitalbikeshare.com/trip-history-data)
* Federal closing days: Data for federal holidays can be found [here](https://catalog.data.gov/dataset/federal-holidays). This data was then supplemented with data of governemtn closing due to budget issues (Oct 2013) and snow days on from the [OPM archives](http://www.opm.gov/policy-data-oversight/snow-dismissal-procedures/status-archives/)
* Number of rail cars was deduced from this [site](http://www.wmata.com/rail/frequency.cfm) that has the frequency of train arrivals. The number of cars was calculated by dividing the total number of minutes in each period and dividing it by the number of minutes in between trains
