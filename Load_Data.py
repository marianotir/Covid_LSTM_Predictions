# -*- coding: utf-8 -*-
"""
Load data automatically from web and prepare it
Original web
https://www.ecdc.europa.eu/en/publications-data/download-todays-data-geographic-distribution-covid-19-cases-worldwide


@author: mariano
"""

#---------------------------
# Import libraries
#---------------------------

import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import settings


#----------------------------
# Load data 
#----------------------------

url = "https://www.ecdc.europa.eu/sites/default/files/documents/COVID-19-geographic-disbtribution-worldwide.xlsx"

df = pd.read_excel(url,sheet_name='COVID-19-geographic-disbtributi')

# Get user name defined in settings
User_Name = settings.User_Name

#------------------------------
# Prepare data 
#-------------------------------

# Change names of some large colum names
df = df.rename(columns={"countriesAndTerritories": "countries", 
                        "countryterritoryCode":"Country_Code",
                        "continentExp":"continent",
                        "Cumulative_number_for_14_days_of_COVID-19_cases_per_100000":"Cum_Cases_14d",
                        })


# Shift cum cases Backwards and down for evaluations
df['CC_14d_Bw'] = df['Cum_Cases_14d'].shift(1,fill_value=0)

# Shift cum cases Backwards and down for evaluations
df['CC_14d_Fw'] = df['Cum_Cases_14d'].shift(-1,fill_value=0)

# Shift cases per day
df['cases_Fw'] = df['cases'].shift(-5,fill_value=0)

# Calculate Groth Factor base en cumulative cases per 1million
df['G_cum14'] = df['Cum_Cases_14d']/df['CC_14d_Fw']

# Calculate Groth Factor base en cases per day
df['G_cases'] = df['cases']/df['cases_Fw']

df.fillna(0,inplace=True)

# Transform date to the right format
df['date'] = pd.to_datetime(df['dateRep'], format='%d/%m/%Y')

# Dropput the unneded columns
df.drop('dateRep', axis=1, inplace=True)

#--------------------------------------
# Add rolling to all the countries
#--------------------------------------

# Define number of days considered for cumulative number
Cum_Sum_Days = settings.Cum_Sum_Days

# Get countries in dataset
countries = df.countries.unique()

# Sort by date all dataset to make the cumsum correctly
df.sort_values(by = 'date', inplace=True)

# Get empty dataframe with columns from the covid dataset
dt_temp = pd.DataFrame({}, columns=np.append(df.columns,'rolling'), index=None)

for i in countries:

  # subset for that country i
  temp = df[df['countries'] == i]

  # get the rolling sum of cases for last 14 days
  temp['rolling']=temp.cases.rolling(Cum_Sum_Days).sum()

  # append the subset dataset to the temporary dataset 
  dt_temp = dt_temp.append(temp)

# Drop na
dt_temp = dt_temp.dropna()

# Save a template
df_covid = dt_temp.copy()


#----------------------------------------
# Plot a country framework
#----------------------------------------

# Define the coutnry to be used
Country = settings.Country

# Subset for that country
df_c = df_covid[df_covid['countries'] == Country] 

x = df_c['date']
y = df_c['rolling']


fig, ax = plt.subplots()
ax.plot_date(x, y, markerfacecolor='CornflowerBlue', markeredgecolor='white')
fig.autofmt_xdate()
ax.set_xlim([datetime.date(2020, 5, 1), datetime.date(2020, 8, 10)])


#--------------------------------
# Keep one country for analysis
#--------------------------------

# Subset the country
data_country = df_covid[df_covid['countries'] == Country] 


#----------------------------------
# Save data for prediction
#----------------------------------

# Save all data for all countries
df_covid.to_csv("C:/Users/{}/Covid/Data/Data_Covid.csv".format(User_Name),index = False, header=True)

# Save data for one country that is going to be used for predictions in production
data_country.to_csv("C:/Users/{}/Covid/Data/Covid_Country.csv".format(User_Name),index = False, header=True)




