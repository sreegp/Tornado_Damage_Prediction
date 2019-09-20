#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import datetime
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import preprocessing

class data_collection_and_merging:   
    def get_tornado_data(storm_events_file):
        storms = pd.DataFrame()
        years = ['2008_c20180718', '2009_c20180718', '2010_c20170726', 
                 '2011_c20180718', '2012_c20170519', '2013_c20170519', 
                 '2014_c20180718', '2015_c20180525', '2016_c20180718',
                 '2017_c20181017', '2018_c20181017']
        # Pull all storm events files (2008-2018)
        # Source URL: https://www1.ncdc.noaa.gov/pub/data/swdi/stormevents/csvfiles/
        for year in years:
            storms = storms.append(pd.read_csv(str(storm_events_file)
                                           +str(year)+'.csv'))
        # Filter out non-tornado events
        tornadoes = storms.loc[storms['EVENT_TYPE'] == 'Tornado']
        return tornadoes
    def get_income_data(income_data_file):
        # Get income data, latest year available: 2016
        # Source URL: https://www.census.gov/data/datasets/2016/demo/saipe/2016-state-and-county.html
        income = pd.read_csv(income_data_file, header=3)
        # Clean income data to match 'State FIPS Code' and 'Name' formatting 
        # with tornado 'STATE_FIPS' and 'CZ_NAME' data
        income['State FIPS Code'] = income['State FIPS Code'].astype(int)
        income = income[~income['Name'].str.contains("County") == False]
        income['Name'].replace(regex=True,inplace=True,to_replace=r' County',value=r'')
        income['Name'] = income['Name'].str.upper()
        # Isolate 'State FIPS Code' and 'Name' (for data merging), and
        # pull 'Median Household Income', which is our only column of interest
        income = income[['State FIPS Code', 'Name', 'Median Household Income']]
        return income
    def get_population_data(population_data_file):
        # Pull population density data (2010)
        # Source URL: https://factfinder.census.gov/faces/tableservices/jsf/pages/productview.xhtml?src=bkmk 
        density = pd.read_csv(population_data_file, encoding='cp1252', header=1)
        # Clean density data to match 'Geographic area' and Geographic area.1' formatting 
        # with tornado+income 'STATE' and 'CZ_NAME' data
        density = density[~density['Geographic area.1'].str.contains("County") == False]
        density['Geographic area.1'].replace(regex=True,inplace=True,to_replace=r' County',value=r'')
        density['Geographic area.1'] = density['Geographic area.1'].str.upper()
        density['Geographic area'].replace(regex=True,inplace=True,to_replace=r'United States - ',value=r'')
        density['Geographic area'] = [x.split(' -')[0] for x in density['Geographic area']]
        density['Geographic area'] = density['Geographic area'].str.upper()
        # Isolate 'Geographic area' and 'Geographic area.1' (for data merging), and
        # pull 'Density per square mile of land area - Population' and
        # 'Density per square mile of landa area - Housing units', 
        # which is the column of interest
        pop_density = density[['Geographic area', 'Geographic area.1', 'Population', 'Housing units', 
                           'Area in square miles - Total area', 'Area in square miles - Land area',
                           'Density per square mile of land area - Population', 
                           'Density per square mile of land area - Housing units']]
        return pop_density
    
    def merge_data(tornado, income, population):
        # Merge income and tornado dataframes on FIPS and county columns 
        tornadoes_with_income = pd.merge(tornado, income, how='left', 
                                         left_on=['STATE_FIPS','CZ_NAME'], 
                                         right_on=['State FIPS Code','Name'])
        # Merge density and tornado+income dataframes on state and county columns (per above)
        tornadoes_with_income_with_density = pd.merge(tornadoes_with_income, population, how='left',
                                                      left_on=['STATE','CZ_NAME'], 
                                                      right_on=['Geographic area','Geographic area.1'])
        merged_dataframe = tornadoes_with_income_with_density
        return merged_dataframe
    
class data_preprocessing:
    def __init__(self, data):
        self.data = data 
        
    def day_of_year_week_weekend(self):
        
        def date_to_nth_day(date):
            # convert date to day of year from 1 to 365
            date = datetime.datetime.strptime(date,'%d-%b-%y %H:%M:%S')
            new_year_day = datetime.datetime(year=date.year, month=1, day=1)
            return (date - new_year_day).days + 1
        
        self.data['day_of_year'] = self.data['BEGIN_DATE_TIME'].apply(date_to_nth_day)
        self.data['day_of_week'] = pd.to_datetime(self.data['BEGIN_DATE_TIME']).dt.weekday
        self.data['weekend'] = self.data['day_of_week'].map({0:0, 1:0, 2:0, 3:0, 4:0, 5:1, 6:1})
        
    def drop_and_rename_columns(self):
        # Remove irrelevant columns and set EVENT_ID as index
        self.data = self.data.drop(columns=['EPISODE_ID', 'EVENT_TYPE', 'WFO', 'SOURCE', 'MAGNITUDE', 
                                  'MAGNITUDE_TYPE', 'FLOOD_CAUSE', 'CATEGORY', 'DAMAGE_PROPERTY', 'DAMAGE_CROPS', 
                                  'TOR_OTHER_WFO', 'TOR_OTHER_CZ_STATE', 'TOR_OTHER_CZ_FIPS', 'TOR_OTHER_CZ_NAME', 
                                  'BEGIN_AZIMUTH', 'END_AZIMUTH', 'BEGIN_LOCATION', 'END_LOCATION', 
                                  'EPISODE_NARRATIVE', 'DATA_SOURCE', 'BEGIN_YEARMONTH', 'END_YEARMONTH', 
                                  'MONTH_NAME', 'CZ_TIMEZONE', 'YEAR', 'STATE_FIPS', 'CZ_TYPE', 'CZ_FIPS', 
                                  'State FIPS Code', 'Name', 'Geographic area', 'Geographic area.1', 
                                  'Population', 'Housing units', 'STATE', 'CZ_NAME', 'END_TIME'])
        self.data = self.data.set_index('EVENT_ID')
        # Rename some columns for clarity
        self.data = self.data.rename(columns={'Density per square mile of land area - Population': 'population_density',
                                    'Density per square mile of land area - Housing units': 'housing_units_density', 
                                    'Area in square miles - Total area': 'total_area', 
                                    'Area in square miles - Land area': 'land_area',
                                    'BEGIN_TIME': 'begin_time', 'Median Household Income': 'median_income'})
    
    def create_duration(self):
        # Convert begin and end dates to datetime and get tornado duration
        self.data['BEGIN_DATE_TIME'] =  pd.to_datetime(self.data['BEGIN_DATE_TIME'])
        self.data['END_DATE_TIME'] =  pd.to_datetime(self.data['END_DATE_TIME'])
        self.data['duration'] = self.data['END_DATE_TIME'] - self.data['BEGIN_DATE_TIME']
        self.data['duration'] = self.data['duration'].dt.seconds/60
        self.data = self.data.drop(columns=['END_DATE_TIME'])

    def create_casualties_column(self):
        # Convert all injury- and death-related columns to numeric, sum
        # and create casualties from sum
        self.data['INJURIES_DIRECT'] = pd.to_numeric(self.data['INJURIES_DIRECT'])
        self.data['INJURIES_INDIRECT'] = pd.to_numeric(self.data['INJURIES_INDIRECT'])
        self.data['DEATHS_DIRECT'] = pd.to_numeric(self.data['DEATHS_DIRECT'])
        self.data['DEATHS_INDIRECT'] = pd.to_numeric(self.data['DEATHS_INDIRECT'])
        self.data["casualties"] = (self.data["INJURIES_INDIRECT"]+self.data["INJURIES_DIRECT"]+
                              self.data["DEATHS_DIRECT"]+self.data["DEATHS_INDIRECT"])
        self.data = self.data.drop(columns=["INJURIES_INDIRECT", "INJURIES_DIRECT", 
                                  "DEATHS_INDIRECT", "DEATHS_DIRECT"])

    def calc_tornado_area(self):
        # Calculate tornado area from length and width
        self.data['TOR_LENGTH'] = self.data['TOR_LENGTH'].map(lambda x: int(x))
        self.data['tornado_area'] = self.data['TOR_LENGTH']*self.data['TOR_WIDTH']
        self.data = self.data.drop(columns=['TOR_LENGTH', 'TOR_WIDTH'])

    def calc_min_and_avg_range(self):
        # Roughly proxying distance from population center
        self.data['average_range'] = self.data.loc[:, ['BEGIN_RANGE','END_RANGE']].mean(axis = 1)
        self.data['minimum_range'] = self.data.loc[:, ['BEGIN_RANGE','END_RANGE']].min(axis = 1)
        self.data = self.data.drop(columns=['BEGIN_RANGE', 'END_RANGE'])

    def calc_avg_lat_and_long(self):
        # Calculate average lat/long
        self.data['average_latitude'] = (self.data['BEGIN_LAT'] + self.data['END_LAT'])/2
        self.data["average_longitude"] = (self.data['BEGIN_LON'] + self.data['END_LON'])/2
        self.data = self.data.drop(columns=['BEGIN_LAT', 'END_LAT', 'BEGIN_LON', 'END_LON'])

    def calc_percentage_land(self):
        # Calculate percent land out of total area.
        # Total area can include water.
        self.data['percent_land'] = self.data['land_area']/self.data['total_area']
        self.data = self.data.drop(columns=['land_area', 'total_area'])
        
    def extract_multi_vortex_ref(self):
        # Extract multi-vortex references from EVENT_NARRATIVE
        self.data['multi_vortex'] = 0
        self.data.loc[self.data.apply(lambda x: 'multi-vortex' in x['EVENT_NARRATIVE'], axis=1), ['multi_vortex']] = 1
        self.data.loc[self.data.apply(lambda x: 'multiple vortex' in x['EVENT_NARRATIVE'], axis=1), ['multi_vortex']] = 1
        self.data.loc[self.data.apply(lambda x: 'multiple vortices' in x['EVENT_NARRATIVE'], axis=1), ['multi_vortex']] = 1
        self.data = self.data.drop(columns=['EVENT_NARRATIVE'])
        
    def fillna(self):
        ## replace NaNs with average
        self.data['median_income'] = self.data['median_income'].str.replace(",", "").astype(float)
        mean_Median_Household_Income = self.data['median_income'].mean(skipna = True)
        self.data['median_income'] = self.data['median_income'].fillna(mean_Median_Household_Income)

        self.data['population_density'] = self.data['population_density'].astype(float)
        mean_Pop_Density = self.data['population_density'].mean(skipna = True)
        self.data['population_density'] = self.data['population_density'].fillna(mean_Pop_Density)

        self.data['housing_units_density'] = self.data['housing_units_density'].astype(float)
        mean_Housing_Units_Density = self.data['housing_units_density'].mean(skipna = True)
        self.data['housing_units_density'] = self.data['housing_units_density'].fillna(mean_Housing_Units_Density)

        self.data['percent_land'] = self.data['percent_land'].astype(float)
        mean_Percent_Land = self.data['percent_land'].mean(skipna = True)
        self.data['percent_land'] = self.data['percent_land'].fillna(mean_Percent_Land)
        
    def sin_and_cosine_time(self):
        
        # Convert time features cyclically
        def convert_to_mins(time_in_24_hours):
            stringtime = str(time_in_24_hours)
            if (len(stringtime) == 4):
                hours_as_mins = int(stringtime[0:2])*60
                mins = int(stringtime[2:4])
                total_mins = hours_as_mins + mins
            elif (len(stringtime) == 3):
                hours_as_mins = int(stringtime[0])*60
                mins = int(stringtime[1:3])
                total_mins = hours_as_mins + mins
            elif (len(stringtime) < 3):
                total_mins = int(stringtime)
            else:
                print('Bad Data')
                assert False
            return total_mins
        
        self.data['begin_time'] = self.data['begin_time'].apply(convert_to_mins)
        minutes_in_a_day = 24*60
        self.data['sin_time'] = np.sin(2*np.pi*self.data['begin_time']/minutes_in_a_day)
        self.data['cos_time'] = np.cos(2*np.pi*self.data['begin_time']/minutes_in_a_day)
        days_in_a_year = 365
        self.data['sin_date'] = np.sin(2*np.pi*self.data['day_of_year']/days_in_a_year)
        self.data['cos_date'] = np.cos(2*np.pi*self.data['day_of_year']/days_in_a_year)
        self.data = self.data.drop(columns=['begin_time', 'day_of_year', 'BEGIN_DAY', 
                                  'END_DAY', 'day_of_week'])
        
    def binary_tornado_intensity_estimate(self):
        # drop rows without tornado intensity estimate
        self.data = self.data[self.data['TOR_F_SCALE'] != 'EFU']
        self.data['TOR_F_SCALE'] = self.data['TOR_F_SCALE'].map(lambda x: int(x.lstrip('EF')))
        
        def makeEFBinary(ef):
            # EF_Scale is an estimate of tornado intensity, 
            # We convert it to binary (1 for 2+, 0 for 0,1)
            if ef <= 1:
                return 0
            elif ef <= 5:
                return 1
            else:
                assert False
            
        self.data["tornado_intensity"] = self.data["TOR_F_SCALE"].apply(makeEFBinary)
        self.data = self.data.drop(columns='TOR_F_SCALE')
        
    def binarize_casualties(self): 
        self.data['binary_casualties'] = np.where(self.data['casualties']>=1, 1, 0)
        self.data = self.data.drop(columns=['casualties'])