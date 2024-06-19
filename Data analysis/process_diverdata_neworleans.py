
import pandas as pd
import numpy as np
import chardet
import matplotlib.pyplot as plt
from scipy import stats
import math
from pathlib import Path
from os.path import basename, dirname, isdir, isfile, join

#%%
WATER_DENSITY = 1000 #kg/m3
GRAVITATIONAL_ACCELERATION = 9.80665 #m/s2

class GWMWell:
    
    def __init__(self, root: str, gwm_id: str, ztop: float, 
                 depth_peilbuis: float, start_date: str, end_date: str):
    
        if depth_peilbuis < 0:
            raise ValueError('Depth of the peilbuis should be a positive value.')
        
        self.root = Path(root)
        self.gwm_id = gwm_id
        self.date_range = pd.date_range(start=pd.to_datetime(start_date, format='%d-%m-%Y'), 
                                        end=pd.to_datetime(end_date, format='%d-%m-%Y'), freq='H')
        self.ztop = ztop
        self.depth_peilbuis = depth_peilbuis
    
    def add_barometer_from_diver(self, fname: str):
        """
        Reads barometer data from a csv file and adds it to the object.
        Specified to take in raw csv files from divers.
    
        Parameters:
        fname -- file name of the csv file containing the barometer data

        """
        # Read data from csv file
        if not Path(fname).is_absolute():
            fname = self.root.joinpath(fname)
            
        barometer_data = pd.read_csv(fname, usecols=[0,1], names=['date','pressure (cmH20)'],
                                     decimal=".", skiprows=52, delimiter=',', encoding="ISO-8859-1", engine='python')

        barometer_data = barometer_data[:-1].replace('     ', np.NaN)
        barometer_data = barometer_data.dropna(subset=['pressure (cmH20)'])
    
        # Convert 'pressure (cmH20)' and 'date' columns to numeric and datetime, respectively
        barometer_data['pressure (cmH20)'] = pd.to_numeric(barometer_data['pressure (cmH20)'])
        barometer_data['date'] = pd.to_datetime(barometer_data['date'], format='%Y/%m/%d %H:%M:%S')
    
        # Set index to 'date' and reindex with object's date_range
        self.barometer_data = barometer_data.set_index('date').reindex(self.date_range)
    
    def add_barometer_from_NOAA(self, fname: str): #TODO: update function to take NOAA data from API
        """
        Reads barometer data from a non-diver csv file and adds it to the object.
        Specified to take in raw csv files from NOAA.
    
        Parameters:
        fname -- file name of the csv file containing the barometer data

        """
        # Read data from csv file
        barometer_data = pd.read_csv(fname, usecols=[1,6], names=['date','pressure (0.1HP)'],
                                     decimal=".", skiprows=1, delimiter=',', encoding="ISO-8859-1", engine='python')

        barometer_data = barometer_data[:-1].replace('99999,9', np.NaN)
        #TODO: check if NOAA data is already numeric, otherwise pd.to_numeric here
        barometer_data['pressure (cmH20)'] = pd.to_numeric(barometer_data['pressure (cmH20)'])
        barometer_data['pressure (cmH2O)'] = (barometer_data['pressure (0.1HP)'] / 10) * 1.0197162129779 #convert from hPa to cmH2O
        barometer_data = barometer_data.dropna(subset=['pressure (cmH20)'])
    
        # Convert 'pressure (cmH20)' and 'date' columns to numeric and datetime, respectively
        barometer_data['pressure (cmH20)'] = pd.to_numeric(barometer_data['pressure (cmH20)'])
        barometer_data['date'] = pd.to_datetime(barometer_data['date'], format='%Y/%m/%dT%H:%M:%S')
    
        # Set index to 'date' and reindex with object's date_range
        self.barometer_data = barometer_data.set_index('date').reindex(self.date_range)
    
    def load_diver_data(self): #TODO: incoroporate cutoff_pres as variable instead of fixed 1320
        """
        Loads Diver data from a csv file located in the specified directory and preprocesses it to match the 
        date range of the peilbuis object. Only records with a pressure greater than {cutoff_pres} (default=1320) cmH20 are considered valid.
    
        """
        path = self.root.joinpath(self.gwm_id).with_suffix('.csv')
        diver_data = pd.read_csv(path, usecols=[0, 1, 2], 
                                 names=["date", "pressure (cmH20)", "temperature (degC)"], 
                                 decimal=".", skiprows=52, delimiter=",", encoding="ISO-8859-1", engine="python")
        diver_data = diver_data[:-1].replace("     ", np.NaN)
        diver_data = diver_data.dropna(subset=["pressure (cmH20)"])
    
        diver_data["pressure (cmH20)"] = pd.to_numeric(diver_data["pressure (cmH20)"])
        diver_data["temperature (degC)"] = pd.to_numeric(diver_data["temperature (degC)"])
        diver_data["date"] = pd.to_datetime(diver_data["date"], format="%Y/%m/%d %H:%M:%S")
        diver_data = diver_data.set_index("date").reindex(self.date_range, method='nearest', limit=1)
        
        # remove outliers
        #z_scores = stats.zscore(diver_data[["pressure (cmH20)"]].dropna())
        #valid = (abs(z_scores.reindex(diver_data.index)) < 3) & (diver_data[["pressure (cmH20)"]] > cutff_pres) #TODO reindex doesn't work on numpy array?
        #self.diver_data = diver_data.where(valid["pressure (cmH20)"])
        self.diver_data = diver_data
    
    def add_handreading(self, datetime: str, handreading: float):
        """
        Add a new handreading to the peilbuis object.
        
        Parameters:
        -----------
        datetime: str
            The date and time of the handreading in the format '%d-%m-%Y %H:%M'.
        handreading: float
            The value of the handreading in meters above ztop.
        """
        
        datetime = pd.to_datetime(datetime, format='%d-%m-%Y %H:%M', errors='raise')
        data = {'date': [datetime],
                'handreading (m-ztop)': [handreading]}
        
        self.handreadings = pd.concat([getattr(self, 'handreadings', pd.DataFrame()) , 
                                       pd.DataFrame(data, index=[0]).set_index('date')]).\
                                        drop_duplicates(keep='last').sort_index()
        
        # Calculate the handreadings in meters datum and store them in a new dataframe
        indices = np.searchsorted(self.date_range, self.handreadings.index)
        handreadings_at_datum = self.ztop.iloc[indices]['ztop (m datum)'].values - self.handreadings['handreading (m-ztop)'].values
        self.handreadings_at_datum = pd.DataFrame({'handreadings (m datum)': handreadings_at_datum}, index=self.handreadings.index)
        
    def barometric_compensation(self, method='last', drop_data_before=None, drop_data_after=None):
        """
        Perform barometric compensation on the pressure data.
        
        Args:
            match_handreadings (bool, optional): Whether to match handreadings or not.
                                                 Defaults to False.
        
        Raises:
            Warning: If manual measurements are available.
            ValueError: If `match_handreadings` is `True` but no handreadings are available.
        """
        
        if not hasattr(self, 'handreadings'):
            raise ValueError('no handreadings available')
            
        water_column = 9806.65 * (self.diver_data['pressure (cmH20)'] - self.barometer_data['pressure (cmH20)']) / (
                    WATER_DENSITY * GRAVITATIONAL_ACCELERATION) #cm
    
        indices = []
        for target_date in self.handreadings.index:
            closet_index = np.abs(self.date_range - target_date).argmin()
            indices.append(closet_index) 
        
        if method == 'last':
            handreading = self.handreadings.iloc[-1].item() * 100 #cm         
            index = indices[-1]
            cable_length = handreading + water_column[index]
            
        else:
            raise ValueError("only method 'last' implemented.")

        water_level = self.ztop - cable_length + water_column
        self.water_level = water_level.to_frame('water_level (m datum)')
        
        if drop_data_after is not None:
            datetime_format = "%d-%m-%Y"
            end_date = pd.to_datetime(drop_data_after, format=datetime_format)
            self.water_level =self.water_level.loc[:end_date]
            
        if drop_data_before is not None:
            datetime_format = "%d-%m-%Y"
            start_date = pd.to_datetime(drop_data_before, format=datetime_format)
            self.water_level =self.water_level.loc[start_date:]

    
#%%

#root = r"N:/Projects/11200500/11200801/F. Other information/Task_H_Real-Time/Diver_data"
metadata = pd.read_excel(r'n:\Projects\11200500\11200801\F. Other information\Task_H_Real-Time\Metadata\NO_gw_well_locations_testonly.xlsx', index_col = 0)
start_date = '20-05-2022'
end_date = '30-01-2024'
    
for peilbuis_name, row in metadata.iterrows():
    print(peilbuis_name)
    
    GWMWell = GWMWell(
        root = r"N:/Projects/11200500/11200801/F. Other information/Task_H_Real-Time/Diver_data/raw_data",
        gwm_id = "Diver_3_VEI_DY706_240521094335_DY706",
        ztop = row['Elevation_m_NAVD88'],
        depth_peilbuis = row['well_depth_cm'],
        start_date = start_date,
        end_date = end_date)
    
    GWMWell.load_diver_data()
    GWMWell.add_barometer_from_NOAA(fname=r'n:\Projects\11200500\11200801\F. Other information\Task_H_Real-Time\Diver_data\baro\3694926.csv')
