from pvlib import clearsky, atmosphere
from pvlib.location import Location
import pandas as pd

### NREL Bird Model implementation: for obtaining clear sky GHI

class ClearSky:
    def __init__(self, test_location, test_year, run_train):
        self.test_location = test_location
        self.test_year = test_year
        self.run_train = run_train

    def cs_ghi(self):

        # All_locations
        timezones = {
            "Bondville":Location(40.1134,-88.3695, 'US/Central', 217.932, 'Bondville'),
            "Boulder":Location(40.0150,-105.2705, 'US/Mountain', 1655.064, 'Boulder'),
            "Desert_Rock":Location(36.621,-116.043, 'US/Pacific', 1010.1072, 'Desert Rock'),
            "Fort_Peck":Location(48,-106.449, 'US/Mountain', 630.0216, 'Fort Peck'),
            "Goodwin_Creek":Location(34.2487,-89.8925, 'US/Central', 98, 'Goodwin Creek'),
            "Penn_State":Location(40.798,-77.859, 'US/Eastern', 351.74, 'Penn State'),
            "Sioux_Falls":Location(43.544,-96.73, 'US/Central', 448.086, 'Sioux Falls')
        }
        bvl = timezones[self.test_location]

        timestamp_series={
         "2009" : pd.DatetimeIndex(start='2009-01-01', end='2010-01-01', freq='1min',tz=bvl.tz),   # 12 months
         "2015" : pd.DatetimeIndex(start='2015-01-01', end='2016-01-01', freq='1min',tz=bvl.tz),   # 12 months
         "2016" : pd.DatetimeIndex(start='2016-01-01', end='2017-01-01', freq='1min',tz=bvl.tz),   # 12 months
         "2017" : pd.DatetimeIndex(start='2017-01-01', end='2018-01-01', freq='1min',tz=bvl.tz)   # 12 months
        }

        times = timestamp_series[self.test_year]

        if self.run_train:
           # TRAIN set
           times2010and2011 = pd.DatetimeIndex(start='2010-01-01', end='2012-01-01', freq='1min',
                                   tz=bvl.tz)   # 24 months of 2010 and 2011 - For training
           cs_2010and2011 = bvl.get_clearsky(times2010and2011) # ineichen with climatology table by default
           cs_2010and2011.drop(['dni','dhi'],axis=1, inplace=True) #updating the same dataframe by dropping two columns
           cs_2010and2011.reset_index(inplace=True)

           #cs_2010and2011['index']=cs_2010and2011['index'].apply(lambda x:x.to_datetime())
           cs_2010and2011['index']=pd.to_datetime(cs_2010and2011['index'])
           cs_2010and2011['year'] = cs_2010and2011['index'].apply(lambda x:x.year)
           cs_2010and2011['month'] = cs_2010and2011['index'].apply(lambda x:x.month)
           cs_2010and2011['day'] = cs_2010and2011['index'].apply(lambda x:x.day)
           cs_2010and2011['hour'] = cs_2010and2011['index'].apply(lambda x:x.hour)
           cs_2010and2011['min'] = cs_2010and2011['index'].apply(lambda x:x.minute)


           cs_2010and2011.drop(cs_2010and2011.index[-1], inplace=True)
           print("cs_2010and2011.shape=",cs_2010and2011.shape)
           cs_2010and2011.head()


        # TEST set
        cs_test = bvl.get_clearsky(times)
        cs_test.drop(['dni','dhi'],axis=1, inplace=True) #updating the same dataframe by dropping two columns
        cs_test.reset_index(inplace=True)

        #cs_test['index']= cs_test['index'].apply(lambda x:x.to_datetime())
        cs_test['index'] = pd.to_datetime(cs_test['index'])
        cs_test['year'] = cs_test['index'].apply(lambda x:x.year)
        cs_test['month'] = cs_test['index'].apply(lambda x:x.month)
        cs_test['day'] = cs_test['index'].apply(lambda x:x.day)
        cs_test['hour'] = cs_test['index'].apply(lambda x:x.hour)
        cs_test['min'] = cs_test['index'].apply(lambda x:x.minute)

        cs_test.drop(cs_test.index[-1], inplace=True)
        print("cs_test.shape=", cs_test.shape)

        return cs_test, cs_2010and2011

