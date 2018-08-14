import pandas as pd
import glob
from pandas.compat import StringIO

### Import files from each year in a separate dataframe


cols = ['year', 'jday', 'month', 'day','hour','min','dt','zen','dw_solar','dw_solar_QC','uw_solar',
       'uw_solar_QC', 'direct_n','direct_n_QC','diffuse', 'diffuse_QC', 'dw_ir', 'dw_ir_QC', 'dw_casetemp',
       'dw_casetemp_QC', 'dw_dometemp','dw_dometemp_QC','uw_ir', 'uw_ir_QC', 'uw_casetemp','uw_casetemp_QC',
       'uw_dometemp','uw_dometemp_QC','uvb','uvb_QC','par','par_QC','netsolar','netsolar_QC','netir','netir_QC',
       'totalnet','totalnet_QC','temp','temp_QC','rh','rh_QC','windspd','windspd_QC','winddir','winddir_QC',
       'pressure','pressure_QC']


if run_train:
   # Train Set
   path = r'./data/' + test_location + '/Exp_1_train'
   print("train_path:",path)
   all_files = glob.glob(path + "/*.dat")
   all_files.sort()

   df_big_train = pd.concat([pd.read_csv(f, skipinitialspace = True, quotechar = '"',skiprows=(2),delimiter=' ',
                    index_col=False,header=None, names=cols) for f in all_files],ignore_index=True)
   print(df_big_train.shape)
   df_train = pd.merge(df_big_train, cs_2010and2011, on=['year','month','day','hour','min'])
   print("loaded training set\n");
   print("df_train.shape=",df_train.shape)



# Test set
path = r'./data/' + test_location + '/Exp_1_test/' + test_year
print(path)
all_files = glob.glob(path + "/*.dat")
all_files.sort()

df_big_test = pd.concat((pd.read_csv(f, skipinitialspace = True, quotechar = '"',skiprows=(2),delimiter=' ',
                 index_col=False,header=None, names=cols) for f in all_files),ignore_index=True)
df_test = pd.merge(df_big_test, cs_test, on=['year','month','day','hour','min'])
print('df_test.shape:',df_test.shape)
print("loaded test set\n");
print('df_big_test.shape:',df_big_test.shape)
