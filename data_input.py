import pandas as pd
import glob

### Import files from each year in a separate dataframe


class DataInput:
    def __init__(self, test_location, test_year, run_train, cs_test, cs_2010and2011):
        self.test_location = test_location
        self.test_year = test_year
        self.run_train = run_train
        self.cs_test = cs_test
        self.cs_2010and2011 = cs_2010and2011

    def load_n_merge(self):

        cols = ['year', 'jday', 'month', 'day','hour','min','dt','zen','dw_solar','dw_solar_QC','uw_solar',
               'uw_solar_QC', 'direct_n','direct_n_QC','diffuse', 'diffuse_QC', 'dw_ir', 'dw_ir_QC', 'dw_casetemp',
               'dw_casetemp_QC', 'dw_dometemp','dw_dometemp_QC','uw_ir', 'uw_ir_QC', 'uw_casetemp','uw_casetemp_QC',
               'uw_dometemp','uw_dometemp_QC','uvb','uvb_QC','par','par_QC','netsolar','netsolar_QC','netir','netir_QC',
               'totalnet','totalnet_QC','temp','temp_QC','rh','rh_QC','windspd','windspd_QC','winddir','winddir_QC',
               'pressure','pressure_QC']

        if self.run_train:
            # Train Set
            path = r'data/' + self.test_location + '/Exp_1_train'
            print("train_path:",path)
            all_files = glob.glob(path + "/*.dat")
            all_files.sort()

            df_big_train = pd.concat([pd.read_csv(f, skipinitialspace = True, quotechar = '"',skiprows=(2),delimiter=' ',
                            index_col=False,header=None, names=cols) for f in all_files],ignore_index=True)
            print(df_big_train.shape)

            ### Merging Clear Sky GHI And the big dataframe

            df_train = pd.merge(df_big_train, self.cs_2010and2011, on=['year','month','day','hour','min'])
            print("loaded training set\n");
            print("df_train.shape=", df_train.shape)

            # Test set
            path = r'data/' + self.test_location + '/Exp_1_test/' + self.test_year
            print(path)
            all_files = glob.glob(path + "/*.dat")
            all_files.sort()

            df_big_test = pd.concat((pd.read_csv(f, skipinitialspace = True, quotechar = '"', skiprows=(2),delimiter=' ',
                             index_col=False, header=None, names=cols) for f in all_files), ignore_index=True)

            ### Merging Clear Sky GHI And the big dataframe

            df_test = pd.merge(df_big_test, self.cs_test, on=['year','month','day','hour','min'])
            print('df_test.shape:', df_test.shape)
            print("loaded test set\n");
            print('df_big_test.shape:', df_big_test.shape)

            return df_train, df_test

        else:
            # Test set
            path = r'./data/' + self.test_location + '/Exp_1_test/' + self.test_year
            print(path)
            all_files = glob.glob(path + "/*.dat")
            all_files.sort()

            df_big_test = pd.concat((pd.read_csv(f, skipinitialspace=True, quotechar='"', skiprows=(2), delimiter=' ',
                                                 index_col=False, header=None, names=cols) for f in all_files),
                                    ignore_index=True)

            ### Merging Clear Sky GHI And the big dataframe

            df_test = pd.merge(df_big_test, self.cs_test, on=['year', 'month', 'day', 'hour', 'min'])
            print('df_test.shape:', df_test.shape)
            print("loaded test set\n");
            print('df_big_test.shape:', df_big_test.shape)

            return df_test
