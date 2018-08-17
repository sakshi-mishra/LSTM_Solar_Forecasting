import numpy as np
import pandas as pd
#from numpy import round

class PreProcess:
    def __init__(self, df_train, df_test, run_train):
        self.df_train = df_train
        self.df_test = df_test
        self.run_train = run_train
        
    def data_prepro(self):
        
        if self.run_train:
            # TRAIN dataset
            self.df_train = self.df_train[self.df_train['ghi']!=0]
            self.df_train['Kt'] = self.df_train['dw_solar']/self.df_train['ghi']
            self.df_train.reset_index(inplace=True)

            print("train Kt max: "+str(self.df_train['Kt'].max()))
            print("train Kt min: "+str(self.df_train['Kt'].min()))
            print("train Kt mean: "+str(self.df_train['Kt'].mean()))


        # TEST dataset
        self.df_test = self.df_test[self.df_test['ghi']!=0]
        self.df_test['Kt'] = self.df_test['dw_solar']/self.df_test['ghi']
        self.df_test.reset_index(inplace=True)

        print("test Kt max: "+str(self.df_test['Kt'].max()))
        print("test Kt min: "+str(self.df_test['Kt'].min()))
        print("test Kt mean: "+str(self.df_test['Kt'].mean()))


        if self.run_train:
            # TRAIN dataset
            self.df_train= self.df_train[self.df_train['Kt']< 5000]
            self.df_train= self.df_train[self.df_train['Kt']> -1000]


        # Test dataset
        self.df_test= self.df_test[self.df_test['Kt']< 5000]
        self.df_test= self.df_test[self.df_test['Kt']> -1000]


        ### Making 4 Kt columns

        if self.run_train:
            # Train dataset
            self.df_train['Kt_2'] = self.df_train['Kt']
            self.df_train['Kt_3'] = self.df_train['Kt']
            self.df_train['Kt_4'] = self.df_train['Kt']


        # Test dataset
        self.df_test['Kt_2'] = self.df_test['Kt']
        self.df_test['Kt_3'] = self.df_test['Kt']
        self.df_test['Kt_4'] = self.df_test['Kt']


        #### Group the data (train dataframe)

        if self.run_train:

            zen = self.df_train.groupby(['year','month','day','hour'])['zen'].mean()
            dw_solar = self.df_train.groupby(['year','month','day','hour'])['dw_solar'].mean()
            uw_solar = self.df_train.groupby(['year','month','day','hour'])['uw_solar'].mean()
            direct_n = self.df_train.groupby(['year','month','day','hour'])['direct_n'].mean()
            diffuse = self.df_train.groupby(['year','month','day','hour'])['diffuse'].mean()
            dw_ir = self.df_train.groupby(['year','month','day','hour'])['dw_ir'].mean()
            dw_casetemp = self.df_train.groupby(['year','month','day','hour'])['dw_casetemp'].mean()
            dw_dometemp = self.df_train.groupby(['year','month','day','hour'])['dw_dometemp'].mean()
            uw_ir = self.df_train.groupby(['year','month','day','hour'])['uw_ir'].mean()
            uw_casetemp = self.df_train.groupby(['year','month','day','hour'])['uw_casetemp'].mean()
            uw_dometemp = self.df_train.groupby(['year','month','day','hour'])['uw_dometemp'].mean()
            uvb = self.df_train.groupby(['year','month','day','hour'])['uvb'].mean()
            par = self.df_train.groupby(['year','month','day','hour'])['par'].mean()
            netsolar = self.df_train.groupby(['year','month','day','hour'])['netsolar'].mean()
            netir = self.df_train.groupby(['year','month','day','hour'])['netir'].mean()
            totalnet = self.df_train.groupby(['year','month','day','hour'])['totalnet'].mean()
            temp = self.df_train.groupby(['year','month','day','hour'])['temp'].mean()
            rh = self.df_train.groupby(['year','month','day','hour'])['rh'].mean()
            windspd = self.df_train.groupby(['year','month','day','hour'])['windspd'].mean()
            winddir = self.df_train.groupby(['year','month','day','hour'])['winddir'].mean()
            pressure = self.df_train.groupby(['year','month','day','hour'])['pressure'].mean()
            ghi = self.df_train.groupby(['year','month','day','hour'])['ghi'].mean()
            Kt = self.df_train.groupby(['year','month','day','hour'])['Kt'].mean()
            Kt_2 = self.df_train.groupby(['year','month','day','hour'])['Kt_2'].mean()
            Kt_3 = self.df_train.groupby(['year','month','day','hour'])['Kt_3'].mean()
            Kt_4 = self.df_train.groupby(['year','month','day','hour'])['Kt_4'].mean()



            df_new_train = pd.concat([zen,dw_solar,uw_solar,direct_n,diffuse,dw_ir,dw_casetemp,dw_dometemp,uw_ir,uw_casetemp,uw_dometemp,
                                uvb,par,netsolar,netir,totalnet,temp,rh,windspd,winddir,pressure,ghi,Kt,Kt_2,Kt_3,Kt_4], axis=1)


        #### Groupdata - test dataframe

        test_zen = self.df_test.groupby(['month','day','hour'])['zen'].mean()
        test_dw_solar = self.df_test.groupby(['month','day','hour'])['dw_solar'].mean()
        test_uw_solar = self.df_test.groupby(['month','day','hour'])['uw_solar'].mean()
        test_direct_n = self.df_test.groupby(['month','day','hour'])['direct_n'].mean()
        test_diffuse = self.df_test.groupby(['month','day','hour'])['diffuse'].mean()
        test_dw_ir = self.df_test.groupby(['month','day','hour'])['dw_ir'].mean()
        test_dw_casetemp = self.df_test.groupby(['month','day','hour'])['dw_casetemp'].mean()
        test_dw_dometemp = self.df_test.groupby(['month','day','hour'])['dw_dometemp'].mean()
        test_uw_ir = self.df_test.groupby(['month','day','hour'])['uw_ir'].mean()
        test_uw_casetemp = self.df_test.groupby(['month','day','hour'])['uw_casetemp'].mean()
        test_uw_dometemp = self.df_test.groupby(['month','day','hour'])['uw_dometemp'].mean()
        test_uvb = self.df_test.groupby(['month','day','hour'])['uvb'].mean()
        test_par = self.df_test.groupby(['month','day','hour'])['par'].mean()
        test_netsolar = self.df_test.groupby(['month','day','hour'])['netsolar'].mean()
        test_netir = self.df_test.groupby(['month','day','hour'])['netir'].mean()
        test_totalnet = self.df_test.groupby(['month','day','hour'])['totalnet'].mean()
        test_temp = self.df_test.groupby(['month','day','hour'])['temp'].mean()
        test_rh = self.df_test.groupby(['month','day','hour'])['rh'].mean()
        test_windspd = self.df_test.groupby(['month','day','hour'])['windspd'].mean()
        test_winddir = self.df_test.groupby(['month','day','hour'])['winddir'].mean()
        test_pressure = self.df_test.groupby(['month','day','hour'])['pressure'].mean()
        test_ghi = self.df_test.groupby(['month','day','hour'])['ghi'].mean()
        test_Kt = self.df_test.groupby(['month','day','hour'])['Kt'].mean()
        test_Kt_2 = self.df_test.groupby(['month','day','hour'])['Kt_2'].mean()
        test_Kt_3 = self.df_test.groupby(['month','day','hour'])['Kt_3'].mean()
        test_Kt_4 = self.df_test.groupby(['month','day','hour'])['Kt_4'].mean()




        df_new_test = pd.concat([test_zen,test_dw_solar,test_uw_solar,test_direct_n,test_diffuse,test_dw_ir,
                                 test_dw_casetemp,test_dw_dometemp,test_uw_ir,test_uw_casetemp,test_uw_dometemp,
                            test_uvb,test_par,test_netsolar,test_netir,test_totalnet,test_temp,test_rh,
                                 test_windspd,test_winddir,test_pressure,test_ghi,test_Kt,test_Kt_2,test_Kt_3,test_Kt_4], axis=1)


        ### Shifting Kt values to make 1 hour, 2 hour, 3 hour and 4 hour ahead forecast
        #### Train dataset

        if self.run_train:
            levels_index= []
            for m in df_new_train.index.levels:
                levels_index.append(m)
            for i in levels_index[0]:
                for j in levels_index[1]:
                    df_new_train.loc[i].loc[j]['Kt'] = df_new_train.loc[i].loc[j]['Kt'].shift(-1)
                    df_new_train.loc[i].loc[j]['Kt_2'] = df_new_train.loc[i].loc[j]['Kt_2'].shift(-2)
                    df_new_train.loc[i].loc[j]['Kt_3'] = df_new_train.loc[i].loc[j]['Kt_3'].shift(-3)
                    df_new_train.loc[i].loc[j]['Kt_4'] = df_new_train.loc[i].loc[j]['Kt_4'].shift(-4)
            df_new_train = df_new_train[~(df_new_train['Kt_4'].isnull())]

        #### Test dataset

        levels_index2= []
        for m in df_new_test.index.levels:
            levels_index2.append(m)

        # for i in levels_index2[0]:
        #     for j in levels_index2[1][:-3]:
        #         df_new_test.loc[i].loc[j]['Kt'] = df_new_test.loc[i].loc[j]['Kt'].shift(-1)
        #         df_new_test.loc[i].loc[j]['Kt_2'] = df_new_test.loc[i].loc[j]['Kt_2'].shift(-2)
        #         df_new_test.loc[i].loc[j]['Kt_3'] = df_new_test.loc[i].loc[j]['Kt_3'].shift(-3)
        #         df_new_test.loc[i].loc[j]['Kt_4'] = df_new_test.loc[i].loc[j]['Kt_4'].shift(-4)

        for i, j in zip(levels_index2[0], levels_index2[1]):
            try:
                df_new_test.loc[i].loc[j]['Kt'] = df_new_test.loc[i].loc[j]['Kt'].shift(-1)
                df_new_test.loc[i].loc[j]['Kt_2'] = df_new_test.loc[i].loc[j]['Kt_2'].shift(-2)
                df_new_test.loc[i].loc[j]['Kt_3'] = df_new_test.loc[i].loc[j]['Kt_3'].shift(-3)
                df_new_test.loc[i].loc[j]['Kt_4'] = df_new_test.loc[i].loc[j]['Kt_4'].shift(-4)
            except KeyError:
                continue

        df_new_test = df_new_test[~(df_new_test['Kt_4'].isnull())]

        ### Normalize train and test dataframe

        if self.run_train:
            # TRAIN set
            train_norm = (df_new_train - df_new_train.mean()) / (df_new_train.max() - df_new_train.min())
            train_norm.reset_index(inplace=True,drop=True)

        # TEST set
        test_norm =  (df_new_test - df_new_test.mean()) / (df_new_test.max() - df_new_test.min())
        test_norm.reset_index(inplace=True,drop=True)

        ### Making train and test sets with train_norm and test_norm

        import math
        def roundup(x):
            return int(math.ceil(x / 100.0)) * 100

        if self.run_train:
            # TRAIN set
            train_lim = roundup(train_norm.shape[0])
            train_random = train_norm.sample(train_lim-train_norm.shape[0])
            train_norm = train_norm.append(train_random)

            X1 = train_norm.drop(['Kt','Kt_2','Kt_3','Kt_4'],axis=1)
            y1 = train_norm[['Kt','Kt_2','Kt_3','Kt_4']]

            print("X1_train shape is {}".format(X1.shape))
            print("y1_train shape is {}".format(y1.shape))

            X_train = np.array(X1)
            y_train  = np.array(y1)


        # TEST set
        test_lim = roundup(test_norm.shape[0])
        test_random = test_norm.sample(test_lim-test_norm.shape[0])
        test_norm = test_norm.append(test_random)

        X2 = test_norm.drop(['Kt','Kt_2','Kt_3','Kt_4'],axis=1)
        y2 = test_norm[['Kt','Kt_2','Kt_3','Kt_4']]

        print("X2_test shape is {}".format(X2.shape))
        print("y2_test shape is {}".format(y2.shape))

        X_test = np.array(X2)
        y_test = np.array(y2)

        return X_train, y_train, X_test, y_test, df_new_test

