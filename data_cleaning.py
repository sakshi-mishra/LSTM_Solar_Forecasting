import numpy as np
import pandas as pd

class CleanData:
    def __init__(self, df_train, df_test, run_train):
        self.df_train = df_train
        self.df_test = df_test
        self.run_train = run_train
        
    def clean(self):
        
        if self.run_train:
            # TRAIN set
            #updating the same dataframe by dropping the index columns from clear sky model
            self.df_train.drop(['index'], axis=1, inplace=True)
            # Resetting Index
            self.df_train.reset_index(drop=True, inplace=True)
        
        
        # TEST set
        #updating the same dataframe by dropping the index columns from clear sky model
        self.df_test.drop(['index'],axis=1, inplace=True)
        # Resetting Index
        self.df_test.reset_index(drop=True, inplace=True)
        
        
        ### Managing missing values
        
        if self.run_train:
            # TRAIN set
            #Dropping rows with two or more -9999.9 values in columns
            missing_data_indices = np.where((self.df_train <=-9999.9).apply(sum, axis=1)>=2)[0] #Get indices of all rows with 2 or more -9999.9
            self.df_train.drop(missing_data_indices, axis=0, inplace=True) # Drop those inddices
            print('self.df_train.shape:',self.df_train.shape)
            self.df_train.reset_index(drop=True, inplace=True) # 2nd time - Resetting index
        
        
        # TEST set
        missing_data_indices_test = np.where((self.df_test <= -9999.9).apply(sum, axis=1)>=2)[0]
        self.df_test.drop(missing_data_indices_test, axis=0, inplace=True)
        print('self.df_test.shape:',self.df_test.shape)
        self.df_test.reset_index(drop=True, inplace=True) # 2nd time - Reseting Index
        
        
        #### First resetting index after dropping rows in the previous part of the code
        
        if self.run_train:
            # TRAIN set
            one_miss_train_idx = np.where((self.df_train <=-9999.9).apply(sum, axis=1)==1)[0]
            print('(len(one_miss_train_idx)',len(one_miss_train_idx))
            self.df_train.shape
        
            col_names = self.df_train.columns
            from collections import defaultdict
            stats = defaultdict(int)
            total_single_missing_values = 0
            for name in col_names:
                col_mean = self.df_train[~(self.df_train[name] == -9999.9)][name].mean()
                missing_indices = np.where((self.df_train[name] == -9999.9))
                stats[name] = len(missing_indices[0])
                self.df_train[name].loc[missing_indices] = col_mean
                total_single_missing_values += sum(self.df_train[name] == -9999.9)
        
            train = np.where((self.df_train <=-9999.9).apply(sum, axis=1)==1)[0]
            print('len(train):',len(train))
        
        # TEST set
        one_miss_test_idx = np.where((self.df_test <=-9999.9).apply(sum, axis=1)==1)[0]
        len(one_miss_test_idx)
        col_names_test = self.df_test.columns
        
        from collections import defaultdict
        stats_test = defaultdict(int)
        total_single_missing_values_test = 0
        for name in col_names_test:
            col_mean = self.df_test[~(self.df_test[name] == -9999.9)][name].mean()
            missing_indices = np.where((self.df_test[name] == -9999.9))
            stats_test[name] = len(missing_indices[0])
            self.df_test[name].loc[missing_indices] = col_mean
            total_single_missing_values_test += sum(self.df_test[name] == -9999.9)
        
        test = np.where((self.df_test <=-9999.9).apply(sum, axis=1)==1)[0]
        print('len(test):', len(test))
        print('self.df_test.shape:', self.df_test.shape)

        return self.df_train, self.df_test