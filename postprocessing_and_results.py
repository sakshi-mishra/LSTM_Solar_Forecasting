import numpy as np
import pandas as pd

class PostProcess:
    def __init__(self, run_train, test_loss, df_new_test, results_dir, train_loss):
        self.run_train = run_train
        self.test_loss = test_loss
        self.df_new_test = df_new_test
        self.RESULTS_DIR = results_dir
        self.train_loss = train_loss
        
    def denormalization(self):

        #Denormalization
        # TODO: Should we be using the mean instead of -1?
        mse_1 = np.array(self.test_loss).squeeze()[-1][0]
        mse_2 = np.array(self.test_loss).squeeze()[-1][1]
        mse_3 = np.array(self.test_loss).squeeze()[-1][2]
        mse_4 = np.array(self.test_loss).squeeze()[-1][3]

        rmse_1 = np.sqrt(mse_1)
        rmse_2 = np.sqrt(mse_2)
        rmse_3 = np.sqrt(mse_3)
        rmse_4 = np.sqrt(mse_4)

        print("rmse_1:",rmse_1)
        print("rmse_2:",rmse_2)
        print("rmse_3:",rmse_3)
        print("rmse_4:",rmse_4)

        rmse_denorm1 = (rmse_1 * (self.df_new_test['Kt'].max() - self.df_new_test['Kt'].min()))+ self.df_new_test['Kt'].mean()
        rmse_denorm2 = (rmse_2 * (self.df_new_test['Kt_2'].max() - self.df_new_test['Kt_2'].min()))+ self.df_new_test['Kt_2'].mean()
        rmse_denorm3 = (rmse_3 * (self.df_new_test['Kt_3'].max() - self.df_new_test['Kt_3'].min()))+ self.df_new_test['Kt_3'].mean()
        rmse_denorm4 = (rmse_4 * (self.df_new_test['Kt_4'].max() - self.df_new_test['Kt_4'].min()))+ self.df_new_test['Kt_4'].mean()

        print("rmse_denorm1:",rmse_denorm1)
        print("rmse_denorm2:",rmse_denorm2)
        print("rmse_denorm3:",rmse_denorm3)
        print("rmse_denorm4:",rmse_denorm4)

        rmse_denorm_all = [rmse_denorm1, rmse_denorm2, rmse_denorm3, rmse_denorm4]

        rmse_mean = np.mean([rmse_denorm1, rmse_denorm2, rmse_denorm3, rmse_denorm4])
        print("rmse_mean:", rmse_mean)

        print(self.df_new_test['Kt'].describe())
        print('\n')
        print(self.df_new_test['Kt_2'].describe())
        print('\n')
        print(self.df_new_test['Kt_3'].describe())
        print('\n')
        print(self.df_new_test['Kt_4'].describe())

        return rmse_denorm_all, rmse_mean

    def write_to_file(self):
        self.rmse_denorm_all, self.rmse_mean = self.denormalization()

        # Write to file
        f = open(self.RESULTS_DIR + '/' + 'results.txt', 'a+')
        j=0
        for i in self.rmse_denorm_all:
            j += 1
            f.write("rmse_denorm{}: {}\r\n".format(j, i))
        f.write('mean_rmse: {}'.format(self.rmse_mean))

        f.close()

        # ### Saving train and test losses to a csv

        if self.run_train:
            df_trainLoss = pd.DataFrame(data={'Train Loss':self.train_loss}, columns=['Train Loss'])
            df_trainLoss.head()

        testloss_unsqueezed = np.array(self.test_loss).squeeze()


        df_testLoss = pd.DataFrame(data=testloss_unsqueezed,columns=['mse1','mse2', 'mse3', 'mse4'])
        df_testLoss.head()

        df_testLoss.to_csv(self.RESULTS_DIR + '/' + '_TestLoss.csv')
        if self.run_train:
            df_trainLoss.to_csv(self.RESULTS_DIR + '/' + '_TrainLoss.csv')