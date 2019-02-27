import matplotlib.pyplot as plt
import seaborn as sns

### Exploratory Data analysis
print("Starting exploratory data analysis\n");

class EDA:
    def __init__(self, df_test):
        self.df_test = df_test

    def ghi_plot(self):

        dw_solar_everyday = self.df_test.groupby(['jday'])['dw_solar'].mean()
        ghi_everyday = self.df_test.groupby(['jday'])['ghi'].mean()
        j_day = self.df_test['jday'].unique()

        fig = plt.figure()

        axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        axes1.scatter(j_day, dw_solar_everyday, label='Observed dw_solar', color='red')
        axes1.scatter(j_day, ghi_everyday, label='Clear Sky GHI', color='green')

        axes1.set_xlabel('Days')
        axes1.set_ylabel('Solar Irradiance (Watts /m^2)')
        axes1.set_title('Solar Irradiance - Test Year 2009')
        axes1.legend(loc='best')

        # fig.savefig('LSTM_Results/Exp2_1/' + test_location + '/'+  test_year + 'Figure 2.jpg', bbox_inches = 'tight')


        sns.jointplot(x=dw_solar_everyday,y=ghi_everyday,kind='reg')
        plt.xlabel('Observed global downwelling solar (Watts/m^2)')
        plt.ylabel('Clear Sky GHI (Watts/m^2)')
        # plt.savefig('LSTM_Results/Exp2_1/' + test_location + '/'+  test_year + 'Figure 3.jpg', bbox_inches='tight')
