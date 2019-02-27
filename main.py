# coding: utf-8

import sys
import numpy as np
import pathlib  # To mimick mkdir -p
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparser
import bird_sky_model
import data_input
import LSTMModel
import exploratory_data_analysis
import data_cleaning
import data_preprocessing
import postprocessing_and_results

sns.set_color_codes()

test_location, test_year, run_train, num_epochs = argparser.get_arguments()

# Define the Directories to save the trained model and results. Create the dir
# if it does not exist using pathlib
MODEL_DIR = 'LSTM_Results/Exp2_1/' + test_location
RESULTS_DIR = 'LSTM_Results/Exp2_1/' + test_location + '/' +  test_year
pathlib.Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
log_file = RESULTS_DIR + '/' + 'console.log'
print("Writing print statements to ", log_file)
#sys.stdout = open(log_file, 'w')  # Redirect print statement's outputs to file
#print("Stdout:")


### CONFIGURE RUNS
#run_train = True # Disables training & processing of train set; Set it to True for the first time to create a model
#test_location = "Bondville" #Folder name
#test_location = "Boulder" #Folder name
#test_location = "Desert_Rock" #Folder name
#test_location = "Fort_Peck" #Folder name
#test_location = "Goodwin_Creek" #Folder name
#test_location = "Penn_State" #Folder name
#test_location = "Sioux_Falls" #Folder name



# bird_sky_model.py's output here
clear_sky_ghi = bird_sky_model.ClearSky(test_location, test_year, run_train)
cs_test, cs_2010and2011 = clear_sky_ghi.cs_ghi()
print("bird_sky_model.py module executed successfully")

# data_input.py's output here
input_data = data_input.DataInput(test_location, test_year, run_train, cs_test, cs_2010and2011)
df_train, df_test = input_data.load_n_merge()
print("data_input.py module executed successfully")

# exploratory data analysis
eda_plots = exploratory_data_analysis.EDA(df_test)
plots = eda_plots.ghi_plot()
print("exploratory_data_analysis.py module executed successfully")

# cleaning the data - removing the outliers
df = data_cleaning.CleanData(df_train, df_test, run_train)
df_train, df_test = df.clean()
print("data_cleaning.py module executed successfully")

### start of LSTM
def main():

    # pre-processing the data by making the Kt (ClearSkyIndex at time t) col by first removing rows with ghi==0

    Xy = data_preprocessing.PreProcess(df_train, df_test, run_train)
    X_train, y_train, X_test, y_test, df_new_test = Xy.data_prepro()
    print("data_preprocessing.py module executed successfully")

    if run_train:
        # Instantiating Model Class
        input_dim = 22
        hidden_dim = 15
        layer_dim = 1
        output_dim = 4
        batch_size = 100

        model = LSTMModel.LSTM_Model(input_dim, hidden_dim, layer_dim, output_dim)
        #model = mod.forward()
        print("LSTM model module executed to instantiate the LSTMmodel, with run_train=True")

        # Instantiating Loss Class
        criterion = nn.MSELoss()

        # Instantiate Optimizer Class
        learning_rate = 0.001
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        # initializing lists to store losses over epochs:
        train_loss = []
        train_iter = []
        print("Preparing model to train");
    else:
        model = torch.load('LSTM_Results/Exp2_1/' + test_location + '/torch_model_2010_2011')
        print("Loaded model from file, given run_train=False\n");


    # Test set

    test_loss = []
    test_iter = []
    # converting numpy array to torch tensor

    X_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test)

    # Convert to Float tensor

    X_test = X_test.type(torch.FloatTensor)
    y_test = y_test.type(torch.FloatTensor)



    if run_train:
        X_train = torch.from_numpy(X_train)
        y_train = torch.from_numpy(y_train)
        X_train = X_train.type(torch.FloatTensor)
        y_train = y_train.type(torch.FloatTensor)

        # Training the model
        seq_dim = 1

        n_iter = 0
        num_samples = len(X_train)
        test_samples = len(X_test)
        batch_size = 100
        #num_epochs = 1000  # Defined earlier using args
        feat_dim = X_train.shape[1]

        print("starting to train the model for {} epochs!".format(num_epochs))
        for epoch in range(num_epochs):
            for i in range(0, int(num_samples/batch_size -1)):


                features = Variable(X_train[i*batch_size:(i+1)*batch_size, :]).view(-1, seq_dim, feat_dim)
                Kt_value = Variable(y_train[i*batch_size:(i+1)*batch_size])

                #print("Kt_value={}".format(Kt_value))

                optimizer.zero_grad()

                outputs = model(features)
                #print("outputs ={}".format(outputs))

                loss = criterion(outputs, Kt_value)

                train_loss.append(loss.data[0])
                train_iter.append(n_iter)

                #print("loss = {}".format(loss))
                loss.backward()

                optimizer.step()

                n_iter += 1
                test_batch_mse =list()
                if n_iter%100 == 0:
                    for i in range(0, int(test_samples/batch_size -1)):
                        features = Variable(X_test[i*batch_size:(i+1)*batch_size, :]).view(-1, seq_dim, feat_dim)
                        Kt_test = Variable(y_test[i*batch_size:(i+1)*batch_size])

                        outputs = model(features)

                        test_batch_mse.append(np.mean([(Kt_test.data.numpy() - outputs.data.numpy().squeeze())**2],axis=1))

                    test_iter.append(n_iter)
                    test_loss.append(np.mean([test_batch_mse], axis=1))

                    print('Epoch: {} Iteration: {}. Train_MSE: {}. Test_MSE: {}'.format(epoch, n_iter, loss.data[0], test_loss[-1]))

        torch.save(model,MODEL_DIR+ '/torch_model_2010_2011')

    try:
        figLossTrain = plt.figure()
        plt.plot(np.array(test_loss).squeeze(),'r')

        figLossTrain.savefig(RESULTS_DIR +'/'+ 'train_loss.jpg', bbox_inches = 'tight')
    except RuntimeError as err:
        print("Skipping error:", err)
        pass

    # JUST TEST CELL

    batch_size = 100
    seq_dim = 1
    test_samples = len(X_test)
    batch_size = 100
    feat_dim = X_test.shape[1]

    # initializing lists to store losses over epochs:
    test_loss = []
    test_iter = []
    test_batch_mse = list()


    print("Starting to test the model")

    for i in range(0,int(test_samples/batch_size -1)):
        features = Variable(X_test[i*batch_size:(i+1)*batch_size, :]).view(-1, seq_dim, feat_dim)
        Kt_test = Variable(y_test[i*batch_size:(i+1)*batch_size])

        outputs = model(features)

        test_batch_mse.append(np.mean([(Kt_test.data.numpy() - outputs.data.numpy().squeeze())**2],axis=1))

        test_iter.append(i)
        test_loss.append(np.mean([test_batch_mse],axis=1))


    if run_train:
        print("len(train_loss):",len(train_loss))
        try:
            plt.plot(train_loss,'-')
        except RuntimeError as E:
            print("Skipping plot. Is X running or Agg backend set?")
            pass


    print("len(test_loss):",len(test_loss))
    try:

        figLoss = plt.figure()
        plt.plot(np.array(test_loss).squeeze(),'r')

        figLoss.savefig(RESULTS_DIR + '/' + 'test_loss.jpg', bbox_inches = 'tight')
    except RuntimeError as E:
        print("Not plotting. Probably no X")

    return test_loss, train_loss, df_new_test

if __name__=='__main__':
    test_loss, train_loss, df_new_test = main()
    print("About to start post processing the results")
    results = postprocessing_and_results.PostProcess(run_train, test_loss, df_new_test, RESULTS_DIR, train_loss)
    results.write_to_file()
