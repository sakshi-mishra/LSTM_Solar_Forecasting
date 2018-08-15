# coding: utf-8

import sys
import numpy as np
import pandas as pd
import pathlib  # To mimick mkdir -p
import argparse

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_color_codes()

import bird_sky_model
import data_input
import LSTMModel
import exploratory_data_analysis
import data_cleaning


parser = argparse.ArgumentParser()
parser.add_argument('l',
                    help="Location whose data needs to be trained/tested with"+
                    "Values can be one of [Bondville, Boulder, Desert_Rock,"
                                        + "Fort_Peck,Goodwin_Creek, Penn_State,"
                                         + "Sioux_Falls]")
parser.add_argument('y',
                    help='4 digit Test year. One among [2009,2015,2016,2017]')
parser.add_argument('t',
                    help='True or False.To train using 2010-2011 data or not')
parser.add_argument('--num-epochs', default = 1000, type=int,
                    help="Number of training, testing epochs")
args, _ = parser.parse_known_args()
# Sanity check the arguments
# args.y
test_year = args.y
if test_year not in ["2009", "2015", "2016", "2017"]:
    print("Test year argument is not valid. Exiting...")
    parser.print_help()
    exit()
# args.t
if args.t  in ["True", "true"]:
    run_train = True
elif args.t in ["False", "false"]:
    run_train = False
else:
    print("Train flag is invalid. It should be True or false. Exiting.")
    parser.print_help()
# args.l
test_location = args.l
if test_location not in ["Bondville", "Boulder", "Desert_Rock",
                         "Fort_Peck,Goodwin_Creek, Penn_State", "Sioux_Falls"]:
    print("Test location is not valide.Exiting...")
    parser.print_help()
# args.num_epochs
num_epochs = args.num_epochs
print("test_location=",test_location, "test_year=",test_year,"run_train=",
      run_train, "num_epochs=", num_epochs)

# Define the Directories to save the trained model and results. Create the dir
# if it does not exist using pathlib
MODEL_DIR = 'LSTM_Results/Exp2_1/' + test_location
RESULTS_DIR = 'LSTM_Results/Exp2_1/' + test_location + '/' +  test_year
pathlib.Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
log_file = RESULTS_DIR + '/' + 'console.log'
print("Writing print statements to ", log_file)
sys.stdout = open(log_file, 'w')  # Redirect print statement's outputs to file
print("Stdout:")


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
cs_test, cs_2010and2011 = bird_sky_model.cs_ghi(test_location, test_year, run_train)

# data_input.py's output here
df_train, df_test = data_input.load_n_merge(test_location, test_year, run_train, cs_test, cs_2010and2011)

# eploratory data analysis
plots = exploratory_data_analysis.EDA(df_test)

# cleaning the data - removing the outliers
df_train, df_test = data_cleaning.CleanData(df_train, df_test, run_train)

# pre-processing the data by making the Kt (clear sky index at time t) column
# by first removing rows with ghi==0



### start of LSTM

import torch
import torch.nn as nn
from torch.autograd import Variable

if run_train:
    # Instantiating Model Class
    input_dim = 22
    hidden_dim = 15
    layer_dim = 1
    output_dim = 4
    batch_size = 100

    model = LSTMModel.LSTM_Model(input_dim, hidden_dim, layer_dim, output_dim)

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
    print("Loaded model from file\n");


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

    n_iter =0
    num_samples = len(X_train)
    test_samples = len(X_test)
    batch_size = 100
    #num_epochs = 1000  # Defined earlier using args
    feat_dim = X_train.shape[1]


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
                for i in range(0,int(test_samples/batch_size -1)):
                    features = Variable(X_test[i*batch_size:(i+1)*batch_size, :]).view(-1, seq_dim, feat_dim)
                    Kt_test = Variable(y_test[i*batch_size:(i+1)*batch_size])

                    outputs = model(features)

                    test_batch_mse.append(np.mean([(Kt_test.data.numpy() - outputs.data.numpy().squeeze())**2],axis=1))

                test_iter.append(n_iter)
                test_loss.append(np.mean([test_batch_mse],axis=1))

                print('Epoch: {} Iteration: {}. Train_MSE: {}. Test_MSE: {}'.format(epoch, n_iter, loss.data[0], test_loss[-1]))

    torch.save(model,MODEL_DIR+ '/torch_model_2010_2011')

figLossTrain = plt.figure()
plt.plot(np.array(test_loss).squeeze(),'r')

figLossTrain.savefig(RESULTS_DIR +'/'+ 'train_loss.jpg', bbox_inches = 'tight')

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



for i in range(0,int(test_samples/batch_size -1)):
    features = Variable(X_test[i*batch_size:(i+1)*batch_size, :]).view(-1, seq_dim, feat_dim)
    Kt_test = Variable(y_test[i*batch_size:(i+1)*batch_size])

    outputs = model(features)

    test_batch_mse.append(np.mean([(Kt_test.data.numpy() - outputs.data.numpy().squeeze())**2],axis=1))

    test_iter.append(i)
    test_loss.append(np.mean([test_batch_mse],axis=1))


if run_train:
    print("len(train_loss):",len(train_loss))
    plt.plot(train_loss,'-')


print("len(test_loss):",len(test_loss))
figLoss = plt.figure()
plt.plot(np.array(test_loss).squeeze(),'r')

figLoss.savefig(RESULTS_DIR + '/' + 'test_loss.jpg', bbox_inches = 'tight')


#### Demornamization

# TODO: Should we be using the mean instead of -1?
mse_1 = np.array(test_loss).squeeze()[-1][0]
mse_2 = np.array(test_loss).squeeze()[-1][1]
mse_3 = np.array(test_loss).squeeze()[-1][2]
mse_4 = np.array(test_loss).squeeze()[-1][3]

rmse_1 = np.sqrt(mse_1)
rmse_2 = np.sqrt(mse_2)
rmse_3 = np.sqrt(mse_3)
rmse_4 = np.sqrt(mse_4)

print("rmse_1:",rmse_1)
print("rmse_2:",rmse_2)
print("rmse_3:",rmse_3)
print("rmse_4:",rmse_4)

rmse_denorm1 = (rmse_1 * (df_new_test['Kt'].max() - df_new_test['Kt'].min()))+ df_new_test['Kt'].mean()
rmse_denorm2 = (rmse_2 * (df_new_test['Kt_2'].max() - df_new_test['Kt_2'].min()))+ df_new_test['Kt_2'].mean()
rmse_denorm3 = (rmse_3 * (df_new_test['Kt_3'].max() - df_new_test['Kt_3'].min()))+ df_new_test['Kt_3'].mean()
rmse_denorm4 = (rmse_4 * (df_new_test['Kt_4'].max() - df_new_test['Kt_4'].min()))+ df_new_test['Kt_4'].mean()

print("rmse_denorm1:",rmse_denorm1)
print("rmse_denorm2:",rmse_denorm2)
print("rmse_denorm3:",rmse_denorm3)
print("rmse_denorm4:",rmse_denorm4)

rmse_denorm_all = [rmse_denorm1, rmse_denorm2, rmse_denorm3, rmse_denorm4]

rmse_mean = np.mean([rmse_denorm1, rmse_denorm2, rmse_denorm3, rmse_denorm4])
print("rmse_mean:",rmse_mean)

print(df_new_test['Kt'].describe())
print('\n')
print(df_new_test['Kt_2'].describe())
print('\n')
print(df_new_test['Kt_3'].describe())
print('\n')
print(df_new_test['Kt_4'].describe())


# Write to file
f = open(RESULTS_DIR + '/' + 'results.txt', 'a+')
j=0
for i in rmse_denorm_all:
    j += 1
    f.write("rmse_denorm{}: {}\r\n".format(j,i))
f.write('mean_rmse: {}'.format(rmse_mean))

f.close()

# ### Saving train and test losses to a csv

if run_train:
    df_trainLoss = pd.DataFrame(data={'Train Loss':train_loss}, columns=['Train Loss'])
    df_trainLoss.head()

testloss_unsqueezed = np.array(test_loss).squeeze()


df_testLoss = pd.DataFrame(data=testloss_unsqueezed,columns=['mse1','mse2', 'mse3', 'mse4'])
df_testLoss.head()

df_testLoss.to_csv(RESULTS_DIR + '/' + '_TestLoss.csv')
if run_train:
    df_trainLoss.to_csv(RESULTS_DIR + '/' + '_TrainLoss.csv')