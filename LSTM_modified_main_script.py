# coding: utf-8

import sys
import numpy as np
import pathlib  # To mimick mkdir -p
import argparse

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
### NREL Bird Model implementation: for obtaining clear sky GHI

import itertools
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_color_codes()

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


# data_input.py's output here


### Merging Clear Sky GHI And the big dataframe

if run_train:
    # TRAIN set
    #updating the same dataframe by dropping the index columns from clear sky model
    df_train.drop(['index'],axis=1, inplace=True)
    # Resetting Index
    df_train.reset_index(drop=True, inplace=True)


# TEST set
#updating the same dataframe by dropping the index columns from clear sky model
df_test.drop(['index'],axis=1, inplace=True)
# Resetting Index
df_test.reset_index(drop=True, inplace=True)


### Managing missing values

if run_train:
    # TRAIN set
    #Dropping rows with two or more -9999.9 values in columns
    missing_data_indices = np.where((df_train <=-9999.9).apply(sum, axis=1)>=2)[0] #Get indices of all rows with 2 or more -9999.9
    df_train.drop(missing_data_indices, axis=0, inplace=True) # Drop those inddices
    print('df_train.shape:',df_train.shape)
    df_train.reset_index(drop=True, inplace=True) # 2nd time - Resetting index


# TEST set
missing_data_indices_test = np.where((df_test <= -9999.9).apply(sum, axis=1)>=2)[0]
df_test.drop(missing_data_indices_test, axis=0, inplace=True)
print('df_test.shape:',df_test.shape)
df_test.reset_index(drop=True, inplace=True) # 2nd time - Reseting Index


#### First resetting index after dropping rows in the previous part of the code

if run_train:
    # TRAIN set
    one_miss_train_idx = np.where((df_train <=-9999.9).apply(sum, axis=1)==1)[0]
    print('(len(one_miss_train_idx)',len(one_miss_train_idx))
    df_train.shape

    col_names = df_train.columns
    from collections import defaultdict
    stats = defaultdict(int)
    total_single_missing_values = 0
    for name in col_names:
        col_mean = df_train[~(df_train[name] == -9999.9)][name].mean()
        missing_indices = np.where((df_train[name] == -9999.9))
        stats[name] = len(missing_indices[0])
        df_train[name].loc[missing_indices] = col_mean
        total_single_missing_values += sum(df_train[name] == -9999.9)

    train = np.where((df_train <=-9999.9).apply(sum, axis=1)==1)[0]
    print('len(train):',len(train))

# TEST set
one_miss_test_idx = np.where((df_test <=-9999.9).apply(sum, axis=1)==1)[0]
len(one_miss_test_idx)
col_names_test = df_test.columns

from collections import defaultdict
stats_test = defaultdict(int)
total_single_missing_values_test = 0
for name in col_names_test:
    col_mean = df_test[~(df_test[name] == -9999.9)][name].mean()
    missing_indices = np.where((df_test[name] == -9999.9))
    stats_test[name] = len(missing_indices[0])
    df_test[name].loc[missing_indices] = col_mean
    total_single_missing_values_test += sum(df_test[name] == -9999.9)

test = np.where((df_test <=-9999.9).apply(sum, axis=1)==1)[0]
print('len(test):',len(test))
print('df_test.shape:',df_test.shape)


### making the Kt (clear sky index at time t) column by first removing rows with ghi==0


if run_train:
    # TRAIN dataset
    df_train = df_train[df_train['ghi']!=0]
    df_train['Kt'] = df_train['dw_solar']/df_train['ghi']
    df_train.reset_index(inplace=True)

    print("train Kt max: "+str(df_train['Kt'].max()))
    print("train Kt min: "+str(df_train['Kt'].min()))
    print("train Kt mean: "+str(df_train['Kt'].mean()))


# TEST dataset
df_test = df_test[df_test['ghi']!=0]
df_test['Kt'] = df_test['dw_solar']/df_test['ghi']
df_test.reset_index(inplace=True)

print("test Kt max: "+str(df_test['Kt'].max()))
print("test Kt min: "+str(df_test['Kt'].min()))
print("test Kt mean: "+str(df_test['Kt'].mean()))


if run_train:
    # TRAIN dataset
    df_train= df_train[df_train['Kt']< 5000]
    df_train= df_train[df_train['Kt']> -1000]


# Test dataset
df_test= df_test[df_test['Kt']< 5000]
df_test= df_test[df_test['Kt']> -1000]


### Making 4 Kt columns

if run_train:
    # Train dataset
    df_train['Kt_2'] = df_train['Kt']
    df_train['Kt_3'] = df_train['Kt']
    df_train['Kt_4'] = df_train['Kt']


# Test dataset
df_test['Kt_2'] = df_test['Kt']
df_test['Kt_3'] = df_test['Kt']
df_test['Kt_4'] = df_test['Kt']


#### Group the data (train dataframe)

if run_train:

    zen = df_train.groupby(['year','month','day','hour'])['zen'].mean()
    dw_solar = df_train.groupby(['year','month','day','hour'])['dw_solar'].mean()
    uw_solar = df_train.groupby(['year','month','day','hour'])['uw_solar'].mean()
    direct_n = df_train.groupby(['year','month','day','hour'])['direct_n'].mean()
    diffuse = df_train.groupby(['year','month','day','hour'])['diffuse'].mean()
    dw_ir = df_train.groupby(['year','month','day','hour'])['dw_ir'].mean()
    dw_casetemp = df_train.groupby(['year','month','day','hour'])['dw_casetemp'].mean()
    dw_dometemp = df_train.groupby(['year','month','day','hour'])['dw_dometemp'].mean()
    uw_ir = df_train.groupby(['year','month','day','hour'])['uw_ir'].mean()
    uw_casetemp = df_train.groupby(['year','month','day','hour'])['uw_casetemp'].mean()
    uw_dometemp = df_train.groupby(['year','month','day','hour'])['uw_dometemp'].mean()
    uvb = df_train.groupby(['year','month','day','hour'])['uvb'].mean()
    par = df_train.groupby(['year','month','day','hour'])['par'].mean()
    netsolar = df_train.groupby(['year','month','day','hour'])['netsolar'].mean()
    netir = df_train.groupby(['year','month','day','hour'])['netir'].mean()
    totalnet = df_train.groupby(['year','month','day','hour'])['totalnet'].mean()
    temp = df_train.groupby(['year','month','day','hour'])['temp'].mean()
    rh = df_train.groupby(['year','month','day','hour'])['rh'].mean()
    windspd = df_train.groupby(['year','month','day','hour'])['windspd'].mean()
    winddir = df_train.groupby(['year','month','day','hour'])['winddir'].mean()
    pressure = df_train.groupby(['year','month','day','hour'])['pressure'].mean()
    ghi = df_train.groupby(['year','month','day','hour'])['ghi'].mean()
    Kt = df_train.groupby(['year','month','day','hour'])['Kt'].mean()
    Kt_2 = df_train.groupby(['year','month','day','hour'])['Kt_2'].mean()
    Kt_3 = df_train.groupby(['year','month','day','hour'])['Kt_3'].mean()
    Kt_4 = df_train.groupby(['year','month','day','hour'])['Kt_4'].mean()



    df_new_train = pd.concat([zen,dw_solar,uw_solar,direct_n,diffuse,dw_ir,dw_casetemp,dw_dometemp,uw_ir,uw_casetemp,uw_dometemp,
                        uvb,par,netsolar,netir,totalnet,temp,rh,windspd,winddir,pressure,ghi,Kt,Kt_2,Kt_3,Kt_4], axis=1)


#### Groupdata - test dataframe

test_zen = df_test.groupby(['month','day','hour'])['zen'].mean()
test_dw_solar = df_test.groupby(['month','day','hour'])['dw_solar'].mean()
test_uw_solar = df_test.groupby(['month','day','hour'])['uw_solar'].mean()
test_direct_n = df_test.groupby(['month','day','hour'])['direct_n'].mean()
test_diffuse = df_test.groupby(['month','day','hour'])['diffuse'].mean()
test_dw_ir = df_test.groupby(['month','day','hour'])['dw_ir'].mean()
test_dw_casetemp = df_test.groupby(['month','day','hour'])['dw_casetemp'].mean()
test_dw_dometemp = df_test.groupby(['month','day','hour'])['dw_dometemp'].mean()
test_uw_ir = df_test.groupby(['month','day','hour'])['uw_ir'].mean()
test_uw_casetemp = df_test.groupby(['month','day','hour'])['uw_casetemp'].mean()
test_uw_dometemp = df_test.groupby(['month','day','hour'])['uw_dometemp'].mean()
test_uvb = df_test.groupby(['month','day','hour'])['uvb'].mean()
test_par = df_test.groupby(['month','day','hour'])['par'].mean()
test_netsolar = df_test.groupby(['month','day','hour'])['netsolar'].mean()
test_netir = df_test.groupby(['month','day','hour'])['netir'].mean()
test_totalnet = df_test.groupby(['month','day','hour'])['totalnet'].mean()
test_temp = df_test.groupby(['month','day','hour'])['temp'].mean()
test_rh = df_test.groupby(['month','day','hour'])['rh'].mean()
test_windspd = df_test.groupby(['month','day','hour'])['windspd'].mean()
test_winddir = df_test.groupby(['month','day','hour'])['winddir'].mean()
test_pressure = df_test.groupby(['month','day','hour'])['pressure'].mean()
test_ghi = df_test.groupby(['month','day','hour'])['ghi'].mean()
test_Kt = df_test.groupby(['month','day','hour'])['Kt'].mean()
test_Kt_2 = df_test.groupby(['month','day','hour'])['Kt_2'].mean()
test_Kt_3 = df_test.groupby(['month','day','hour'])['Kt_3'].mean()
test_Kt_4 = df_test.groupby(['month','day','hour'])['Kt_4'].mean()




df_new_test = pd.concat([test_zen,test_dw_solar,test_uw_solar,test_direct_n,test_diffuse,test_dw_ir,
                         test_dw_casetemp,test_dw_dometemp,test_uw_ir,test_uw_casetemp,test_uw_dometemp,
                    test_uvb,test_par,test_netsolar,test_netir,test_totalnet,test_temp,test_rh,
                         test_windspd,test_winddir,test_pressure,test_ghi,test_Kt,test_Kt_2,test_Kt_3,test_Kt_4], axis=1)


#df_new_test.loc[2].xs(17,level='day')


### Shifting Kt values to make 1 hour, 2 hour, 3 hour and 4 hour ahead forecast

#### Train dataset


if run_train:
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


for i in levels_index2[0]:
    for j in levels_index2[1]:
        df_new_test.loc[i].loc[j]['Kt'] = df_new_test.loc[i].loc[j]['Kt'].shift(-1)
        df_new_test.loc[i].loc[j]['Kt_2'] = df_new_test.loc[i].loc[j]['Kt_2'].shift(-2)
        df_new_test.loc[i].loc[j]['Kt_3'] = df_new_test.loc[i].loc[j]['Kt_3'].shift(-3)
        df_new_test.loc[i].loc[j]['Kt_4'] = df_new_test.loc[i].loc[j]['Kt_4'].shift(-4)


df_new_test = df_new_test[~(df_new_test['Kt_4'].isnull())]


### Normalize train and test dataframe

if run_train:
    # TRAIN set
    train_norm = (df_new_train - df_new_train.mean()) / (df_new_train.max() - df_new_train.min())
    train_norm.reset_index(inplace=True,drop=True)

# TEST set
test_norm =  (df_new_test - df_new_test.mean()) / (df_new_test.max() - df_new_test.min())
test_norm.reset_index(inplace=True,drop=True)


### Making train and test sets with train_norm and test_norm

if run_train:
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

    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)

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