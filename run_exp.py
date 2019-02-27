#!/usr/bin/env python
# NOTE: This script may need to be run twice IF the trained models are not
# already present for all the sites. The first run will train (also test for the first test_year
# which is 2009) and then save the trained model. Once the saveed model is
# available, we need to launch this script once more to generate test results
# for all the test_year
import subprocess
from subprocess import Popen
import pathlib

sites = ["Bondville", "Boulder", "Desert_Rock",
                         "Fort_Peck","Goodwin_Creek", "Penn_State", "Sioux_Falls"]
test_year = ["2009", "2015", "2016", "2017"]
SCRIPT_NAME = "main.py"

script_path = "/home/reopt/Documents/solar_forecasting/LSTM_Solar_Forecasting/"


#for site in sites:
for i in range(1):
    site = sites[i]

    MODEL_DIR = 'LSTM_Results/Exp2_1/' + site
    # NOTE: The Name of the model file has to match with the training script
    MODEL_FILE = MODEL_DIR + '/torch_model_2010_2011'
    if pathlib.Path(MODEL_FILE).is_file():
        # A trained model exists. So just run tests
        # for year in test_year[0]:
        for i in range(1):
            year = test_year[i]

            RESULTS_DIR = 'LSTM_Results/Exp2_1/' + site + '/' + year
            pathlib.Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
            log_file = RESULTS_DIR + '/' + 'stdout.log'

            # proc = ["xterm", "-e","python", SCRIPT_NAME , site
            #                   , year, "false", "2>&1", "|", "tee", log_file]
            proc = ["python", SCRIPT_NAME, site, year, "true", ">", log_file]

            print(' '.join(proc))
            Popen(proc,stdin=subprocess.PIPE)
    else :
        # No trained model exists. We need to train as well
        # Train and test with test_year[0]

        RESULTS_DIR = 'LSTM_Results/Exp2_1/' + site + '/' + test_year[0]
        pathlib.Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
        log_file = script_path+ '/'+ RESULTS_DIR + '/' + 'stdout.log'
        # proc = ["xterm", "-e", "python", SCRIPT_NAME , site
        #                       , test_year[0], "true", "2>&1", "|", "tee", log_file]

        proc = ["python", SCRIPT_NAME, site, test_year[0], "true", ">", log_file]

        print(' '.join(proc))   
        Popen(proc, stdin=subprocess.PIPE)


        for i in range(1):
        # for year in test_year[1:2]:
            year = test_year[i]
            # Launch test scripts for this site and this year RESULTS_DIR = 'LSTM_Results/Exp2_1/' + site + '/' + year
            pathlib.Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
            log_file = RESULTS_DIR + '/' + 'stdout.log'

            # proc = ["xterm", "-e","python", SCRIPT_NAME , site
            #                   , year, "false", "2>&1", "|", "tee", log_file]
            proc = ["python",
                    SCRIPT_NAME, site, year, "false", ">", log_file]

            print(' '.join(proc))
            Popen(proc, stdin=subprocess.PIPE)
