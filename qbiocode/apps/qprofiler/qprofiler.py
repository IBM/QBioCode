# ====== Base class imports ======
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timezone
import json
import pickle
import os
import re
import csv
import time
import sys
# ====== Hydra imports ======
import hydra
from hydra import compose, initialize_config_dir

# ====== Scikit-learn imports ======
from sklearn.model_selection import train_test_split

# ====== Qiskit imports ======
from qiskit_algorithms.utils import algorithm_globals

import sys
dir_home = re.sub( 'QBioCode.*', 'QBioCode', os.getcwd() )
sys.path.append( dir_home )

# ====== Scaling and encoding functions imports ======
from qbiocode import scaler_fn, feature_encoding
from qbiocode import get_embeddings
# ====== Evaluation functions imports ====
#from qmlbench.evaluation.dataset_evaluation_no_var_threshold import evaluate2 # use this for moons/circles data, otherwise you'll run into an error with finding no features with minimum variance threshold
from qbiocode import evaluate
from qbiocode import model_run

# Get the directory where this script is located for default config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(SCRIPT_DIR, 'configs')

current_dir = os.getcwd()
# Begin the main function and instatiate Hydra class
# config_path=None allows --config-dir to work properly
@hydra.main(config_path=None, config_name='config', version_base='1.1')
def main(args):
    """
    Main function to run the qprofiler. It initializes logging, sets up the environment, and processes datasets.
    The function reads datasets from the specified folder, applies feature encoding, splits the data into training and test sets,
    applies scaling and embeddings, and evaluates the models using various quantum machine learning methods.
    It logs the results and saves them in a structured format for further analysis. 
    The function also handles parallel processing of multiple machine learning methods and datasets.

    Args:
        args (dict): Configuration parameters for the profiler, including dataset paths, model parameters, and evaluation settings.

    Returns:
        None
    """
    beg_time = time.time() 
    log = logging.getLogger(__name__)
    log.info(f"Main program initiated")
    log.info(f"The number of ML methods being parallelized is {min(args['n_jobs'], len(args['model']))}")
    log.info(f"Chosen backend for quantum algorithms is: {args['backend']}") 
    # Normalize path separators for cross-platform compatibility
    folder_path = args['folder_path'].replace('/', os.sep).replace('\\', os.sep)
    path_to_input = os.path.join(dir_home, folder_path)
    if args['file_dataset'] == 'ALL':
        input_files = [file for file in os.listdir(path_to_input) if file.endswith('csv')]
    else:
        input_files = [file for file in os.listdir(path_to_input) if file in args['file_dataset'] and file.endswith('csv')]
    
    # need to populate raw data evaluation for each file, so start an empty list
    appended_raw_data_eval = []
    
    # start looping over datasets
    # start count
    file_count = 0 
    for file in sorted(input_files):
        print(f"Processing file: {file}")
        # this is where the seed needs to be set so the splits are consistent
        np.random.seed(args['seed']) 
        algorithm_globals.random_seed = args['q_seed']

        dataset_start_time = time.time()
        # dataset timestamps (may or may not be neccessary, since hydra already timestamps every log?)
        TIMESTAMP = datetime.now(timezone.utc)
        dataset_timestamp_str = TIMESTAMP.strftime("%Y_%m_%d_%H_%M_%S_%f")
        #
        summary = {}
        model_results = {}
        summary.update({'Dataset':file})
        model_results.update({'Dataset':file})
        
        # Load data with optional index column support
        if args.get('index_col', False):
            # First column contains row names/IDs
            rawdata = pd.read_csv(os.path.join(path_to_input, file), sep=r'\t|,', index_col=0)
            log.info(f"Loaded dataset with row names from first column")
        else:
            # Standard loading without index column
            rawdata = pd.read_csv(os.path.join(path_to_input, file), sep=r'\t|,')
        
        X = rawdata.iloc[:, :-1].to_numpy()
        y = rawdata.iloc[:,-1:].to_numpy()
        y_encoded = feature_encoding(y, feature_encoding='OrdinalEncoder')
        y_encoded = y_encoded.reshape(-1)
        y_encoded = y_encoded.astype(int)
        y_map = dict(zip(y_encoded.astype(str), y.tolist()))
        summary.update({'label_mapping': y_map})
        
        # Check for binary classification
        n_classes = len(np.unique(y_encoded))
        if n_classes != 2:
            log.warning(f"Dataset {file} has {n_classes} classes. QProfiler is currently optimized for binary classification.")
            log.warning(f"Multi-class classification support is experimental. Results may vary.")
            print(f"\n⚠️  WARNING: Dataset '{file}' has {n_classes} classes.")
            print(f"   QProfiler is currently optimized for binary classification.")
            print(f"   Multi-class support is experimental. Proceed with caution.\n")

        # call and run evaluation functions
        df_dataset = pd.DataFrame(X)
        raw_data_eval = evaluate(df_dataset, y_encoded, file)
        appended_raw_data_eval.append(raw_data_eval)

        # create csv file storing the evaluation of the raw, unembedded data
        all_raw_data_evaluation = pd.concat(appended_raw_data_eval)
        all_raw_data_evaluation.to_csv('RawDataEvaluation.csv', index=False)
        
        # log info
        log.info(f"Started processing data set {file}")
        log.info(f"Dataset has {n_classes} classes: {np.unique(y_encoded).tolist()}")
        
        use_stratify = args.get('stratify', [])
        test_size = args['test_size']
        iter = 0
        # makes number of iterations an argument from config
        for iter in range(args['iter']):
        ## run all this in a loop N_times, while leaving the seed fixed above. The train_test_split will change at each iteration, but will be based on the seed.
            iter=iter+1
            # track iteration time
            iter_start_time = time.time()
            
            # Apply stratification based on config
            # stratify can be: ['y'], ['Y'], or empty list/None for no stratification
            if use_stratify and len(use_stratify) > 0:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, stratify=y_encoded, test_size=test_size
                )
                log.info(f"Begin processing iteration (split) {iter} of {args['iter']} with stratified sampling")
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=test_size
                )
                log.info(f"Begin processing iteration (split) {iter} of {args['iter']} without stratification")
            #Scale the features
            if 'True' in args['scaling']:
                X_train = scaler_fn(X_train, scaling='MinMaxScaler')
                X_test = scaler_fn(X_test, scaling='MinMaxScaler')
        
            # Embed the training data and test data separately
            for embed in args['embeddings']:
                if embed == 'none':
                    log.info(f"No feature reduction (embedding) applied in this iteration")
                else:
                    log.info(f"Feature reduction (embedding) applied with {embed}")    
                X_train_emb, X_test_emb = get_embeddings(embed, X_train, X_test, n_components=args["n_components"], method=None)
                summary.update({'embeddings': embed})
                model_results.update({'embeddings': embed})
                
                # TODO: move PQK here as an embedding?

                # call and run evalution functions again if data is embedded, save outputs in the log file
                df_dataset = pd.DataFrame(X_train_emb)
                evaluate_data = evaluate(df_dataset, y_train, file)
                evaluate_data_listofdict = evaluate_data.to_dict(orient='records')
                evaluate_data_dict = {k: v for d in evaluate_data_listofdict for k, v in d.items()}
                # print(evaluate_data_dict)
                model_results.update(evaluate_data_dict)
                #log.info(f"\nThe characteristics of the embedding train dataset are: \n{evaluate_data}")
                summary.update({'iteration': iter})
                model_results.update({'iteration': iter})
                data_key = '_'.join( [re.sub( r'\..*', '', file ), embed, str(args["n_components"]), str(iter)])
                summary.update(model_run(X_train_emb, X_test_emb, y_train, y_test, data_key, args))
                # print(summary)
                for outerkey, outervalue in summary.items():
                    # print (outerkey, outervalue)
                    if outerkey.startswith("results_"):
                        for inner_key, inner_value in outervalue[0].items():
                            # print(f"{inner_key}: {inner_value}")
                            # model_results[inner_key]=inner_value
                            update = {inner_key:inner_value}
                            model_results.update(**update)
                            # Save model_results data
                        with open('ModelResults.csv', 'a', newline='') as csvfile:
                            model_results_write = csv.writer(csvfile)
                            if csvfile.tell() == 0:
                                model_results_write.writerow(model_results.keys())
                            model_results_write.writerow(model_results.values())
                # Read existing summary data from the file, if any
                try:
                    with open("results.pkl", "rb") as pklfile:
                        results = pickle.load(pklfile)
                except FileNotFoundError:
                    results = []
                # #Append the list with new summary data
                results.append(summary)
                # Save summary data
                with open('results.pkl', 'wb') as pklfile:
                    pickle.dump(results, pklfile)
            iter_run_time = time.time() - iter_start_time
            
        # start logging times
            log.info(f"The run time for iteration (split) {iter} is: {iter_run_time}")
            
        file_count += 1
        dataset_run_time = time.time() - dataset_start_time
        log.info(f"The total run time for data set {file} is: \n{dataset_run_time}")
        log.info(f"Program has processed {file_count} out of {len(input_files)} data sets")
        log.info(f"Program has {len(input_files)-file_count} data sets left to process")
    
    # log total run time of entire job
    total_run_time = time.time() - beg_time
    log.info(f"\nThe total run time of program is: \n{total_run_time}")

if __name__ == "__main__":
    main()

