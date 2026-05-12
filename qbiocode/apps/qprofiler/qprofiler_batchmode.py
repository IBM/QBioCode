# ====== Base class imports ======
import os
import json
import pandas as pd
import subprocess
import yaml
import glob
import argparse
from datetime import datetime, timezone
import time

# ======= Parallelization =======
from joblib import Parallel, delayed

# ======= checkpointing =========
from qbiocode import checkpoint_restart


def run_job(data_file, configfile, output_folder_timestamp, data_type):
    """Run QProfiler on a single dataset file with custom configuration.
    
    This function creates a temporary configuration file for each dataset by:
    1. Loading the base configuration from the specified config file
    2. Adding dataset-specific parameters (filename, timestamp, data type)
    3. Saving a new config file with a unique name
    4. Executing qprofiler with the custom configuration
    5. Cleaning up temporary config files after processing
    
    This function is designed for batch processing where multiple datasets are
    processed in parallel, each with its own configuration variant.

    Args:
        data_file (str): Name of the CSV data file to process (e.g., 'dataset1.csv')
        configfile (str): Path to the base YAML configuration file to use as template
        output_folder_timestamp (str): Timestamp string for organizing output directories
        data_type (str): Label for this batch of data (used in output directory naming)
        
    Returns:
        None
        
    Example:
        >>> run_job('cancer_data.csv', 'configs/base.yaml', '2024_01_15_120000', 'cancer_study')
        # Creates configs/config_cancer_study_2024_01_15_120000__cancer_data.yaml
        # Runs: qprofiler --config-name=config_cancer_study_2024_01_15_120000__cancer_data
    """

    ## edit YAML    

    # Read the YAML file
    
    with open(configfile, "r+") as yaml_file:
        data = yaml.safe_load(yaml_file)
        # add timestamp to output dir key of config file
        data['timestamp'] = output_folder_timestamp
        data['data_type'] = data_type

    # Modify the entry
    data["file_dataset"] = data_file

    # Write the updated data back to the file
    config_name = 'config_'+data_type+'_'+output_folder_timestamp+'__'+data_file.replace('.csv','').replace('.txt','')
    config_dir = os.path.abspath('configs')
    config_file = os.path.join(config_dir, config_name + '.yaml')
    
    # Ensure configs directory exists
    os.makedirs(config_dir, exist_ok=True)
    
    with open(config_file, "w") as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)
    
    commands = ["qprofiler", f"--config-dir={config_dir}", f"--config-name={config_name}"]
    subprocess.run(commands)


def parse_args():
    """Parse command-line arguments for batch mode processing.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description='QProfiler Batch Mode - Process multiple datasets in parallel',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with defaults
  qprofiler-batch
  
  # Custom input directory and config
  qprofiler-batch --input-dir data/my_datasets --config configs/my_config.yaml
  
  # Parallel processing with 4 jobs
  qprofiler-batch --input-dir data/datasets --n-jobs 4
  
  # Resume from previous run
  qprofiler-batch --input-dir data/datasets --checkpoint results/batch_2024_01_15
  
  # Custom data type label
  qprofiler-batch --input-dir data/cancer_data --data-type cancer_study
        """
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/tutorial_test_data/lower_dim_datasets',
        help='Path to directory containing input CSV datasets (default: data/tutorial_test_data/lower_dim_datasets)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/basic_config.yaml',
        help='Path to base configuration YAML file (default: configs/basic_config.yaml)'
    )
    
    parser.add_argument(
        '--data-type',
        type=str,
        default='test_data',
        help='Label for this batch of data (used in output directory naming) (default: test_data)'
    )
    
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=1,
        help='Number of parallel jobs to run (default: 1)'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to previous results directory to resume from (optional)'
    )
    
    return parser.parse_args()


def main():
    """Main function to run qprofiler in batch mode. It sets up the environment, processes datasets in parallel, and collects results.
    This function is designed to handle multiple datasets efficiently, allowing for parallel processing of machine learning methods and datasets.
    
    Args:
        None (uses command-line arguments)
    Returns:
        None
    """
    # Parse command-line arguments
    args = parse_args()
    
    # Set up parameters from arguments
    input_data_path = args.input_dir
    configfile = args.config
    data_type = args.data_type
    n_jobs = args.n_jobs
    checkpoint_dir = args.checkpoint
    
    # Generate timestamp for this batch run
    output_folder_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H_%M_%S_%f")
    beg_time = time.time()
    
    # Validate inputs
    if not os.path.exists(input_data_path):
        raise FileNotFoundError(f"Input directory not found: {input_data_path}")
    
    if not os.path.exists(configfile):
        raise FileNotFoundError(f"Config file not found: {configfile}")
    
    current_dir = os.getcwd()
    path_to_input = os.path.join(current_dir, input_data_path)
    
    print(f"QProfiler Batch Mode")
    print(f"=" * 60)
    print(f"Input directory: {input_data_path}")
    print(f"Config file: {configfile}")
    print(f"Data type: {data_type}")
    print(f"Parallel jobs: {n_jobs}")
    print(f"Output timestamp: {output_folder_timestamp}")
    if checkpoint_dir:
        print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"=" * 60)
    
    # Handle checkpoint restart if specified
    if checkpoint_dir:
        if not os.path.exists(checkpoint_dir):
            print(f"Warning: Checkpoint directory not found: {checkpoint_dir}")
            print("Proceeding without checkpoint...")
            completed_files = []
        else:
            print(f"Resuming from checkpoint: {checkpoint_dir}")
            completed_files = checkpoint_restart(checkpoint_dir, verbose=True)
            print(f"Found {len(completed_files)} completed datasets")
        
        # Process only incomplete datasets
        results = Parallel(n_jobs=n_jobs)(
            delayed(run_job)(file, configfile, output_folder_timestamp, data_type)
            for file in os.listdir(path_to_input)
            if file.endswith('csv') and file not in completed_files
        )
    else:
        # Process all datasets
        results = Parallel(n_jobs=n_jobs)(
            delayed(run_job)(file, configfile, output_folder_timestamp, data_type)
            for file in os.listdir(path_to_input)
            if file.endswith('csv')
        )
    
    # Collect results
    print("\nCollecting results...")
    final_model_results = pd.DataFrame()
    final_rde_results = pd.DataFrame()
    
    for file in os.listdir(path_to_input):
        if file.endswith('csv'):
            indv_results = f'results/{data_type}_batch_{output_folder_timestamp}/dataset={file}/ModelResults.csv'
            if os.path.isfile(indv_results):
                print(f"Processing results for: {file}")
                model_results = pd.read_csv(indv_results, index_col=0)
                final_model_results = pd.concat([final_model_results, model_results])
                
                rde = pd.read_csv(
                    f'results/{data_type}_batch_{output_folder_timestamp}/dataset={file}/RawDataEvaluation.csv',
                    index_col=0
                )
                final_rde_results = pd.concat([final_rde_results, rde])
                
                # Clean up temporary config files
                for f in glob.glob(f'configs/config_{output_folder_timestamp}*'):
                    os.remove(f)
            else:
                print(f"Warning: Results not found for {file}")
    
    # Save combined results
    output_dir = f'results/{data_type}_batch_{output_folder_timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    final_model_results.to_csv(f'{output_dir}/ModelResults.csv')
    final_rde_results.to_csv(f'{output_dir}/RawDataEvaluation.csv')
    
    total_time = (time.time() - beg_time) / 3600
    print(f"\n{'=' * 60}")
    print(f"Batch processing complete!")
    print(f"Total run time: {round(total_time, 2)} hours")
    print(f"Results saved to: {output_dir}")
    print(f"{'=' * 60}")
    
    return None


if __name__ == "__main__":
    main()

