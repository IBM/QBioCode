import os, re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
import dill as pickle
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
import json
import ast


#####


class QuantumSageV2():
    '''
    Enhanced Sage class (Version 2) that extends the original QuantumSage functionality.
    
    This version includes:
    - Model parameters as input features alongside data characteristics
    - Bidirectional prediction: predict performance from parameters OR predict parameter ranges from desired performance
    - Support for all three sage types: Random Forest, MLP, and XGBoost-Optuna
    - Enhanced prediction capabilities with parameter recommendations
    
    The class learns relationships between:
    1. Dataset characteristics + Model parameters -> Performance metrics
    2. Dataset characteristics + Desired performance -> Recommended parameter ranges
    '''

    def __init__(self, data_input, include_model_params=True):
        '''
        Initialize Sage V2 with input data containing data characteristics, model parameters, and performance metrics.
        
        Parameters
        ----------
        data_input : pd.DataFrame
            DataFrame containing data characteristics, model parameters, and performance metrics
        include_model_params : bool, optional
            Whether to include model parameters as input features (default: True)
            If False, behaves like original QuantumSage
        '''

        self._columns_data_features = [ '# Features', '# Samples',
                                        'Feature_Samples_ratio', 'Intrinsic_Dimension', 'Condition number',
                                        'Fisher Discriminant Ratio', 'Total Correlations', 'Mutual information',
                                        '# Non-zero entries', '# Low variance features', 'Variation', 'std_var',
                                        'Coefficient of Variation %', 'std_co_of_v', 'Skewness', 'std_skew',
                                        'Kurtosis', 'std_kurt', 'Mean Log Kernel Density',
                                        'Isomap Reconstruction Error', 'Fractal dimension', 'Entropy',
                                        'std_entropy']
        self._columns_metrics = ['accuracy', 'f1_score', 'auc']
        self._columns_metadata = ['Dataset', 'embeddings','datatype', 'model_embed_datatype', 'iteration', 'model']
        
        self._include_model_params = include_model_params
        self._input_data_raw = data_input.copy()
        
        # Parse model parameters if available and requested
        if include_model_params and 'Model_Parameters' in data_input.columns:
            self._param_features = self._parse_model_parameters(data_input)
            # Combine data features with parameter features
            self._input_data_features_only = pd.concat([
                data_input[self._columns_data_features],
                self._param_features
            ], axis=1)
            self._param_column_names = list(self._param_features.columns)
        else:
            self._input_data_features_only = data_input[self._columns_data_features]
            self._param_features = None
            self._param_column_names = []
        
        self._input_data_metrics = data_input[self._columns_metrics]
        self._input_data_metadata = data_input[self._columns_metadata]

        self._available_models = list(set(self._input_data_metadata['model']))
        if 'none' in self._available_models:
            self._available_models.remove('none')
        self._available_models.sort()
        self._available_embeddings = list(set(self._input_data_metadata['embeddings']))
        self._available_embeddings.sort()
        self._available_metrics = self._columns_metrics
        self._available_metrics.sort()

        self._results_subsages = {}
        self._results_inverse_sages = {}  # For inverse prediction (performance -> parameters)

        self.set_seed()

    def _parse_model_parameters(self, data_input):
        '''
        Parse Model_Parameters column (dict/JSON string) into separate feature columns.
        Handles nested dictionaries by flattening them with underscore-separated keys.
        
        Example input:
            "{'feature_map': 'ZZFeatureMap', 'feature_map_reps': 10, 'entanglement': 'pairwise',
              'svc_best_params': {'C': 1, 'gamma': 1, 'kernel': 'rbf'}}"
        
        Example output columns:
            feature_map, feature_map_reps, entanglement, svc_best_params_C,
            svc_best_params_gamma, svc_best_params_kernel
        
        Parameters
        ----------
        data_input : pd.DataFrame
            Input dataframe with Model_Parameters column
            
        Returns
        -------
        pd.DataFrame
            DataFrame with parsed and flattened parameter columns
        '''
        
        def flatten_dict(d, parent_key='', sep='_'):
            '''
            Flatten nested dictionary structure.
            
            Example:
                {'a': 1, 'b': {'c': 2, 'd': 3}} -> {'a': 1, 'b_c': 2, 'b_d': 3}
            '''
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)
        
        param_dicts = []
        
        for idx, row in data_input.iterrows():
            if pd.isna(row.get('Model_Parameters')):
                param_dicts.append({})
                continue
                
            param_str = row['Model_Parameters']
            
            # Try to parse as dict or JSON
            try:
                if isinstance(param_str, dict):
                    params = param_str
                elif isinstance(param_str, str):
                    # Clean up the string (remove extra quotes if present)
                    param_str = param_str.strip()
                    
                    # Try ast.literal_eval first (safer for Python dict strings)
                    try:
                        params = ast.literal_eval(param_str)
                    except:
                        # Try JSON as fallback
                        try:
                            params = json.loads(param_str)
                        except:
                            params = {}
                else:
                    params = {}
            except Exception as e:
                print(f"Warning: Could not parse Model_Parameters at index {idx}: {e}")
                params = {}
            
            # Flatten nested dictionaries
            if params:
                params = flatten_dict(params)
            
            param_dicts.append(params)
        
        # Convert to DataFrame
        param_df = pd.DataFrame(param_dicts)
        
        if param_df.empty:
            return param_df
        
        # Handle non-numeric columns
        for col in param_df.columns:
            if param_df[col].dtype == 'object':
                # Try to convert to numeric first
                try:
                    param_df[col] = pd.to_numeric(param_df[col], errors='coerce')
                except:
                    pass
                
                # If still object type, use label encoding for categorical values
                if param_df[col].dtype == 'object':
                    # Create a mapping for categorical values
                    unique_vals = param_df[col].dropna().unique()
                    if len(unique_vals) > 0:
                        # Use categorical codes
                        param_df[col] = pd.Categorical(param_df[col]).codes
                        # Replace -1 (NaN code) with 0
                        param_df[col] = param_df[col].replace(-1, 0)
        
        # Fill remaining NaN with 0
        param_df = param_df.fillna(0)
        
        # Ensure all columns are numeric
        for col in param_df.columns:
            if param_df[col].dtype == 'object':
                param_df[col] = 0
        
        return param_df

    def predict(self, input_data, metric='f1_score', include_params=True):
        '''
        Predict performance metric for given data characteristics and optionally model parameters.
        
        Parameters
        ----------
        input_data : pd.DataFrame
            DataFrame with data characteristics (and optionally model parameters)
        metric : str, optional
            Metric to predict (default: 'f1_score')
        include_params : bool, optional
            Whether input_data includes model parameters (default: True)
            
        Returns
        -------
        pd.DataFrame
            Predictions for each model with R2 scores
        '''
        predictions = []
        for model in self._available_models:
            pred = self._results_subsages[metric][model]['fit_model'].predict(input_data)[0]
            r2 = self._results_subsages[metric][model]['r2']
            predictions.append([model, pred, r2])
        predictions_df = pd.DataFrame(predictions, columns=['model', metric, 'r2'])
        predictions_df[metric+'*r2'] = predictions_df[metric] * predictions_df['r2']
        predictions_df = predictions_df.sort_values(metric+'*r2', ascending=False)
        return predictions_df

    def predict_parameter_ranges(self, data_characteristics, target_metric_value, metric='f1_score', 
                                 model_name=None, n_samples=1000, confidence=0.95):
        '''
        Inverse prediction: Given data characteristics and desired performance, 
        predict parameter ranges that would achieve that performance.
        
        Parameters
        ----------
        data_characteristics : pd.DataFrame or dict
            Data characteristics (single row)
        target_metric_value : float
            Desired performance metric value
        metric : str, optional
            Target metric (default: 'f1_score')
        model_name : str, optional
            Specific model to get parameters for. If None, returns for best model.
        n_samples : int, optional
            Number of samples for Monte Carlo estimation (default: 1000)
        confidence : float, optional
            Confidence level for parameter ranges (default: 0.95)
            
        Returns
        -------
        dict
            Dictionary with parameter ranges and statistics for each model
        '''
        if not self._include_model_params:
            raise ValueError("Model parameters were not included during initialization. "
                           "Create QuantumSageV2 with include_model_params=True")
        
        if metric not in self._results_inverse_sages:
            raise ValueError(f"Inverse sage not trained for metric '{metric}'. "
                           f"Call train_inverse_sages() first.")
        
        # Convert data_characteristics to DataFrame if dict
        if isinstance(data_characteristics, dict):
            data_characteristics = pd.DataFrame([data_characteristics])
        
        results = {}
        models_to_process = [model_name] if model_name else self._available_models
        
        for model in models_to_process:
            if model not in self._results_inverse_sages[metric]:
                continue
                
            # Get the trained inverse models for this model's parameters
            inverse_models = self._results_inverse_sages[metric][model]
            
            # Prepare input: data characteristics + target metric value
            input_features = data_characteristics.copy()
            input_features[f'target_{metric}'] = target_metric_value
            
            # Predict each parameter
            param_predictions = {}
            param_ranges = {}
            
            for param_name, param_model_info in inverse_models.items():
                # Predict parameter value
                pred_value = param_model_info['fit_model'].predict(input_features)[0]
                
                # Estimate uncertainty using model's training error
                rmse = param_model_info['rmse']
                
                # Calculate confidence interval
                alpha = 1 - confidence
                z_score = 1.96  # For 95% confidence
                margin = z_score * rmse
                
                param_predictions[param_name] = pred_value
                param_ranges[param_name] = {
                    'predicted': float(pred_value),
                    'lower_bound': float(pred_value - margin),
                    'upper_bound': float(pred_value + margin),
                    'rmse': float(rmse),
                    'r2': float(param_model_info['r2'])
                }
            
            results[model] = {
                'parameter_predictions': param_predictions,
                'parameter_ranges': param_ranges,
                'target_metric': metric,
                'target_value': target_metric_value
            }
        
        return results

    def train_sub_sages(self, test_size=0.2, sage_type='random_forest', n_iter=None, cv=5):
        """
        Train sub-sage predictors for each ML model and performance metric.
        
        This trains: Data Characteristics + Model Parameters -> Performance Metrics
        
        Parameters
        ----------
        test_size : float, optional
            Proportion of data for testing (default: 0.2)
        sage_type : str, optional
            Type of regressor: 'random_forest', 'mlp', or 'xgboost_optuna' (default: 'random_forest')
        n_iter : int, optional
            Number of iterations for hyperparameter search
        cv : int, optional
            Cross-validation folds (default: 5)
        """
        valid_sage_types = ['random_forest', 'mlp', 'xgboost_optuna']
        if sage_type not in valid_sage_types:
            raise ValueError(
                f"Invalid sage_type '{sage_type}'. Must be one of {valid_sage_types}."
            )
        
        for metric in self._available_metrics:
            print(f"Working on {metric}")

            self._results_subsages[metric] = {}
            for model in self._available_models:
                model_indices = self._input_data_metadata[self._input_data_metadata['model'] == model].index
                X = self._input_data_features_only.loc[model_indices]
                X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
                y = self._input_data_metrics.loc[model_indices][metric].fillna(0).to_numpy()
                
                print(f"  Working on {model}")
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=self._seed)

                # Calculate SLGH (Scaled Latent Geometric Hardness)
                X_train = calculate_SLGH(X_train, len(X_train))
                X_test = calculate_SLGH(X_test, len(X_train))

                if sage_type == 'random_forest':
                    rf_n_iter = n_iter if n_iter is not None else 50
                    self._results_subsages[metric][model] = self._sage_random_forest(
                        X_train, X_test, y_train, y_test, n_iter=rf_n_iter, cv=cv
                    )
                elif sage_type == 'mlp':
                    mlp_n_iter = n_iter if n_iter is not None else 1000
                    self._results_subsages[metric][model] = self._sage_mlp(
                        X_train, X_test, y_train, y_test, n_iter=mlp_n_iter, cv=cv
                    )
                elif sage_type == 'xgboost_optuna':
                    xgb_n_iter = n_iter if n_iter is not None else 100
                    self._results_subsages[metric][model] = self._sage_xgboost_optuna(
                        X_train, X_test, y_train, y_test, n_iter=xgb_n_iter, cv=cv
                    )

    def train_inverse_sages(self, test_size=0.2, sage_type='xgboost_optuna', n_iter=None, cv=5):
        """
        Train inverse sage predictors: Data Characteristics + Target Performance -> Model Parameters
        
        This enables predicting what parameter values would achieve a desired performance level.
        
        Parameters
        ----------
        test_size : float, optional
            Proportion of data for testing (default: 0.2)
        sage_type : str, optional
            Type of regressor: 'random_forest', 'mlp', or 'xgboost_optuna' (default: 'xgboost_optuna')
        n_iter : int, optional
            Number of iterations for hyperparameter search
        cv : int, optional
            Cross-validation folds (default: 5)
        """
        if not self._include_model_params:
            raise ValueError("Model parameters were not included during initialization. "
                           "Create QuantumSageV2 with include_model_params=True")
        
        if not self._param_column_names:
            raise ValueError("No model parameters found in the data.")
        
        valid_sage_types = ['random_forest', 'mlp', 'xgboost_optuna']
        if sage_type not in valid_sage_types:
            raise ValueError(
                f"Invalid sage_type '{sage_type}'. Must be one of {valid_sage_types}."
            )
        
        print("Training inverse sages (Performance -> Parameters)...")
        
        for metric in self._available_metrics:
            print(f"Working on {metric}")
            self._results_inverse_sages[metric] = {}
            
            for model in self._available_models:
                print(f"  Working on {model}")
                model_indices = self._input_data_metadata[self._input_data_metadata['model'] == model].index
                
                # Input: data characteristics + metric value
                X_data_chars = self._input_data_raw.loc[model_indices][self._columns_data_features]
                X_metric = self._input_data_metrics.loc[model_indices][[metric]]
                X = pd.concat([X_data_chars, X_metric], axis=1)
                X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
                X.columns = list(X_data_chars.columns) + [f'target_{metric}']
                
                self._results_inverse_sages[metric][model] = {}
                
                # Train a separate model for each parameter
                for param_name in self._param_column_names:
                    print(f"    Training for parameter: {param_name}")
                    
                    # Output: parameter value
                    y = self._param_features.loc[model_indices][param_name].fillna(0).to_numpy()
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=self._seed
                    )
                    
                    if sage_type == 'random_forest':
                        rf_n_iter = n_iter if n_iter is not None else 50
                        result = self._sage_random_forest(
                            X_train, X_test, y_train, y_test, n_iter=rf_n_iter, cv=cv
                        )
                    elif sage_type == 'mlp':
                        mlp_n_iter = n_iter if n_iter is not None else 1000
                        result = self._sage_mlp(
                            X_train, X_test, y_train, y_test, n_iter=mlp_n_iter, cv=cv
                        )
                    elif sage_type == 'xgboost_optuna':
                        xgb_n_iter = n_iter if n_iter is not None else 100
                        result = self._sage_xgboost_optuna(
                            X_train, X_test, y_train, y_test, n_iter=xgb_n_iter, cv=cv
                        )
                    
                    self._results_inverse_sages[metric][model][param_name] = result

    def _sage_mlp(self, X_train, X_test, y_train, y_test, n_iter=1000, cv=5):
        """Train MLP regressor with grid search."""
        from sklearn.model_selection import GridSearchCV

        param_grid = {
            'hidden_layer_sizes': [(32, 10), (64, 32), (100,), (50, 25)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'lbfgs'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }

        mlp = MLPRegressor(
            batch_size='auto',
            learning_rate_init=0.001,
            max_iter=n_iter,
            random_state=self._seed,
            n_iter_no_change=10,
            early_stopping=True,
            validation_fraction=0.1
        )

        mlp_grid = GridSearchCV(
            estimator=mlp,
            param_grid=param_grid,
            cv=cv,
            n_jobs=-1,
            scoring='r2'
        )
        
        X_train = X_train.astype(np.float64).replace([np.inf, -np.inf], np.nan).fillna(0)
        X_test = X_test.astype(np.float64).replace([np.inf, -np.inf], np.nan).fillna(0)
        
        mlp_grid.fit(X_train, y_train)
        preds = mlp_grid.predict(X_test)
        params = mlp_grid.best_params_

        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)

        return {
            'fit_model': mlp_grid,
            'preds': preds,
            'y_test': y_test,
            'params': params,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }

    def _sage_xgboost_optuna(self, X_train, X_test, y_train, y_test, n_iter=100, cv=5):
        """Train XGBoost regressor with Optuna optimization."""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is required. Install it with: pip install xgboost")
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required. Install it with: pip install optuna")
        
        from sklearn.model_selection import cross_val_score
        
        X_train = X_train.astype(np.float64).replace([np.inf, -np.inf], np.nan).fillna(0)
        X_test = X_test.astype(np.float64).replace([np.inf, -np.inf], np.nan).fillna(0)
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': self._seed,
                'n_jobs': -1,
                'verbosity': 0
            }
            
            model = xgb.XGBRegressor(**params)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2', n_jobs=-1)
            return scores.mean()
        
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self._seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        )
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=n_iter, show_progress_bar=False)
        
        best_params = study.best_params
        best_params['random_state'] = self._seed
        best_params['n_jobs'] = -1
        best_params['verbosity'] = 0
        
        best_model = xgb.XGBRegressor(**best_params)
        best_model.fit(X_train, y_train)
        preds = best_model.predict(X_test)
        
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)
        
        return {
            'fit_model': best_model,
            'preds': preds,
            'y_test': y_test,
            'params': best_params,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'study': study
        }

    def _sage_random_forest(self, X_train, X_test, y_train, y_test, n_iter=50, cv=5):
        """Train Random Forest regressor with randomized search."""
        param_distributions = {
            'n_estimators': np.arange(100, 1000, 100),
            'max_depth': np.arange(5, 20),
            'min_samples_split': np.arange(2, 10),
            'min_samples_leaf': np.arange(1, 5),
            'bootstrap': [True, False]
        }

        rf = RandomForestRegressor(random_state=self._seed)
        rf_random = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            random_state=self._seed,
            n_jobs=-1
        )
        
        X_train = X_train.astype(np.float64).replace([np.inf, -np.inf], np.nan).fillna(0)
        X_test = X_test.astype(np.float64).replace([np.inf, -np.inf], np.nan).fillna(0)
        
        rf_random.fit(X_train, y_train)
        preds = rf_random.predict(X_test)
        params = rf_random.best_params_

        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)

        return {
            'fit_model': rf_random,
            'preds': preds,
            'y_test': y_test,
            'params': params,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }

    def plot_results(self, figsize=(6, 4), saveFile='', plot_type='forward'):
        '''
        Plot results of trained sages.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure size (default: (6, 4))
        saveFile : str, optional
            File path to save plots (default: '')
        plot_type : str, optional
            Type of results to plot: 'forward' (performance prediction) or 
            'inverse' (parameter prediction) (default: 'forward')
        '''
        if plot_type == 'forward':
            results_dict = self._results_subsages
        elif plot_type == 'inverse':
            results_dict = self._results_inverse_sages
        else:
            raise ValueError("plot_type must be 'forward' or 'inverse'")
        
        if not results_dict:
            print(f"Warning: No {plot_type} results to plot. Train sages first.")
            return
        
        results = []
        preds = pd.DataFrame()
        
        for metric in self._available_metrics:
            if metric not in results_dict:
                continue
                
            for model in self._available_models:
                if model not in results_dict[metric]:
                    continue
                
                if plot_type == 'forward':
                    scores = pd.Series(
                        results_dict[metric][model].values(),
                        index=results_dict[metric][model].keys()
                    )
                    results.append([model, metric] + list(scores[['mae', 'mse', 'rmse', 'r2']]))
                    p = results_dict[metric][model]['preds']
                    y = results_dict[metric][model]['y_test']
                    preds = pd.concat([preds, pd.DataFrame({
                        'model': [model] * len(p),
                        'metric': [metric] * len(p),
                        'pred': p,
                        'y_test': y
                    })])
                else:  # inverse
                    # Aggregate results across parameters
                    param_results = results_dict[metric][model]
                    if param_results:
                        avg_mae = np.mean([v['mae'] for v in param_results.values()])
                        avg_mse = np.mean([v['mse'] for v in param_results.values()])
                        avg_rmse = np.mean([v['rmse'] for v in param_results.values()])
                        avg_r2 = np.mean([v['r2'] for v in param_results.values()])
                        results.append([model, metric, avg_mae, avg_mse, avg_rmse, avg_r2])
        
        if not results:
            print("Warning: No results to plot.")
            return
        
        results_df = pd.DataFrame(results, columns=['model', 'metric', 'mae', 'mse', 'rmse', 'r2'])
        results_df = results_df.melt(id_vars=['model', 'metric'])
        
        for metric in self._available_metrics:
            metric_data = results_df[results_df['metric'] == metric]
            if metric_data.empty:
                continue
                
            plt.figure(figsize=figsize)
            sns.barplot(data=metric_data, x='variable', y='value', hue='model', hue_order=self._available_models)
            plt.title(f"{plot_type.capitalize()} prediction performance for {metric}")
            plt.xlabel("Metric")
            plt.ylabel("Value")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            if saveFile:
                plt.savefig(re.sub('.pdf', '', saveFile) + f'_{plot_type}_{metric}_barplot.pdf', bbox_inches='tight')
            plt.show()
            plt.close()
            
            if plot_type == 'forward' and not preds.empty:
                toPlot = preds[preds['metric'] == metric]
                if not toPlot.empty:
                    plt.figure(figsize=figsize)
                    plt.title(f"Predictive performance for {metric}")
                    sns.scatterplot(data=toPlot, x='y_test', y='pred', hue='model')
                    plt.xlabel("Actual")
                    plt.ylabel("Predicted")
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.tight_layout()
                    if saveFile:
                        plt.savefig(re.sub('.pdf', '', saveFile) + f'_{metric}_scatterplot.pdf', bbox_inches='tight')
                    plt.show()
                    plt.close()

    def set_seed(self, seed=42):
        self._seed = seed

def calculate_SLGH(df, train_pct = 0.7):
    id_col = 'Intrinsic_Dimension'
    fdr_col = 'Fisher Discriminant Ratio'
    num_samples = '# Samples'
    n_train = np.ceil(df[num_samples] * train_pct)
    eps = 1e-8

    df = pd.DataFrame(df)
    df['SLGH'] = (-np.log(df[id_col] + eps) - np.log(1.0 + df[fdr_col] * n_train))
    return(df)

def main():
    """Command-line interface for QSage V2."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description='QSage V2: Enhanced Quantum-inspired model selection with parameter optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input', '-i', required=True,
                       help='Path to input CSV file')
    parser.add_argument('--output', '-o', required=True,
                       help='Output directory for results')
    parser.add_argument('--seed', '-s', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--model-type', default='xgboost_optuna',
                       choices=['rf', 'mlp', 'random_forest', 'xgboost', 'xgboost_optuna'],
                       help='Sage model type (default: xgboost_optuna)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set proportion (default: 0.2)')
    parser.add_argument('--n-iter', type=int, default=None,
                       help='Hyperparameter search iterations')
    parser.add_argument('--cv', type=int, default=5,
                       help='Cross-validation folds (default: 5)')
    parser.add_argument('--train-inverse', action='store_true',
                       help='Also train inverse sages (performance -> parameters)')
    parser.add_argument('--no-model-params', action='store_true',
                       help='Exclude model parameters from features')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)
    
    os.makedirs(args.output, exist_ok=True)
    
    print("="*80)
    print("QSage V2: Enhanced Quantum Model Selection Oracle")
    print("="*80)
    print(f"Input file: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Include model parameters: {not args.no_model_params}")
    print(f"Train inverse sages: {args.train_inverse}")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    data = pd.read_csv(args.input)
    # data['Model_Parameters'] = [ 'None' if str(x) == 'nan' else x for x in data['Model_Parameters'] ]
    data['embeddings'] = [ 'None' if str(x) == 'nan' else x for x in data['embeddings'] ]
    print(f"Loaded {len(data)} rows with {len(data.columns)} columns")
    
    # Initialize QSage V2
    print("\nInitializing QSage V2...")
    sage = QuantumSageV2(data, include_model_params=not args.no_model_params)
    sage.set_seed(args.seed)
    
    # Map model type
    sage_type_map = {
        'rf': 'random_forest',
        'random_forest': 'random_forest',
        'mlp': 'mlp',
        'xgboost': 'xgboost_optuna',
        'xgboost_optuna': 'xgboost_optuna'
    }
    sage_type = sage_type_map[args.model_type]
    
    # Train forward sages
    print(f"\nTraining forward sages ({sage_type})...")
    sage.train_sub_sages(
        test_size=args.test_size,
        sage_type=sage_type,
        n_iter=args.n_iter,
        cv=args.cv
    )
    
    # Train inverse sages if requested
    if args.train_inverse and not args.no_model_params:
        print(f"\nTraining inverse sages ({sage_type})...")
        sage.train_inverse_sages(
            test_size=args.test_size,
            sage_type=sage_type,
            n_iter=args.n_iter,
            cv=args.cv
        )
    
    # Save the trainted Sage
    pickle.dump(sage, file = open(os.path.join(args.output, 'sage_v2.pkl'), 'wb' ))

    # Generate plots
    print("\nGenerating plots...")
    output_file = os.path.join(args.output, 'sage_v2_results.pdf')
    sage.plot_results(saveFile=output_file, plot_type='forward')
    
    if args.train_inverse and not args.no_model_params:
        sage.plot_results(saveFile=output_file, plot_type='inverse')
    
    print("\n" + "="*80)
    print("QSage V2 training completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()

# Made with Bob
