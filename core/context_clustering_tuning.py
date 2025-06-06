import optuna
import numpy as np
import os
from sklearn.model_selection import cross_val_score
from context_clustering import cluster_sentences

def objective(trial):
    """
    Objective function for Optuna hyperparameter optimization
    
    Parameters:
    -----------
    trial : optuna.trial.Trial
        A single trial of hyperparameter configuration
    
    Returns:
    --------
    float
        Average coherence score (to be maximized)
    """
    # Define hyperparameter search space
    config = {
        'min_cluster_size': trial.suggest_int('min_cluster_size', 2, 20),
        'min_samples': trial.suggest_int('min_samples', 2, 20),
        'cluster_selection_epsilon': trial.suggest_float('cluster_selection_epsilon', 0.0, 0.5),
        'cluster_selection_method': trial.suggest_categorical(
            'cluster_selection_method', 
            ['eom', 'leaf']
        ),
        'ngram_range': trial.suggest_categorical(
            'ngram_range', 
            [(1, 1), (1, 2), (1, 3), (2, 3), (2, 2), (3, 3)]
        ),
        'nr_topics': trial.suggest_int('nr_topics', 3, 20)
    }
    
    # Perform clustering
    try:
        total_coherence = 0
        file_count = 0
        for filename in os.listdir('dir/'):
            if filename.endswith('.txt'):
                with open(os.path.join('dir/', filename), 'r') as f:
                    sentences = f.readlines()
                groups, coherence = cluster_sentences(sentences, config)
                total_coherence += coherence
                file_count += 1
        
        average_coherence = total_coherence / file_count if file_count > 0 else -np.inf
        
        return average_coherence
    
    except Exception as e:
        print(f"Trial failed: {e}")
        return -np.inf

def tune_hyperparameters(
    sentences, 
    n_trials=100, 
    timeout=3600,  # 1 hour timeout
    show_progress_bar=True
):
    """
    Perform hyperparameter tuning using Optuna
    
    Parameters:
    -----------
    sentences : list of str
        List of sentences to cluster
    n_trials : int, optional
        Number of trials to run
    timeout : int, optional
        Maximum time (in seconds) to run optimization
    show_progress_bar : bool, optional
        Whether to show Optuna progress bar
    
    Returns:
    --------
    dict
        Best hyperparameters and corresponding coherence score
    """
    # Create a study object and optimize the objective function
    study = optuna.create_study(direction='maximize')
    study.optimize(
        objective, 
        n_trials=n_trials, 
        timeout=timeout,
        show_progress_bar=show_progress_bar
    )
    
    # Print results
    print("Best trial:")
    trial = study.best_trial
    
    print(f"  Value (Coherence Score): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    return {
        'best_params': trial.params,
        'best_score': trial.value
    }

if __name__ == "__main__":
    # Perform hyperparameter tuning
    best_params = tune_hyperparameters(None, n_trials=50, timeout=3600)
    print(best_params)