from pathlib import Path

from pathlib import Path
from typing import Tuple, Union
import warnings
from sklearn.metrics import log_loss

from .feature_generation_pipeline import Parameters, generate_modelling_data

from .pandemic_model import train_model,test_model, run_logistic_baseline

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import auc, plot_precision_recall_curve
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd


def fit(
    person_data_path: Path,
    household_data_path: Path,
    residence_location_data_path: Path,
    activity_location_data_path: Path,
    activity_location_assignment_data_path: Path,
    population_network_data_path: Path,
    disease_outcome_data_path: Path,
    model_dir: Path,
) -> None:
    """Function that fits your model on the provided training data and saves
    your model to disk in the provided directory.

    Args:
        person_data_path (Path): Path to CSV data file for the Person table.
        household_data_path (Path): Path to CSV data file for the House table.
        residence_location_data_path (Path): Path to CSV data file for the
            Residence Locations table.
        activity_location_data_path (Path): Path to CSV data file for the
            Activity Locations on table.
        activity_location_assignment_data_path (Path): Path to CSV data file
            for the Activity Location Assignments table.
        population_network_data_path (Path): Path to CSV data file for the
            Population Network table.
        disease_outcome_data_path (Path): Path to CSV data file for the Disease
            Outcome table.
        model_dir (Path): Path to a directory that is constant between the train
            and test stages. You must use this directory to save and reload
            your trained model between the stages.
        preds_format_path (Path): Path to CSV file matching the format you must
            write your predictions with, filled with dummy values.
        preds_dest_path (Path): Destination path that you must write your test
            predictions to as a CSV file.

    Returns: None
    """
    dict_of_loc = {
        "person": person_data_path,
        "household":household_data_path,
        "residence_location":residence_location_data_path,
        "activity_location":activity_location_data_path,
        "activity_location_assignment":activity_location_assignment_data_path,
        "population_network":population_network_data_path,
        "disease_outcome_training":disease_outcome_data_path
    }
    
    train_run_Parameters = Parameters(end_day = 49,fixed_ending = 49,window_lenght = 1,is_training = True)
    final_data_train = generate_modelling_data(train_run_Parameters,dict_of_loc,tables_loaded = False,add_pats = None,client_dir= model_dir,db_name = 'my-db.duckdb')

    log_clf,scaler = run_logistic_baseline(final_data_train)
    pickle.dump(log_clf, open(str(model_dir)+'/model.pkl', 'wb'))
    pickle.dump(scaler, open(str(model_dir)+'/scaler.pkl', 'wb'))
    
    
    
    
    
    


def predict(
    person_data_path: Path,
    household_data_path: Path,
    residence_location_data_path: Path,
    activity_location_data_path: Path,
    activity_location_assignment_data_path: Path,
    population_network_data_path: Path,
    disease_outcome_data_path: Path,
    model_dir: Path,
    preds_format_path: Path,
    preds_dest_path: Path,
) -> None:
    """Function that loads your model from the provided directory and performs
    inference on the provided test data. Predictions should match the provided
    format and be written to the provided destination path.

    Args:
        person_data_path (Path): Path to CSV data file for the Person table.
        household_data_path (Path): Path to CSV data file for the House table.
        residence_location_data_path (Path): Path to CSV data file for the
            Residence Locations table.
        activity_location_data_path (Path): Path to CSV data file for the
            Activity Locations on table.
        activity_location_assignment_data_path (Path): Path to CSV data file
            for the Activity Location Assignments table.
        population_network_data_path (Path): Path to CSV data file for the
            Population Network table.
        disease_outcome_data_path (Path): Path to CSV data file for the Disease
            Outcome table.
        model_dir (Path): Path to a directory that is constant between the train
            and test stages. You must use this directory to save and reload
            your trained model between the stages.
        preds_format_path (Path): Path to CSV file matching the format you must
            write your predictions with, filled with dummy values.
        preds_dest_path (Path): Destination path that you must write your test
            predictions to as a CSV file.

    Returns: None
    """
    log_clf = pickle.load(open(str(model_dir)+'/model.pkl', 'rb'))
    scaler = pickle.load(open(str(model_dir)+'/scaler.pkl','rb'))

    dict_of_loc = {
        "person": person_data_path,
        "household":household_data_path,
        "residence_location":residence_location_data_path,
        "activity_location":activity_location_data_path,
        "activity_location_assignment":activity_location_assignment_data_path,
        "population_network":population_network_data_path,
        "disease_outcome_training":disease_outcome_data_path
    }
    
    test_run_Parameters = Parameters(end_day = 56,fixed_ending = 55,is_training=False,window_lenght = 1)
    test_data_49,inf_data_49 = generate_modelling_data(test_run_Parameters,dict_of_loc,tables_loaded = True,add_pats = None,client_dir= model_dir,db_name = 'my-db.duckdb')
    metrics_49,prediction_49 = test_model(log_clf,scaler,test_data_49,y_col='covid')
    
    
    model_final_results = pd.DataFrame({'pid':prediction_49.pid.values,'score_model':prediction_49.score.values})
    
    pred_format = pd.read_csv(preds_format_path)
    
    
    final_result = pred_format.merge(model_final_results,how = 'left', on = 'pid')[['pid','score_model']]
    final_result.fillna(0.5,inplace = True)
    
    final_result.rename(columns = {'score_model':'score'}, inplace = True)
    
    final_result.to_csv(preds_dest_path,index = False)
    
    
