import pandas as pd
import numpy as np


def disease_replace_func(a):
    if a == 'I':
        return 1
    else:
        return 0

def hacky_feature_banao(dict_of_loc):
    patient_df = pd.read_csv(dict_of_loc['person'])
    diseas_outcome_training = pd.read_csv(dict_of_loc['disease_outcome_training'])
    print(diseas_outcome_training.head())
    diseas_outcome_training = diseas_outcome_training[diseas_outcome_training.day == 54][['pid','state']]
    diseas_outcome_training['covid'] = diseas_outcome_training.state.apply(disease_replace_func)
    model_zindabad = patient_df[['pid','age','sex']]
    
    final_data = model_zindabad.merge(diseas_outcome_training,on='pid',how='left').fillna(0)
    return final_data[['pid','age','sex','covid']]
    
    