import pandas as pd
import numpy as np
import os
import gc
from pathlib import Path
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def get_covid_patients_csv(dict_of_loc_absoulte, window_len):
    print("Creating covid patients dataframe")
    t1 = time.time()
    disease_outcome_training = pd.read_csv(dict_of_loc_absoulte['disease_outcome_training'])
    disease_outcome_training = disease_outcome_training[disease_outcome_training['day'] >= window_len - 1]
    affected_patients = disease_outcome_training[(disease_outcome_training.state == 'I') | (disease_outcome_training.state == 'R')]
    affected_patients_list = affected_patients.pid.unique()
    del disease_outcome_training
    affected_patients = affected_patients.groupby('pid')['day'].min().reset_index()
    
    population_network = pd.read_csv(dict_of_loc_absoulte['population_network'])
    population_network.drop(['start_time', 'duration', 'activity1',
           'activity2'], axis=1,inplace=True)
    population_network = population_network.drop_duplicates()
    population_network.shape

    population_network['pid1'] = population_network['pid1'].astype('int32')
    population_network['pid2'] = population_network['pid2'].astype('int32')
    population_network['lid'] = population_network['lid'].astype('int32')
    
    
    affected_patients.rename(columns={'pid':'pid1'}, inplace=True)
    affected_patients.set_index('pid1', inplace=True)


    population_network.set_index('pid1', inplace=True)

    affected_patients['day'] = affected_patients['day'].astype('int32')

    population_network = population_network.join(affected_patients, how='left')

    population_network.reset_index(inplace=True)

    affected_patients.reset_index(inplace=True)
    affected_patients.rename(columns={'pid1':'pid2'}, inplace=True)
    affected_patients.set_index('pid2', inplace=True)

    population_network.set_index('pid2', inplace=True)

    population_network.rename(columns={'day':'day_p1'}, inplace=True)

    population_network = population_network.join(affected_patients, how='left')

    population_network.reset_index(inplace=True)
    population_network.rename(columns={'day':'day_p2'}, inplace=True)

    population_network.fillna(-1, inplace=True)

    del affected_patients
    gc.collect()

    for col in population_network.columns:
        population_network[col] = population_network[col].astype('int32')

    population_network['infect_p2'] = np.where((population_network['day_p2'] >= 0) & (population_network['day_p1'] - population_network['day_p2'] <= window_len) & (population_network['day_p1'] - population_network['day_p2'] > 0), 1,0)
    #was p2 infected when p1 met p2 
    population_network['infect_p1'] = np.where((population_network['day_p1'] >= 0) & (population_network['day_p2'] - population_network['day_p1'] <= window_len) & (population_network['day_p2'] - population_network['day_p1'] > 0), 1,0)

    p1_agg = population_network.groupby('pid1').agg({
        'infect_p1':'count',
        'infect_p2':'sum'
    }).reset_index()

    p2_agg = population_network.groupby('pid2').agg({
        'infect_p2':'count',
        'infect_p1':'sum'
    }).reset_index()

    p1_agg.columns = ['pid','total_pat','infect_pat']
    p2_agg.columns = ['pid','total_pat','infect_pat']

    pat_agg = p1_agg.append(p2_agg)

    del p1_agg
    del p2_agg

    pat_agg = pat_agg.groupby('pid')['total_pat','infect_pat'].sum().reset_index()

    p1_loc_agg = population_network[['lid','pid1','infect_p1']].drop_duplicates()
    p1_loc_agg_n = p1_loc_agg.groupby('lid')['infect_p1'].sum().reset_index()
    p2_loc_agg = population_network[['lid','pid2','infect_p2']].drop_duplicates()
    p2_loc_agg_n = p2_loc_agg.groupby('lid')['infect_p2'].sum().reset_index()

    p1_loc_agg_n.columns = ['lid','loc_infect_pat']
    p2_loc_agg_n.columns = ['lid','loc_infect_pat']

    loc_agg = p1_loc_agg_n.append(p2_loc_agg_n)

    loc_agg = loc_agg.groupby('lid')['loc_infect_pat'].sum().reset_index()

    loc_agg.set_index('lid', inplace=True)

    p1_loc_agg.drop(['infect_p1'], axis=1, inplace=True)
    p2_loc_agg.drop(['infect_p2'], axis=1, inplace=True)

    p1_loc_agg.columns = ['lid','pid']
    p2_loc_agg.columns = ['lid','pid']

    loc_agg_n = p1_loc_agg.append(p2_loc_agg)

    del p1_loc_agg_n
    del p2_loc_agg_n

    gc.collect()

    loc_agg_n.drop_duplicates(inplace=True)

    loc_agg_n.set_index('lid', inplace=True)

    loc_agg_n = loc_agg_n.join(loc_agg, how='left')

    loc_agg_n.reset_index(inplace=True)

    del loc_agg

    pat_loc_agg = loc_agg_n.groupby('pid')['loc_infect_pat'].sum().reset_index()

    del loc_agg_n

    del p1_loc_agg
    del p2_loc_agg

    gc.collect()

    pat_loc_agg.set_index('pid', inplace=True)
    pat_agg.set_index('pid', inplace=True)

    pat_agg = pat_agg.join(pat_loc_agg, how='left')

    pat_agg.reset_index(inplace=True)

    del pat_loc_agg
    del population_network

    gc.collect()

    for col in pat_agg.columns:
        pat_agg[col] = pat_agg[col].astype('int32')



    pat_agg = pat_agg[pat_agg['pid'].isin(affected_patients_list)]


    pat_agg = pat_agg.reset_index(drop=True)
    

    gc.collect()
    t2 = time.time()
    print("Covid patient dataframe creation time: ", t2-t1)
    
    return pat_agg



def get_non_covid_patients_csv(dict_of_loc_absoulte, window_len,end_day = 49,fixed_ending = 49):
    print("Creating non covid patients dataframe")
    t1 = time.time()
    disease_outcome_training = pd.read_csv(dict_of_loc_absoulte['disease_outcome_training'])
    
    affected_patients = disease_outcome_training[(disease_outcome_training.state == 'I') | (disease_outcome_training.state == 'R')].pid.unique()


    lower_l =  fixed_ending- 2*window_len
    upper_l =  end_day- window_len


    affected_patients_win = disease_outcome_training[(disease_outcome_training['day'] >= lower_l) & (disease_outcome_training['day'] < upper_l)]


    affected_patients_win = affected_patients_win[(affected_patients_win.state == 'I') | (affected_patients_win.state == 'R')].pid.unique()


    del disease_outcome_training

    gc.collect()

    ##Load Person Table and select covid patient

    person = pd.read_csv(dict_of_loc_absoulte['person'])

    person.drop(['hid','person_number','age','sex'],axis=1, inplace=True)
    person.pid = person.pid.astype('int32')

    person['infect'] = np.where(person['pid'].isin(affected_patients_win), 1, 0)

    person['infect'] = person['infect'].astype('int32')


    population_network = pd.read_csv(dict_of_loc_absoulte['population_network'])

    population_network.drop(['start_time', 'duration', 'activity1',
           'activity2'], axis=1,inplace=True)
    population_network = population_network.drop_duplicates()

    population_network['pid1'] = population_network['pid1'].astype('int32')
    population_network['pid2'] = population_network['pid2'].astype('int32')
    population_network['lid'] = population_network['lid'].astype('int32')


    person.rename(columns={'pid':'pid1'}, inplace=True)
    person.set_index('pid1', inplace=True)

    population_network.set_index('pid1', inplace=True)

    x = population_network.join(person, how='left')

    x.reset_index(inplace=True)
    x.rename(columns={'infect':'infect_p1'}, inplace=True)

    del population_network
    gc.collect()

    person.reset_index(inplace=True)
    person.rename(columns={'pid1':'pid2'}, inplace=True)
    person.set_index('pid2', inplace=True)

    x.set_index('pid2', inplace=True)

    x = x.join(person, how='left')

    x.reset_index(inplace=True)
    x.rename(columns={'infect':'infect_p2'}, inplace=True)

    p1_agg = x.groupby('pid1').agg({
        'infect_p1':'count',
        'infect_p2':'sum'
    }).reset_index()

    p2_agg = x.groupby('pid2').agg({
        'infect_p2':'count',
        'infect_p1':'sum'
    }).reset_index()


    p1_agg.columns = ['pid','total_pat','infect_pat']
    p2_agg.columns = ['pid','total_pat','infect_pat']

    pat_agg = p1_agg.append(p2_agg)

    del p1_agg
    del p2_agg

    pat_agg = pat_agg.groupby('pid')['total_pat','infect_pat'].sum().reset_index()

    p1_loc_agg = x[['lid','pid1','infect_p1']].drop_duplicates()
    p1_loc_agg_n = p1_loc_agg.groupby('lid')['infect_p1'].sum().reset_index()
    p2_loc_agg = x[['lid','pid2','infect_p2']].drop_duplicates()
    p2_loc_agg_n = p2_loc_agg.groupby('lid')['infect_p2'].sum().reset_index()

    p1_loc_agg_n.columns = ['lid','loc_infect_pat']
    p2_loc_agg_n.columns = ['lid','loc_infect_pat']

    loc_agg = p1_loc_agg_n.append(p2_loc_agg_n)

    loc_agg = loc_agg.groupby('lid')['loc_infect_pat'].sum().reset_index()

    loc_agg.set_index('lid', inplace=True)

    p1_loc_agg.drop(['infect_p1'], axis=1, inplace=True)
    p2_loc_agg.drop(['infect_p2'], axis=1, inplace=True)

    p1_loc_agg.columns = ['lid','pid']
    p2_loc_agg.columns = ['lid','pid']

    loc_agg_n = p1_loc_agg.append(p2_loc_agg)

    del p1_loc_agg_n
    del p2_loc_agg_n

    gc.collect()

    loc_agg_n.drop_duplicates(inplace=True)

    loc_agg_n.set_index('lid', inplace=True)

    loc_agg_n = loc_agg_n.join(loc_agg, how='left')

    loc_agg_n.reset_index(inplace=True)

    del loc_agg

    pat_loc_agg = loc_agg_n.groupby('pid')['loc_infect_pat'].sum().reset_index()

    del loc_agg_n

    del p1_loc_agg
    del p2_loc_agg

    gc.collect()

    pat_loc_agg.set_index('pid', inplace=True)
    pat_agg.set_index('pid', inplace=True)

    pat_agg = pat_agg.join(pat_loc_agg, how='left')

    pat_agg.reset_index(inplace=True)

    del pat_loc_agg
    del x

    gc.collect()

    pat_agg['pid'] = pat_agg['pid'].astype('int32')
    pat_agg['total_pat'] = pat_agg['total_pat'].astype('int32')


    pat_agg = pat_agg[~pat_agg['pid'].isin(affected_patients)]

    pat_agg = pat_agg.reset_index(drop=True)
    
    t2 = time.time()
    print("Non Covid patient dataframe creation time: ", t2-t1)
    
    return pat_agg
    

def generate_modelling_data(dict_of_loc_absoulte, end_day = 49,fixed_ending = 49,window_lenght = 2,is_training = True):
    
    #dict_of_loc_absoulte= {k:client_dir.joinpath(v) for k,v in dict_of_loc.items()}
#     disease_outcome_training = pd.read_csv(dict_of_loc_absoulte['disease_outcome_training'])
    
#     population_network = pd.read_csv(dict_of_loc_absoulte['population_network'])
    ##if is_trainin False then only use non_covid_patients
    if is_training:
        covid_patients = get_covid_patients_csv(dict_of_loc_absoulte, window_len=window_lenght)
        non_covid_patients = get_non_covid_patients_csv(dict_of_loc_absoulte, window_len=window_lenght, end_day = end_day,fixed_ending = fixed_ending)
        non_covid_patients['covid'] = 0
        covid_patients['covid'] = 1
        total_patients = pd.concat([covid_patients, non_covid_patients])
    else:
        non_covid_patients = get_non_covid_patients_csv(dict_of_loc_absoulte, window_len=window_lenght, end_day = end_day,fixed_ending = fixed_ending)
        non_covid_patients['covid'] = 0

        total_patients = non_covid_patients
    
    patients = pd.read_csv(dict_of_loc_absoulte['person'])
    
    #patients.drop(['hid','person_number'], axis=1, inplace=True)
    patients.drop(['person_number'], axis=1, inplace=True)

    total_patients.set_index('pid', inplace=True)
    patients.set_index('pid', inplace=True)

    total_patients = total_patients.join(patients, how='right')
    total_patients.fillna(0, inplace=True)

    del patients
    gc.collect()

    total_patients.reset_index(inplace=True)

    for col in total_patients.columns:
        total_patients[col] = total_patients[col].astype('int32')
        
    family_infect = total_patients.groupby(['hid'])['infect_pat'].sum().reset_index().rename(columns={'infect_pat':'family_infect'})
    total_patients = total_patients.merge(family_infect, on='hid', how='left')
    total_patients.drop(['hid'], axis=1, inplace=True)
    #total_patients.drop(['hid','infect_pat'], axis=1, inplace=True)

    del non_covid_patients

    gc.collect()
    #total_patients = total_patients[['pid','age','sex','covid']]
    return total_patients
    
    

