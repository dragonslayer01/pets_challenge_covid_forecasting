import duckdb
import pandas as pd
import numpy as np
import os
import gc
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import auc, plot_precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score,recall_score
import pathlib
gc.collect()

def run_query(statement):
    cursor.execute(statement)
    
def get_data(statement):
    return cursor.execute(statement).fetch_df()

class Parameters:
    def __init__(self,window_lenght= 5, end_day = 56,fixed_ending = 49,is_training = True,look_forward = 5):
        self.window_lenght = window_lenght
        self.end_day = end_day
        self.is_training = is_training
        self.fixed_ending = fixed_ending
        self.look_forward = look_forward
        






def run_logistic_recovery(final_data):
    X = final_data.drop(['pid', 'recovery_time','person_number'],axis = 1).fillna(0)
    y = final_data['recovery_time']

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    X_std = pd.DataFrame(data = X_std,columns = X.columns)

    # Split into training and test
    X_train, X_test, y_train, y_test = train_test_split(X_std, y,
                                                        test_size=.1,
                                                        random_state=0,
                                                        stratify=y
                                                       )
    
    clf = LinearRegression(n_jobs=-1)

    clf.fit(X_train,y_train)
    
    y_score = clf.predict(X_train)
    
    average_precision = mean_squared_error(y_train, y_score)
    print('Training MSE --- ',average_precision)
    
    y_score = clf.predict(X_test)
    
    average_precision = mean_squared_error(y_test, y_score)
    print('Testing MSE --- ',average_precision)
    
    clf.fit(X_std,y)
    
    return clf,scaler



def covid_pats_timeline(train_Parameters):
    '''
    Find out truly Sympltomatic Covid Patients. First Infected day before data cut
    data cut end -> fixed_ending
    '''
    
    end_day = train_Parameters.fixed_ending
    
   
    statement = f'''
    drop table if exists covid_pats_timeline;
    create table covid_pats_timeline as
    select pid,first_recovered_day as last_day,first_infected_day as first_day
    from (
    select a.pid,b.first_infected_day,min(a.day) as first_recovered_day
    from disease_outcome_training a
    inner join 
    (
        select pid,min(day) as first_infected_day
        from disease_outcome_training
        where state = 'I'  
        group by 1
    )b
    on a.pid = b.pid and a.state = 'R' and a.day >= b.first_infected_day
    group by 1,2
    )  
    where first_day <= {end_day} and last_day <= {end_day}
    
    '''

    run_query(statement)

# train_run_Parameters = Parameters(end_day = 49,fixed_ending = 49)
# covid_pats_timeline(train_run_Parameters)

def asym_covid_pats_timeline(train_Parameters):
    
    '''
    Find out truly ASympltomatic Covid Patients. First Infected day before data cut
    data cut end -> fixed_ending
    '''
    
    end_day = train_Parameters.fixed_ending
    
    statement = f'''
    drop table if exists asym_covid_pats_timeline;
    create table asym_covid_pats_timeline as
    select pid, min(day) as last_day
    from disease_outcome_training
    where state in ('R')
    and pid not in (select pid from covid_pats_timeline)
    group by 1
    having last_day <= {end_day};
    
    '''

    run_query(statement)
    
    
# asym_covid_pats_timeline(train_run_Parameters)

def generate_covid_pats_regress():
    statement = '''
    select a.pid,sex,age,person_number,
    max(person_number) over(partition by hid) as total_family_members,
    last_day - first_day as recovery_time
    from person a
    inner join covid_pats_timeline b
    on a.pid = b.pid
    '''

    recovery_data = get_data(statement)
    return recovery_data

# recovery_data = generate_covid_pats_regress()

def generate_noncovid_pats_regress():
    statement = '''
    select a.pid,sex,age,person_number,
    max(person_number) over(partition by hid) as total_family_members,
    last_day
    from person a
    inner join asym_covid_pats_timeline b
    on a.pid = b.pid
    '''

    asym_recovery_data = get_data(statement)
    return asym_recovery_data

# asym_recovery_data = generate_noncovid_pats_regress()


def predict_revocery_time_noncovid(asym_recovery_data,clf,scaler):
    
    asym_recovery_data['recovery_time'] = clf.predict(scaler.transform(
    asym_recovery_data.drop(['pid','last_day','person_number'],axis = 1))).round(0)
    
    asym_recovery_data['first_day'] = asym_recovery_data.last_day - asym_recovery_data.recovery_time
    asym_recovery_data = asym_recovery_data[['pid','first_day','last_day']]
    
    cursor.execute('''
        drop table if exists asym_covid_pats_timeline;
        create table asym_covid_pats_timeline as
        select * from asym_recovery_data
    ''')
    
    print('Prediction Complete')
    
# predict_revocery_time_noncovid(asym_recovery_data,clf,scaler)

def merge_sympt_asympt_pats(train_Parameters,add_pats = None):
    
    '''
    sym_asym_pats -> Selected actual Sym and Aysm covid patients before datacut (fixed ending)
    covid_recovered_pats_with_predicted -> Adding additional predicted patients to the covid cohort
    Assumption -> For the time being for add_pats last day is first_day + look_forwrd days
    For Non-Covid patients last day is end_day - lookforward
    
    For additional patients -> Non-Covid Patients at end day
    
    '''
    
    end_day = train_Parameters.end_day
    fixed_ending = train_Parameters.fixed_ending
    look_forward = train_Parameters.look_forward
    window_lenght = train_Parameters.window_lenght
    
    statement = f'''
    drop table if exists sym_asym_pats;
    create table sym_asym_pats as

    select *,1 as sym
    from (
    select pid,min(day) as first_day,max(day) as last_day
    from disease_outcome_training
    where state = 'I'
    group by 1
    )
    where first_day <= {fixed_ending}

    union

    select pid,first_day,last_day,0 as sym
    from asym_covid_pats_timeline
    where first_day <= {fixed_ending}

    '''

    run_query(statement)
    
    statement = '''
    drop table if exists covid_recovered_pats_with_predicted;
    create table covid_recovered_pats_with_predicted as
    select *
    from sym_asym_pats
    
    '''
    
    add_pats_query = None
    
    if add_pats is not None:
    
        cursor.execute(f'''
            drop table if exists covid_pats_add;
            create table covid_pats_add as
            select *,first_day + {look_forward} as last_day,2 as sym from add_pats
        ''')
        
        add_pats_query = '''
        union
        select *
        from covid_pats_add
        where pid not in (select pid from sym_asym_pats)
        '''
        statement = statement + add_pats_query
    

    run_query(statement)
    
    statement = f'''
    drop table if exists covid_recovered_pats;
    create table covid_recovered_pats as
    select distinct pid,first_day,last_day
    from covid_recovered_pats_with_predicted;
    
    drop table if exists non_covid_pats;
    create table non_covid_pats as
    select pid,{end_day - look_forward} as last_day,{end_day - look_forward - window_lenght}as first_day
    from person
    where pid not in (select pid from covid_recovered_pats);
    
    drop table if exists non_covid_pats_inference;
    create table non_covid_pats_inference as
    select pid,{end_day} as last_day,{end_day - window_lenght} as first_day
    from person
    where pid not in (select pid from covid_recovered_pats);
    
    '''
    
    run_query(statement)
    
    
#merge_sympt_asympt_pats(train_run_Parameters)

def contact_network_covid():
    '''
    Adding First Infection day in the contact network
    '''
    
    statement = '''
    drop table if exists population_network_outcome;
    create table population_network_outcome as
    select pid1,pid2,first_dat_pid1,last_dat_pid1,first_dat_pid2,last_dat_pid2,count(*) as total_interactions
    from (
    select a.*,
    b.first_day as first_dat_pid1,b.last_day as last_dat_pid1,
    c.first_day as first_dat_pid2,c.last_day as last_dat_pid2
    from population_network a
    left join covid_recovered_pats b
    on a.pid1 = b.pid

    left join covid_recovered_pats c
    on a.pid2 = c.pid
    )
    group by 1,2,3,4,5,6;
    '''
    run_query(statement)
#contact_network_covid()


def window_based_feature_covid(run_Parameters):
    
    '''
    Check if covid patients have interacted with other covid patients in the time window
    '''
    
    window_lenght = run_Parameters.window_lenght
    
    statement = f'''
    drop table if exists covid_pats_interaction_feature_1;
    create table covid_pats_interaction_feature_1 as
    select pid1 as pid,first_dat_pid1 as window_end,
    max_end,min_end,sum(total_interactions) as total_interactions
    from
    (
        select *,
        case when first_dat_pid2 is not null and
                (first_dat_pid2 <= first_dat_pid1 and last_dat_pid2>=first_dat_pid1 - {window_lenght})
        then 1 else 0 end as infected_at_time,
        
        
        case 
            when first_dat_pid2 >= first_dat_pid1 -{window_lenght} 
            then first_dat_pid2 
            else first_dat_pid1 -{window_lenght}
            end as max_end,
        case 
            when last_dat_pid2 <= first_dat_pid1 
            then last_dat_pid2 else first_dat_pid1 
            end as min_end,
            
        min_end - max_end +1  as overlap
        
        from population_network_outcome
        where pid1 in (select pid from covid_recovered_pats)
    )
    where infected_at_time = 1
    group by 1,2,3,4;
    '''
    run_query(statement)
    print('Completed 1')
    
    statement = f'''
    drop table if exists covid_pats_interaction_feature_2;
    create table covid_pats_interaction_feature_2 as
    select pid2 as pid,first_dat_pid2 as window_end,
    max_end,min_end,sum(total_interactions) as total_interactions
    from
    (
        select *,
        case when first_dat_pid1 is not null and 
                (first_dat_pid1 <= first_dat_pid2 and last_dat_pid1>=first_dat_pid2 - {window_lenght})
        then 1 else 0 end as infected_at_time,
        
        case 
            when first_dat_pid1 >= first_dat_pid2 -{window_lenght} 
            then first_dat_pid1 
            else first_dat_pid2 -{window_lenght}
            end as max_end,
        case 
            when last_dat_pid1 <= first_dat_pid2
            then last_dat_pid1 else first_dat_pid2
            end as min_end,
            
        min_end - max_end +1 as overlap
        
        from population_network_outcome
        where pid2 in (select pid from covid_recovered_pats)
    )
    where infected_at_time = 1
    group by 1,2,3,4;
    '''
    run_query(statement)
    print('Completed 2')
    
    statement = '''
    drop table if exists covid_pats_interaction_feature;
    create table covid_pats_interaction_feature as
    select pid,window_end,
    max_end,min_end,sum(total_interactions) as total_interactions
    from (
    select * from covid_pats_interaction_feature_1
    union
    select * from covid_pats_interaction_feature_2
    )
    group by 1,2,3,4;
    '''
    run_query(statement)
    print('Completed 3')
    
def window_based_feature_noncovid(run_Parameters):
    
    '''
    Check if non covid patients have interacted with covid patients in the time period
    time period -> end_day - lookforward - window lenght to end_day - lookforward
    '''
    
    look_forward = run_Parameters.look_forward
    window_lenght = run_Parameters.window_lenght
    end_day = run_Parameters.end_day
    
    
    statement = f'''
    drop table if exists noncovid_pats_interaction_feature_1;
    create table noncovid_pats_interaction_feature_1 as
    select pid1 as pid,{end_day-look_forward} as window_end,
    max_end,min_end,sum(total_interactions) as total_interactions
    from
    (
        select *,
        case when first_dat_pid2 is not null and
                (first_dat_pid2 <= {end_day - look_forward} 
                and last_dat_pid2>= {end_day - look_forward -window_lenght})
        then 1 else 0 end as infected_at_time,
        
        case 
            when first_dat_pid2 >= {end_day - look_forward -window_lenght}
            then first_dat_pid2 
            else {end_day - look_forward -window_lenght}
            end as max_end,
        case 
            when last_dat_pid2 <= {end_day - look_forward} 
            then last_dat_pid2 
            else {end_day - look_forward} 
            end as min_end,
            
        min_end - max_end + 1 as overlap
        
        from population_network_outcome
        where pid1 not in (select pid from covid_recovered_pats)
    )
    where infected_at_time = 1
    group by 1,2,3,4;
    '''
    run_query(statement)
    print('Completed 1')
    
    statement = f'''
    drop table if exists noncovid_pats_interaction_feature_2;
    create table noncovid_pats_interaction_feature_2 as
    select pid2 as pid,{end_day-look_forward} as window_end,
    max_end,min_end,sum(total_interactions) as total_interactions
    from
    (
        select *,
        case when first_dat_pid1 is not null and 
                (first_dat_pid1 <= {end_day - look_forward}
                and last_dat_pid1>={end_day - look_forward -window_lenght})
        then 1 else 0 end as infected_at_time,
        
        case 
            when first_dat_pid1 >= {end_day - look_forward -window_lenght} 
            then first_dat_pid1 
            else {end_day - look_forward -window_lenght}
            end as max_end,
        case 
            when last_dat_pid1 <= {end_day - look_forward} 
            then last_dat_pid1 
            else {end_day - look_forward} 
            end as min_end,
            
        min_end - max_end + 1 as overlap
        
        from population_network_outcome
        where pid2 not in (select pid from covid_recovered_pats)
    )
    where infected_at_time = 1
    group by 1,2,3,4;
    '''
    run_query(statement)
    
    print('Completed 2')
    
    statement = '''
    drop table if exists noncovid_pats_interaction_feature;
    create table noncovid_pats_interaction_feature as
    select pid,window_end,
    max_end,min_end,sum(total_interactions) as total_interactions
    from (
    select * from noncovid_pats_interaction_feature_1
    union
    select * from noncovid_pats_interaction_feature_2
    )
    group by 1,2,3,4;
    '''
    run_query(statement)
    print('Completed 3')
    
def window_based_feature_noncovid_inference(run_Parameters):
    
    '''
    Check if non covid patients have interacted with covid patients in the time period (for inference phase)
    time period -> end_day - window lenght to end_day
    '''
    
    look_forward = 0
    window_lenght = run_Parameters.window_lenght
    end_day = run_Parameters.end_day
    
    
    statement = f'''
    drop table if exists noncovid_pats_inference_interaction_feature_1;
    create table noncovid_pats_inference_interaction_feature_1 as
    select pid1 as pid,{end_day-look_forward} as window_end,
    max_end,min_end,sum(total_interactions) as total_interactions
    from
    (
        select *,
        case when first_dat_pid2 is not null and
                (first_dat_pid2 <= {end_day - look_forward} 
                and last_dat_pid2>= {end_day - look_forward -window_lenght})
        then 1 else 0 end as infected_at_time,
        
        case 
            when first_dat_pid2 >= {end_day - look_forward -window_lenght} 
            then first_dat_pid2 
            else {end_day - look_forward -window_lenght}
            end as max_end,
        case 
            when last_dat_pid2 <= {end_day - look_forward} 
            then last_dat_pid2 
            else {end_day - look_forward} 
            end as min_end,
            
        min_end - max_end + 1 as overlap
        
        from population_network_outcome
        where pid1 not in (select pid from covid_recovered_pats)
    )
    where infected_at_time = 1
    group by 1,2,3,4;
    '''
    run_query(statement)
    print('Completed 1')
    
    statement = f'''
    drop table if exists noncovid_pats_inference_interaction_feature_2;
    create table noncovid_pats_inference_interaction_feature_2 as
    select pid2 as pid,{end_day-look_forward} as window_end,
    max_end,min_end,sum(total_interactions) as total_interactions
    from
    (
        select *,
        case when first_dat_pid1 is not null and 
                (first_dat_pid1 <= {end_day - look_forward}
                and last_dat_pid1>={end_day - look_forward -window_lenght})
        then 1 else 0 end as infected_at_time,
        
        case 
            when first_dat_pid1 >= {end_day - look_forward -window_lenght} 
            then first_dat_pid1 
            else {end_day - look_forward -window_lenght}
            end as max_end,
        case 
            when last_dat_pid1 <= {end_day - look_forward} 
            then last_dat_pid1 
            else {end_day - look_forward} 
            end as min_end,
            
        min_end - max_end + 1 as overlap
        
        from population_network_outcome
        where pid2 not in (select pid from covid_recovered_pats)
    )
    where infected_at_time = 1
    group by 1,2,3,4;
    '''
    run_query(statement)
    
    print('Completed 2')
    
    statement = '''
    drop table if exists noncovid_pats_inference_interaction_feature;
    create table noncovid_pats_inference_interaction_feature as
    select pid,window_end,
    max_end,min_end,sum(total_interactions) as total_interactions
    from (
    select * from noncovid_pats_inference_interaction_feature_1
    union
    select * from noncovid_pats_inference_interaction_feature_2
    )
    group by 1,2,3,4;
    '''
    run_query(statement)
    print('Completed 3')

    
def calculate_day_level_feature(run_Parameters,day = 1,day_str = 'lag_day',
                                table = 'pats_interaction_feature',feature = 'total_interactions',
                               final_feature_name = 'infected_interactions',
                               master_table = 'pats_interaction_feature_lag'):
    '''
    Calculate Feature at each day (selected by user) of the window_lenght
    '''
    window_lenght = run_Parameters.window_lenght
    
    
    statement = f'''
    drop table if exists {table}_{day_str}_{day};
    create table {table}_{day_str}_{day} as
    select pid,{day} as {day_str},{final_feature_name}_{day_str}_{day}
    from (
    select pid,sum(is_present*{feature}) as {final_feature_name}_{day_str}_{day}
    from (
    select *,
    case when window_end - {day} between max_end and min_end then 1 else 0 end as is_present
    from {table}
    )
    group by 1
    )
    '''
    
    run_query(statement)
    
    statement = f'''
    drop table if exists {master_table}_temp;
    create table {master_table}_temp as
    select * from {master_table};
    
    drop table if exists {master_table};
    create table {master_table} as
    select a.*,b.{final_feature_name}_{day_str}_{day}
    from {master_table}_temp a
    left join {table}_{day_str}_{day} b
    on a.pid = b.pid;
    
    drop table if exists {master_table}_temp;
    
    '''
    run_query(statement)
    
    return f'{final_feature_name}_{day_str}_{day}'
    
    # Creating final big table
    
#calculate_day_level_feature(train_run_Parameters)


def interaction_based_features(run_Parameters):
    contact_network_covid()
    window_based_feature_covid(run_Parameters)
    window_based_feature_noncovid(run_Parameters)
    
    if run_Parameters.is_training == False:
        pass
    else:
        window_based_feature_noncovid_inference(run_Parameters)
    
    statement = '''
    drop table if exists pats_interaction_feature;
    create table pats_interaction_feature as
    select * from covid_pats_interaction_feature
    union
    select * from noncovid_pats_interaction_feature;
    
    drop table if exists pats_interaction_feature_lag;
    create table pats_interaction_feature_lag as
    select distinct pid from pats_interaction_feature;
    
    
    '''
    if run_Parameters.is_training == False:
        pass
    else:
        statement += '''drop table if exists noncovid_pats_inference_interaction_feature_lag;
    create table noncovid_pats_inference_interaction_feature_lag as
    select distinct pid from noncovid_pats_inference_interaction_feature;'''
        
    run_query(statement)
    
    feat_list = list()
    
    for i in range(run_Parameters.window_lenght):
        feature_1 = calculate_day_level_feature(run_Parameters,day = i,table = 'pats_interaction_feature',
                                   feature = 'total_interactions',
                                   final_feature_name = 'infected_interactions',
                                   master_table = 'pats_interaction_feature_lag')
        
        if run_Parameters.is_training == True:
            feature_2 = calculate_day_level_feature(run_Parameters,day = i,table = 'noncovid_pats_inference_interaction_feature',
                                   feature = 'total_interactions',
                                   final_feature_name = 'infected_interactions',
                                   master_table = 'noncovid_pats_inference_interaction_feature_lag')
        
        
        feat_list.append(feature_1)
    
    
    print('Interaction Based Features Completed')
    
    return feat_list
    
#interaction_feats = interaction_based_features(train_run_Parameters)

def create_location_based_feature(run_Parameters):
    
    '''
    Calculate number of covid patients visited the lcoation in the window time period
    '''
    window_lenght = run_Parameters.window_lenght
    end_day = run_Parameters.end_day
    look_forward = run_Parameters.look_forward
    
    statement = f'''
    drop table if exists location_based_feature_covid;
    create table location_based_feature_covid as
    select pid,first_day as window_end,max_end,min_end,count(*) as total_interaction,
    sum(infected_duration) as sum_infected_duration
    
    from(
    
    select a.*,b.duration as infected_duration,
    case 
        when b.first_day >= a.first_day - {window_lenght} 
        then b.first_day 
        else a.first_day - {window_lenght}
        end as max_end,
    
    case 
        when b.last_day <= a.first_day 
        then b.last_day 
        else a.first_day
        end as min_end,
    
    min_end - max_end + 1 as overlap
        
    from location_assignment_covid a
    inner join location_assignment_covid b
    on a.lid = b.lid and a.pid <> b.pid 
    and b.first_day <=  a.first_day and b.last_day >= a.first_day - {window_lenght}
    )group by 1,2,3,4
    '''
    
    run_query(statement)
    print('Completed 1')
    
    
    statement = f'''
    drop table if exists location_based_feature_noncovid;
    create table location_based_feature_noncovid as
    select pid,{end_day-look_forward} as window_end,max_end,min_end,count(*) as total_interaction,
    sum(infected_duration) as sum_infected_duration
    
    from(
    
    select a.*,b.duration as infected_duration,
    case 
        when b.first_day >= {end_day-look_forward-window_lenght} 
        then b.first_day 
        else {end_day-look_forward-window_lenght}
        end as max_end,
    
    case 
        when b.last_day <= {end_day-look_forward} 
        then b.last_day 
        else {end_day-look_forward}
        end as min_end,
    
    min_end - max_end + 1 as overlap
    
    from location_assignment_noncovid a
    inner join location_assignment_covid b
    on a.lid = b.lid and a.pid <> b.pid 
    and b.first_day <=  {end_day-look_forward} and b.last_day >= {end_day-look_forward-window_lenght}
    and a.pid not in (select pid from covid_recovered_pats)
    and b.pid in (select pid from covid_recovered_pats)
    )group by 1,2,3,4;
    
    --################################## For Inference ####################
    
    drop table if exists location_based_feature_noncovid_inference;
    create table location_based_feature_noncovid_inference as
    select pid,{end_day} as window_end,max_end,min_end,count(*) as total_interaction,
    sum(infected_duration) as sum_infected_duration
    
    from(
    
    select a.*,b.duration as infected_duration,
    case 
        when b.first_day >= {end_day-window_lenght} 
        then b.first_day 
        else {end_day-window_lenght} 
        end as max_end,
    
    case 
        when b.last_day <= {end_day} 
        then b.last_day 
        else {end_day}
        end as min_end,
    
    min_end - max_end + 1 as overlap
    
    from location_assignment_noncovid_inference a
    inner join location_assignment_covid b
    on a.lid = b.lid and a.pid <> b.pid 
    and b.first_day <= {end_day} and b.last_day >= {end_day-window_lenght}
    and a.pid not in (select pid from covid_recovered_pats)
    and b.pid in (select pid from covid_recovered_pats)
    )group by 1,2,3,4;
    
    '''
    
    run_query(statement)
    print('Completed 2')
    
    statement = '''
    drop table if exists location_based_feature_total;
    create table location_based_feature_total as
    select a.pid,sum(total_loc) as total_loc,
    sum(total_interaction_loc) as total_interaction_loc
    from activity_location_assignment a
    inner join 
    (
        select lid,count(pid) as total_loc,sum(duration/60) as total_interaction_loc
        from activity_location_assignment
        group by 1
    )b
    on a.lid = b.lid
    group by 1
    '''
    
    run_query(statement)
    print('Completed 3')
    
    
    statement='''
    
    drop table if exists location_based_feature_final;
    create table location_based_feature_final as
    
    select *
    from location_based_feature_covid
    
    union
    
    select *
    from location_based_feature_noncovid
    
    '''
    
    run_query(statement)
    print('Completed 4')
    
def location_based_features(run_Parameters):
    
    statement = '''
    drop table if exists location_assignment_covid;
    create table location_assignment_covid as
    select a.*,b.first_day,b.last_day
    from activity_location_assignment a
    inner join covid_recovered_pats b
    on a.pid = b.pid;
    
    drop table if exists location_assignment_noncovid;
    create table location_assignment_noncovid as
    select a.*,b.first_day,b.last_day
    from activity_location_assignment a
    inner join non_covid_pats b
    on a.pid = b.pid;
    
    drop table if exists location_assignment_noncovid_inference;
    create table location_assignment_noncovid_inference as
    select a.*,b.first_day,b.last_day
    from activity_location_assignment a
    inner join non_covid_pats_inference b
    on a.pid = b.pid;
    
    '''
    run_query(statement)
    
    feat_list = list()
    
    create_location_based_feature(run_Parameters)
    
    statement = '''
    drop table if exists location_based_feature_final_lag;
    create table location_based_feature_final_lag as
    select distinct pid from location_based_feature_final;
    
    drop table if exists location_based_feature_noncovid_inference_lag;
    create table location_based_feature_noncovid_inference_lag as
    select distinct pid from location_based_feature_noncovid_inference;
    '''
    run_query(statement)
    
    for i in range(run_Parameters.window_lenght):
        feature_1 = calculate_day_level_feature(run_Parameters,day = i,table = 'location_based_feature_final',
                                   feature = 'total_interaction',
                                   final_feature_name = 'infected_interactions_loc',
                                               master_table = 'location_based_feature_final_lag')
        
        feature_2 = calculate_day_level_feature(run_Parameters,day = i,table = 'location_based_feature_noncovid_inference',
                                   feature = 'total_interaction',
                                   final_feature_name = 'infected_interactions_loc',
                                               master_table = 'location_based_feature_noncovid_inference_lag')
    
        
        
        feat_list.append(feature_1)
        
    print('Location Based Features Created')
    
    return feat_list
    

#loca_feats = location_based_features(train_run_Parameters)

def create_family_feature(run_Parameters):
    
    '''
    Calculate number of covid patients in the family in the window time period
    '''
    window_lenght = run_Parameters.window_lenght
    end_day = run_Parameters.end_day
    look_forward = run_Parameters.look_forward
    
    
    
    statement = f'''
    drop table if exists covid_recovered_pats_family;
    create table covid_recovered_pats_family as
    select pid,first_day as window_end,max_end,min_end,count(*) as family_infected
    from (
    
    select a.*,
    case 
        when b.first_day >= a.first_day - {window_lenght} 
        then b.first_day 
        else a.first_day - {window_lenght}
        end as max_end,
    
    case 
        when b.last_day <= a.first_day 
        then b.last_day 
        else a.first_day
        end as min_end,
    
    min_end - max_end + 1 as overlap
    
    from covid_recovered_pats_person a
    inner join covid_recovered_pats_person b
    on a.hid = b.hid and a.pid <> b.pid and
    b.first_day <= a.first_day and b.last_day >=  a.first_day - {window_lenght}
    )group by 1,2,3,4
    '''
    run_query(statement)
    
    statement = f'''
    drop table if exists noncovid_recovered_pats_family;
    create table noncovid_recovered_pats_family as
    select pid,{end_day - look_forward} as window_end,max_end,min_end,count(*) as family_infected
    from (
    
    select a.*,
    case
        when b.first_day >= {end_day - look_forward - window_lenght}
        then b.first_day
        else {end_day - look_forward - window_lenght}
        end as max_end,
    
    case
        when b.last_day <= {end_day - look_forward}
        then b.last_day
        else {end_day - look_forward}
        end as min_end,
        
    min_end - max_end + 1 as overlap
    
    from person a
    inner join covid_recovered_pats_person b
    on a.hid = b.hid and a.pid <> b.pid 
    and a.pid not in (select pid from covid_recovered_pats) and
    b.first_day <= {end_day - look_forward}
    and b.last_day >= {end_day - look_forward - window_lenght}
    )
    group by 1,2,3,4
    having pid not in (select pid from covid_recovered_pats_family);
    
    --############### For Inference ###########################
    
    drop table if exists noncovid_recovered_pats_family_inference;
    create table noncovid_recovered_pats_family_inference as
    select pid,{end_day} as window_end,max_end,min_end,count(*) as family_infected
    from (
    
    select a.*,
    case
        when b.first_day >= {end_day - window_lenght}
        then b.first_day
        else {end_day - window_lenght}
        end as max_end,
    
    case
        when b.last_day <= {end_day}
        then b.last_day
        else {end_day}
        end as min_end,
        
    min_end - max_end + 1 as overlap
    
    from person a
    inner join covid_recovered_pats_person b
    on a.hid = b.hid and a.pid <> b.pid
    and a.pid not in (select pid from covid_recovered_pats) and
    b.first_day <= {end_day}
    and b.last_day >= {end_day - window_lenght}
    )
    group by 1,2,3,4
    having pid not in (select pid from covid_recovered_pats_family);
    '''
    run_query(statement) 
    
    statement = '''
    drop table if exists family_feature;
    create table family_feature as
    select * from covid_recovered_pats_family
    union
    select * from noncovid_recovered_pats_family
    '''
    
    run_query(statement)
    
#create_family_feature()

def family_based_features(run_Parameters):
    
    statement = '''
    drop table if exists covid_recovered_pats_person;
    create table covid_recovered_pats_person as
    select a.*,first_day,last_day
    from person a
    inner join covid_recovered_pats b
    on a.pid = b.pid;
    
    '''
    run_query(statement)
    
    create_family_feature(run_Parameters)
    
    statement = '''
    drop table if exists family_feature_lag;
    create table family_feature_lag as
    select distinct pid from family_feature;
    
    drop table if exists noncovid_recovered_pats_family_inference_lag;
    create table noncovid_recovered_pats_family_inference_lag as
    select distinct pid from noncovid_recovered_pats_family_inference;
    '''
    
    run_query(statement)
    
    feat_list = list()
    
    
    for i in range(run_Parameters.window_lenght):
        feature_1 = calculate_day_level_feature(run_Parameters,day = i,table = 'family_feature',
                                   feature = 'family_infected',
                                   final_feature_name = 'family_infected',
                                    master_table = 'family_feature_lag'
                                   )
        
        feature_2 = calculate_day_level_feature(run_Parameters,day = i,table = 'noncovid_recovered_pats_family_inference',
                                   feature = 'family_infected',
                                   final_feature_name = 'family_infected',
                                   master_table = 'noncovid_recovered_pats_family_inference_lag')
        
        feat_list.append(feature_1)
       
    print('Family Based Features Completes')
    
    return feat_list
    
#family_feats = family_based_features(train_run_Parameters)

def create_train_test_pats(run_Parameters):
    
    '''
    During Training
        Positive - All Covid Patients (Sym and Asym) in time period (window_lenght to end_day)
        Negative - All Non Covid Patients with first_day (end_day - look_forward)
    
    During Testing
        Positive - All Covid Patients (Sym and Asym) on end day (current time)
        Negative - All Non Covid Patients with first_day (current time - look_forward)
    
    During Inference
        Predict model on all Non-Covid Patients on current day
        
    '''
    
    window_lenght = run_Parameters.window_lenght
    end_day = run_Parameters.end_day
    look_forward = run_Parameters.look_forward
    fixed_ending = run_Parameters.fixed_ending
    
    
    statement = f'''
    drop table if exists covid_recovered_pats_training;
    create table covid_recovered_pats_training as
    
    select distinct pid,1 as covid
    from covid_recovered_pats
    where first_day <= {fixed_ending}
    --where first_day between {window_lenght} and {fixed_ending}
    
    union
    
    select distinct pid,0 as covid
    from non_covid_pats
    where pid not in (select pid from covid_recovered_pats)

    '''
    run_query(statement)
    
    statement = f'''
    drop table if exists covid_recovered_pats_current_day;
    create table covid_recovered_pats_current_day as
    select pid,1 as covid
    from(
    select pid,min(day) as first_day
    from disease_outcome_training
    where state in ('I','R')
    group by 1
    )
    where first_day = {end_day}

    '''
    run_query(statement)
    
    
    statement = f'''
    drop table if exists covid_recovered_pats_testing;
    create table covid_recovered_pats_testing as
    
    select * from covid_recovered_pats_current_day
    where pid not in (select pid from covid_recovered_pats)
    
    union
    
    select pid,0 as covid
    from non_covid_pats
    where pid not in (select pid from covid_recovered_pats)
    and pid not in (select pid from covid_recovered_pats_current_day)
    '''
    run_query(statement)
    
    
    statement =  '''
    drop table if exists covid_recovered_pats_inference;
    create table covid_recovered_pats_inference as
    select distinct pid,0 as covid
    from non_covid_pats
    where pid not in (select pid from covid_recovered_pats)
    and pid not in (select pid from covid_recovered_pats_current_day)
    '''
    
    run_query(statement)

#create_train_test_pats()

def generate_master_data(feature_list,is_inference = False):
    
    indexes = list(feature_list.keys())
    
    add_name = ''
    if is_inference:
        add_name = '_inference'
    
    statement = f'''
    drop table if exists patient_level_features{add_name};
    create table patient_level_features{add_name} as
    select a.pid,
    age,
    sex, 
    {','.join(feature_list[indexes[0]])},
    {','.join(feature_list[indexes[1]])},
    {','.join(feature_list[indexes[2]])}
    
    from person a
    
    left join {indexes[0]} b
    on a.pid = b.pid

    left join {indexes[1]} c
    on a.pid = c.pid

    left join {indexes[2]} d
    on a.pid = d.pid

    '''
    return statement


def final_master_data(run_Parameters,train_feat_list,infer_feat_list):
    
    is_training = run_Parameters.is_training
    
    statement = generate_master_data(train_feat_list,is_inference = False)

    run_query(statement)
    
    statement = generate_master_data(infer_feat_list,is_inference = True)

    run_query(statement)
    
    statement = '''
    drop table if exists patient_level_features_master_training;
    create table patient_level_features_master_training as
    select a.*,b.covid
    from patient_level_features a
    inner join covid_recovered_pats_training b
    on a.pid = b.pid;
    '''

    run_query(statement)
    
    
    
    statement = '''
    drop table if exists patient_level_features_master_testing;
    create table patient_level_features_master_testing as
    select a.*,b.covid
    from patient_level_features a
    inner join 
    covid_recovered_pats_testing b
    on a.pid = b.pid;
    '''

    run_query(statement)
    
    
    statement = '''
    drop table if exists patient_level_features_master_inference;
    create table patient_level_features_master_inference as
    select a.*,b.covid
    from patient_level_features_inference a
    inner join covid_recovered_pats_inference b
    on a.pid = b.pid;
    '''

    run_query(statement)
    
    
    
    if is_training:
        final_data = get_data('select * from patient_level_features_master_training')
        return final_data
    else:
        final_data_1 = get_data('select * from patient_level_features_master_testing')
        final_data_2 = get_data('select * from patient_level_features_master_inference')
        
        return final_data_1,final_data_2
    
def create_table_duck_db(table,loc):
    statement = f'''
        drop table if exists {table};
        create table {table} as
        select * from read_csv_auto('{loc}');
    '''

    cursor.execute(statement)
    print('Loading --- ',table,' -- Complete')
    

def generate_modelling_data(run_Parameters,file_loc,tables_loaded = True,add_pats = None,client_dir='',db_name = 'my-db.duckdb'):
    
    
    # Data Loading
    
    global cursor
    db_path = pathlib.Path(client_dir, db_name)
    cursor = duckdb.connect(database=str(db_path),read_only=False)
    
    if tables_loaded:      
        pass
    else:
        
        for i in file_loc.keys():
            create_table_duck_db(i,file_loc[i])
    
    # Feature Generation Pipeline
    covid_pats_timeline(run_Parameters)
    asym_covid_pats_timeline(run_Parameters)
    
    # Training Linear Regression for recovery time
    recovery_data = generate_covid_pats_regress()
    clf,scaler = run_logistic_recovery(recovery_data)
    asym_recovery_data = generate_noncovid_pats_regress()
    predict_revocery_time_noncovid(asym_recovery_data,clf,scaler)
    
    # Creating Features (Interaction + Location + Family)
    merge_sympt_asympt_pats(run_Parameters,add_pats)
    interaction_feats = interaction_based_features(run_Parameters)
    loca_feats = location_based_features(run_Parameters)
    family_feats = family_based_features(run_Parameters)
    
    create_train_test_pats(run_Parameters)
    
    
    train_feat_list = {'pats_interaction_feature_lag':interaction_feats,
            'location_based_feature_final_lag':loca_feats,
            'family_feature_lag':family_feats}
    
    
    infer_feat_list = {'noncovid_pats_inference_interaction_feature_lag':interaction_feats,
                'location_based_feature_noncovid_inference_lag':loca_feats,
                'noncovid_recovered_pats_family_inference_lag':family_feats}
    
    return final_master_data(run_Parameters,train_feat_list,infer_feat_list)