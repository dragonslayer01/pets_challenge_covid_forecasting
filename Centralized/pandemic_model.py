
from .feature_generation_pipeline import *
from sklearn.metrics import log_loss


def run_logistic_baseline(final_data, y = 'covid'):
    X = final_data.drop(['pid', y],axis = 1).fillna(0)
    y = final_data[y]

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    X_std = pd.DataFrame(data = X_std,columns = X.columns)

    # Split into training and test
    X_train, X_test, y_train, y_test = train_test_split(X_std, y,
                                                        test_size=.1,
                                                        random_state=0,
                                                        stratify=y
                                                       )
    
    clf = LogisticRegression(random_state=0,n_jobs=-1)
#     clf = RandomForestClassifier(random_state=0,n_jobs = -1,n_estimators= 50,class_weight = 'balanced')
#     clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
#                                      max_depth=1, random_state=0,n_jobs = -1)

    #clf = MLPClassifier(random_state=1, max_iter=300)

    clf.fit(X_train,y_train) # Change According to Pytorch ka function
    
    y_score = clf.predict_proba(X_train)[:, 1] # Probability from the pytorch function
    
    average_precision = average_precision_score(y_train, y_score)
    print('Training AUPRC --- ',average_precision)
    
    y_score = clf.predict_proba(X_test)[:, 1]
    
    average_precision = average_precision_score(y_test, y_score)
    print('Testing AUPRC --- ',average_precision)
    
    
    return clf,scaler



def test_model(clf,scaler,final_data,y_col='covid',threshold = 0.5):
    X = final_data.drop(['pid', y_col],axis = 1).fillna(0)
    y = final_data[y_col].values

    X_std = scaler.transform(X)
    X_std = pd.DataFrame(data = X_std,columns = X.columns)

    
    y_score = clf.predict_proba(X_std)[:, 1]
    
    
    average_precision = average_precision_score( y, y_score)
   
    
    pred_pats = (pd.Series(y_score) >= threshold).astype(int)
    
    prediction = pd.DataFrame(data = {'pid':final_data.pid.values,
                           'covid_predicted':pred_pats.values,
                          'score':y_score,
                          'covid_ground_truth':y})
    
    
    
    
    
    precision,recall = precision_score(y,pred_pats),recall_score(y,pred_pats)
    #log_loss = log_loss(y, y_score)
    
    print('AUPRC --- ',average_precision, ' Precision ---',precision,' Recall --- ',recall)
    
    return [average_precision,precision,recall],prediction


def daily_pred_engine(run_Parameters,log_clf,scaler,threshold = 0.5,add_pats=None,file_loc = None):
    test_data,inf_data = generate_modelling_data(run_Parameters,file_loc,tables_loaded = True,add_pats=add_pats)
    metrics,prediction = test_model(log_clf,scaler,test_data,y_col='covid')
    
    pred = pd.DataFrame(data ={'pid':inf_data.pid.values,'first_day':run_Parameters.end_day})
    pred['score'] = log_clf.predict_proba(scaler.transform(inf_data.drop('pid',axis =1).fillna(0)))[:,1]
    
    add_pats = pred[pred.score>= threshold].drop(['score'],axis = 1)
    
    
    return metrics,add_pats


def train_model(train_run_Parameters,file_loc,tables_loaded = True,add_pats = None):
    final_data_train = generate_modelling_data(train_run_Parameters,file_loc,tables_loaded,add_pats)
    log_clf,scaler = run_logistic_baseline(final_data_train)
    return log_clf,scaler


# if __name__ == "__main__":
#     train_run_Parameters = Parameters(end_day = 49,fixed_ending = 49,window_lenght = 5,is_training = True)
#     test_run_Parameters = Parameters(end_day = 50,fixed_ending = 49,is_training=False,window_lenght = 15)
#     return train_model(train_run_Parameters,file_loc,tables_loaded = True,add_pats = None):