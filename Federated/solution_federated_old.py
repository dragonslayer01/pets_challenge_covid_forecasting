"""
@ZS_RDE_AI

"""
from typing import Tuple, Union
import warnings
from pathlib import Path 

import flwr as fl
import torch 
import pickle
import pathlib
import time 
from .feature_generation_pipeline import *
from .utils import *
# from .utils_he import *
from .hacky_feature_creation import hacky_feature_banao
from .pandemic_model import train_model,test_model

epochs = 5
lr=0.001
privacy_budget = 0.003 #define between 1 to 10 (1 => very strict budget)
mu = 0.01
max_grad_norm = 10
batch_size = 128
threshold = 3


class TrainClient(fl.client.NumPyClient):
    """Custom Flower NumPyClient class for training."""
    
    def __init__(self, client_dir, X_train, y_train, X_test, y_test, scalar = None):
        print(f"{client_dir} train client instantiated")
        
        self.client_dir = client_dir
        
        #Generating Key for Homomorphic Encryption 
#         pk, sk = KeyGen()
        self.pk = 1
        self.sk = 2
        
        # Create Model
        self.model = LogisticRegression()
        # Setting initial parameters, akin to model.compile for keras models
        set_initial_params(self.model)
        
        #Convert data to tensors
        self.X_train = torch.Tensor(X_train.values)
        self.y_train = torch.Tensor(y_train.values)
        self.X_test = torch.Tensor(X_test.values)
        self.y_test = torch.Tensor(y_test.values)
       
        self.scalar = scalar
        feature_count = self.X_train.shape[1]
        
    
    def get_parameters(self, config):  # type: ignore
        print(f"{self.client_dir} get Parameters called")
        return get_model_parameters(self.model)

    def fit(self, parameters, config):  # type: ignore
        #print("Train Client Fit Fn called")
        print(f"{self.client_dir} Fit called")

#         decoded_parameters = []
#         for param in parameters:
#         decoded_parameters = homomorphic_encryption.decrypt_weights_param(self.sk, parameters, num_clients)
        decoded_parameters= parameters
        self.model = set_model_params(self.model, decoded_parameters)
        
        # Ignore convergence failure due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") 
            self.model = model_fit(self.model, decoded_parameters, mu, self.X_train, self.y_train, 
        privacy_budget, max_grad_norm, lr,batch_size,epochs)
            
        #encryption
        local_model_parameters = get_model_parameters(self.model)
        encrypted_params= local_model_parameters
#         encrypted_params = homomorphic_encryption.encrypt_weights_params(self.pk, local_model_parameters)
#         print(f"Training finished for round {config['server_round']}")
        
        return encrypted_params, len(self.X_train), {}
    

    def evaluate(self, parameters, config):  # type: ignore
        print(f"{self.client_dir} Evaluate called")
        #print("Train Client Evaluate Fn called")
        decoded_parameters = parameters
#         decoded_parameters = homomorphic_encryption.decrypt_weights_param(self.sk, parameters, num_clients)
        self.model = set_model_params(self.model, decoded_parameters)
        
#         #Writing model parameters
#         local_model_path = pathlib.Path(self.client_dir, "model_param.pkl")
#         with open(local_model_path, 'wb') as f:
#             pickle.dump(decoded_parameters, f)
        
        predictions, loss, accuracy, avg_precision = utils_test_model(self.model, self.X_test, self.y_test,threshold)
        print(f"Loss - {loss}, Accuracy - {accuracy}, Average Precision - {avg_precision}")
       
        return (float(loss), len(self.X_test), {"accuracy": avg_precision})



def train_client_factory(
    cid: str=None,
    person_data_path: Path = None,
    household_data_path: Path = None,
    residence_location_data_path: Path = None,
    activity_location_data_path: Path = None,
    activity_location_assignment_data_path: Path = None,
    population_network_data_path: Path = None,
    disease_outcome_data_path: Path = None,
    client_dir: Path = None,
) -> Union[fl.client.Client, fl.client.NumPyClient]:
    """
    Factory function that instantiates and returns a Flower Client for training.
    The federated learning simulation engine will use this function to
    instantiate clients with all necessary dependencies.

    Args:
        cid (str): Identifier for a client node/federation unit. Will be
            constant over the simulation and between train and test stages.
        person_data_path (Path): Path to CSV data file for the Person table, for
            the partition specific to this client.
        household_data_path (Path): Path to CSV data file for the House table,
            for the partition specific to this client.
        residence_location_data_path (Path): Path to CSV data file for the
            Residence Locations table, for the partition specific to this
            client.
        activity_location_data_path (Path): Path to CSV data file for the
            Activity Locations on table, for the partition specific to this
            client.
        activity_location_assignment_data_path (Path): Path to CSV data file
            for the Activity Location Assignments table, for the partition
            specific to this client.
        population_network_data_path (Path): Path to CSV data file for the
            Population Network table, for the partition specific to this client.
        disease_outcome_data_path (Path): Path to CSV data file for the Disease
            Outcome table, for the partition specific to this client.
        client_dir (Path): Path to a directory specific to this client that is
            available over the simulation. Clients can use this directory for
            saving and reloading client state.

    Returns:
        (Union[Client, NumPyClient]): Instance of Flower Client or NumPyClient.
    """
    print("Train Client Factory Fn called for ", cid)
    dict_of_loc = {
        "person": person_data_path,
        "household":household_data_path,
        "residence_location":residence_location_data_path,
        "activity_location":activity_location_data_path,
        "activity_location_assignment":activity_location_assignment_data_path,
        "population_network":population_network_data_path,
        "disease_outcome_training":disease_outcome_data_path
    }
    
#     train_run_Parameters = Parameters(end_day = 49,fixed_ending = 49,window_lenght = 2,is_training = True)
#     #Generating Modelling Data 
#     final_data_train = generate_modelling_data(train_run_Parameters, dict_of_loc, client_dir, 
#                                                db_name = f'my-{cid}-db.duckdb',tables_loaded = False, add_pats = None)
    final_data_train = hacky_feature_banao(dict_of_loc)
   
    X = final_data_train.drop(['pid','covid'],axis = 1).fillna(0)
    y = final_data_train['covid']

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    X_std = pd.DataFrame(data = X_std,columns = X.columns)

    # Split into training and test
    X_train, X_test, y_train, y_test = train_test_split(X_std, y,
                                                        test_size=.1,
                                                        random_state=0,
                                                        stratify=y
                                                       )
   
    print(f'{cid} has {set(y_test)}')
    
    ##Writing scaler model to Client Directory
    
    
    return TrainClient(client_dir,X_train, y_train, X_test, y_test, scalar = scaler)


class TrainStrategy(fl.server.strategy.Strategy):
    """Custom Flower strategy for training."""
    def __init__(self, server_dir):
        print(f"{server_dir} Train strategy class instantiated")
        self.strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn = fed_metrics_weighted_average)
        self.server_dir = server_dir
    def initialize_parameters(self, client_manager):
        print(f"{self.server_dir} Initialize parameter called")
        return self.strategy.initialize_parameters(client_manager)
    def configure_fit(self, server_round, parameters, client_manager):
        print(f"{self.server_dir} Config fit called")
        return self.strategy.configure_fit(server_round, parameters, client_manager)
    def aggregate_fit(self,server_round,results,failures):
        print(f"{self.server_dir} Aggreate fit called")
        agg_params, agg_metrics = self.strategy.aggregate_fit(server_round, results, failures)
        
        #storing aggregated params in server directory
        #Writing model parameters
        params_server_dir_path = pathlib.Path(self.server_dir, "model_param.pkl")
        metrics_server_dir_path = pathlib.Path(self.server_dir, "model_metrics.pkl")
        with open(params_server_dir_path, 'wb') as f:
            pickle.dump(agg_params, f)
            
        with open(metrics_server_dir_path, 'wb') as f:
            pickle.dump(agg_metrics, f)
            
        return agg_params, agg_metrics 
    def configure_evaluate(self, server_round, parameters, client_manager):
        print(f"{self.server_dir} Configure evaluate called")
        return self.strategy.configure_evaluate(server_round, parameters, client_manager)
    def aggregate_evaluate(self,server_round,results,failures):
        print(f"{self.server_dir} aggregate evaluate called")
        return self.strategy.aggregate_evaluate(server_round, results, failures)
    def evaluate(self, server_round, parameters):
        print(f"{self.server_dir} evaluate called")
        return self.strategy.evaluate(server_round, parameters)
    

def train_strategy_factory(
    server_dir: Path,
) -> Tuple[fl.server.strategy.Strategy, int]:
    """
    Factory function that instantiates and returns a Flower Strategy, plus the
    number of federated learning rounds to run.

    Args:
        server_dir (Path): Path to a directory specific to the server/aggregator
            that is available over the simulation. The server can use this
            directory for saving and reloading server state. Using this
            directory is required for the trained model to be persisted between
            training and test stages.

    Returns:
        (Strategy): Instance of Flower Strategy.
        (int): Number of federated learning rounds to execute.
    """
    print("Train Strategy Factory Fn called", server_dir)
    return TrainStrategy(server_dir),1


class TestClient(fl.client.NumPyClient):
    """Custom Flower NumPyClient class for test."""
    def __init__(self, client_dir, preds_format_path, preds_dest_path, data, X_test, y_test):
        
        self.client_dir = client_dir
        
        #Generating Key for Homomorphic Encryption 
#         pk, sk = KeyGen()
        self.pk = 1
        self.sk = 2
        
        # Create Model
        self.model = LogisticRegression()
        print("Model initiated", self.model)
        
        # Setting initial parameters, akin to model.compile for keras models
        set_initial_params(self.model)
    
        self.X_test = torch.Tensor(X_test.values)
        self.y_test = torch.Tensor(y_test.values)
        
        self.final_op = data[["pid", "covid"]]
        self.threshold = 0.5
        self.preds_format_path = preds_format_path
        self.preds_dest_path = preds_dest_path
      
        feature_count = self.X_test.shape[1]
        
    
    def get_parameters(self, config):  # type: ignore
            return get_model_parameters(self.model)

    def fit(self, parameters, config):  # type: ignore
        print("Test Client Fit Fn called")
        return get_model_parameters(self.model), len(self.X_test), {}

    def evaluate(self, parameters, config):  # type: ignore
        print("Test Client Evaluate Fn called")
   
        self.model = set_model_params(self.model, parameters)
        predictions, loss, accuracy, avg_precision = utils_test_model(self.model, self.X_test, self.y_test, self.threshold)
        print(f"Loss - {loss}, Accuracy - {accuracy}, Average Precision - {avg_precision}")
        
        self.final_op["score"] = predictions
        self.final_op.drop(columns=["covid"], axis = 1, inplace = True)
        print("OP Head:", self.final_op.head(3))
#         destination_path = pathlib.Path(self.client_dir, self.preds_dest_path)
        self.final_op.to_csv(self.preds_dest_path, index = False)
#         final_op_format = pd.read_csv(self.preds_format_path)
#         final_op_format = pred_format.merge(model_final_results,how = 'left', on = 'pid')[['pid','score_model']]
        

        return float(loss), len(self.X_test), {"accuracy": avg_precision}


def test_client_factory(
    cid: str = None,
    person_data_path: Path= None,
    household_data_path: Path= None,
    residence_location_data_path: Path= None,
    activity_location_data_path: Path= None,
    activity_location_assignment_data_path: Path= None,
    population_network_data_path: Path= None,
    disease_outcome_data_path: Path= None,
    client_dir: Path= None,
    preds_format_path: Path= None,
    preds_dest_path: Path= None,
) -> Union[fl.client.Client, fl.client.NumPyClient]:
    """
    Factory function that instantiates and returns a Flower Client for test-time
    inference. The federated learning simulation engine will use this function
    to instantiate clients with all necessary dependencies.

    Args:
        cid (str): Identifier for a client node/federation unit. Will be
            constant over the simulation and between train and test stages.
        person_data_path (Path): Path to CSV data file for the Person table, for
            the partition specific to this client.
        household_data_path (Path): Path to CSV data file for the House table,
            for the partition specific to this client.
        residence_location_data_path (Path): Path to CSV data file for the
            Residence Locations table, for the partition specific to this
            client.
        activity_location_data_path (Path): Path to CSV data file for the
            Activity Locations on table, for the partition specific to this
            client.
        activity_location_assignment_data_path (Path): Path to CSV data file
            for the Activity Location Assignments table, for the partition
            specific to this client.
        population_network_data_path (Path): Path to CSV data file for the
            Population Network table, for the partition specific to this client.
        disease_outcome_data_path (Path): Path to CSV data file for the Disease
            Outcome table, for the partition specific to this client.
        client_dir (Path): Path to a directory specific to this client that is
            available over the simulation. Clients can use this directory for
            saving and reloading client state.
        preds_format_path (Path): Path to CSV file matching the format you must
            write your predictions with, filled with dummy values.
        preds_dest_path (Path): Destination path that you must write your test
            predictions to as a CSV file.

    Returns:
        (Union[Client, NumPyClient]): Instance of Flower Client or NumPyClient.
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
    
#     test_run_Parameters = Parameters(end_day = 49,fixed_ending = 49,window_lenght = 2,is_training = True)
#     #Generating Modelling Data 
#     final_data_test = generate_modelling_data(test_run_Parameters, dict_of_loc, client_dir, 
#                                                db_name = f'my-{cid}-db.duckdb',tables_loaded = False, add_pats = None)
    final_data_test = hacky_feature_banao(dict_of_loc)
   
   
    X = final_data_test.drop(['pid','covid'],axis = 1).fillna(0)
    y = final_data_test['covid']

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    X_std = pd.DataFrame(data = X_std,columns = X.columns)
    
    return TestClient(client_dir, preds_format_path, preds_dest_path, final_data_test, X_std, y)


class TestStrategy(fl.server.strategy.Strategy):
    """Custom Flower strategy for test."""
    def __init__(self, server_dir):
        self.strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn = fed_metrics_weighted_average)
        self.server_dir = server_dir
        
    def initialize_parameters(self, client_manager):
        print("CLIENT MANAGER CHECK NUM_AVLB::::", client_manager.num_available())
        local_model_path = pathlib.Path(self.server_dir, "model_param.pkl")
        with open(local_model_path, 'rb') as f:
            parameters = pickle.load(f)
            print("Model Param Read")
           
        initial_parameters = parameters
        self.strategy.initial_parameters = None
        
        return initial_parameters
    
    def configure_fit(self, server_round, parameters, client_manager):
        return self.strategy.configure_fit(server_round, parameters, client_manager)
    
    def aggregate_fit(self,server_round,results,failures):
        return self.strategy.aggregate_fit(server_round, results, failures)
    def configure_evaluate(self, server_round, parameters, client_manager):
        return self.strategy.configure_evaluate(server_round, parameters, client_manager)
    def aggregate_evaluate(self,server_round,results,failures):
        return self.strategy.aggregate_evaluate(server_round, results, failures)
    def evaluate(self, server_round, parameters):
        return self.strategy.evaluate(server_round, parameters)


def test_strategy_factory(
    server_dir: Path,
) -> Tuple[fl.server.strategy.Strategy, int]:
    """
    Factory function that instantiates and returns a Flower Strategy, plus the
    number of federation rounds to run.

    Args:
        server_dir (Path): Path to a directory specific to the server/aggregator
            that is available over the simulation. The server can use this
            directory for saving and reloading server state. Using this
            directory is required for the trained model to be persisted between
            training and test stages.

    Returns:
        (Strategy): Instance of Flower Strategy.
        (int): Number of federated learning rounds to execute.
    """
    ...
    return TestStrategy(server_dir), 1
