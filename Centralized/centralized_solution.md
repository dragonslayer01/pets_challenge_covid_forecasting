# Table of contents
* [Introduction](#introduction)
* [Data Prep](#data_prep)
* [Data Flow](#data_flow)



# Pandemic Federated learning

## Introduction <a name="introduction"></a>
We have developed a COVID forecasting model based on contact and mobility information. To develop the model we have created 3 categories of features:

* Interaction based - Indicating number of infected people the person has contacted on 
* Location based - Infected people at the locations where a person has visited
* Family based - Infected people n the person's family

## Data Prep <a name = "data_prep"></a>

To triger feature generation , we first have to instantiate `Parameter` separately for both training and testing.

```python
# For Training
train_run_Parameters = Parameters(end_day = 49,fixed_ending = 49,window_lenght = 2,is_training = True)

# For Testing
test_run_Parameters = Parameters(end_day = 50,fixed_ending = 49,is_training=False,window_lenght = 2)
```

Next the `Parameter` object is passed into `generate_modelling_data` to generate the person-level features for model training

## Data Flow <a name = 'data_flow'></a>

After `generate_modelling_data` function is triggered, the following functions are triggered sequentially

* `covid_pats_timeline` - Creates modelling data for covid timeline prediction, i.e days from I state to R state for symptomatic patients
* `interaction_feats` - Function to develop interaction based features
* `loca_feats` - Function to develop location based features
*  `family_feats` - Function to develop family based features
* `final_master_data` - Generates final modelling data

