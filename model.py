"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json

#bruh = pd.read_csv('C:/Users/RGRCH/Desktop/API/FM1_Predict_API/utils/data/df_test.csv')
#bruh = df.drop(df.columns[0], axis=1)

#print(bruh.info())

#pred_vec = bruh[['Madrid_wind_speed','Bilbao_rain_1h','Valencia_wind_speed']]

#print(pred_vec.info())

#print(bruh)

#model_load_path = "C:/Users/RGRCH/Desktop/API/FM1_Predict_API/assets/trained-models/rfr_model.pkl"
#with open(model_load_path,'rb') as file:
#    unpickled_model = pickle.load(file)

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.
    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    #print(feature_vector_df)

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    
    #predict_vector = feature_vector_df[['Madrid_wind_speed','Bilbao_rain_1h','Valencia_wind_speed']]

    predict_vector = feature_vector_df.copy()
    
    predict_vector = predict_vector.drop(predict_vector.columns[0], axis=1)
    
    if len(predict_vector) == 1:
        predict_vector["Valencia_pressure"] = predict_vector["Valencia_pressure"].fillna(float(1000)) 
    else:
        predict_vector["Valencia_pressure"] = predict_vector["Valencia_pressure"].fillna(predict_vector["Valencia_pressure"].mode()[0])
     
    predict_vector = predict_vector.loc[:, ~predict_vector.columns.str.contains("temp_min")]
    predict_vector = predict_vector.loc[:, ~predict_vector.columns.str.contains("temp_max")]
    predict_vector["Valencia_wind_deg"] = predict_vector["Valencia_wind_deg"].str.extract('(\d+)')
    predict_vector["Valencia_wind_deg"] = pd.to_numeric(predict_vector["Valencia_wind_deg"])
    predict_vector["Seville_pressure"] = predict_vector["Seville_pressure"].str.extract('(\d+)')
    predict_vector["Seville_pressure"] = pd.to_numeric(predict_vector["Seville_pressure"])

    predict_vector["time"] = pd.to_datetime(predict_vector["time"])
    predict_vector["Hour"] = predict_vector["time"].dt.hour
    predict_vector["Day"] = predict_vector["time"].dt.day
    predict_vector["Weekday"] = predict_vector["time"].dt.weekday
    predict_vector["Week"] = predict_vector["time"].dt.isocalendar().week
    predict_vector['Week'] = predict_vector['Week'].astype('int64')
    predict_vector["Month"] = predict_vector["time"].dt.month
    predict_vector["Year"] = predict_vector["time"].dt.year
    predict_vector = predict_vector.drop(['time'], axis=1)

    seasons = {1: 'Winter',
               2: 'Winter',
               3: 'Spring',
               4: 'Spring',
               5: 'Spring',
               6: 'Summer',
               7: 'Summer',
               8: 'Summer',
               9: 'Autumn',
               10: 'Autumn',
               11: 'Autumn',
               12: 'Winter',
               }
    
    if len(predict_vector) == 1:
        predict_vector['Season_Winter'] = 0
        predict_vector['Season_Spring'] = 0
        predict_vector['Season_Summer'] = 0
        predict_vector['Season_Autumn'] = 0

        if predict_vector['Month'].iloc[0] == (12 or 1 or 2):
            predict_vector['Season_Winter'] = 1
        elif predict_vector['Month'].iloc[0] == (3 or 4 or 5):
            predict_vector['Season_Spring'] = 1
        elif predict_vector['Month'].iloc[0] == (6 or 7 or 8):
            predict_vector['Season_Summer'] = 1
        elif predict_vector['Month'].iloc[0] == (9 or 10 or 11):
            predict_vector['Season_Autumn'] = 1

    else:
        predict_vector['Season'] = predict_vector['Month'].apply(lambda x: seasons[x])
        predict_vector = pd.get_dummies(predict_vector, columns=['Season'])
        predict_vector.columns = [col.replace(" ","_") for col in predict_vector.columns]

    
    predict_vector['Season_Winter'] = predict_vector['Season_Winter'].astype('int64')
    predict_vector['Season_Spring'] = predict_vector['Season_Spring'].astype('int64')
    predict_vector['Season_Summer'] = predict_vector['Season_Summer'].astype('int64')
    predict_vector['Season_Autumn'] = predict_vector['Season_Autumn'].astype('int64')

    # ------------------------------------------------------------------------

    return predict_vector

#test = pd.read_csv('C:/Users/RGRCH/Desktop/API/FM1_Predict_API/utils/data/df_test.csv')

#feature_vec_json = test.iloc[1].to_json()
#feature_vec_dict = json.loads(feature_vec_json)
#feature_vec_df = pd.DataFrame.from_dict([feature_vec_dict])

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))


""" You may use this section (above the make_prediction function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""

def make_prediction(data, model):
    """Prepare request data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standardisation.
    return prediction[0].tolist()
