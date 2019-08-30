import sys
import os.path
import numpy as np
import pandas as pd
import typing

import shap

from sklearn.ensemble import RandomForestClassifier

class Tree():
    '''
        Primitive that takes a dataset and a fitted tree-based model and returns the global explanation
        for the model predictions on the given dataset, and a function to produce explanations for any 
        given sample from the explanations.

        Parameters
        ----------
        model : model object
        The tree based machine learning model that we want to explain. XGBoost, LightGBM, CatBoost,
        and most tree-based scikit-learn models are supported.

        X :numpy.array, pandas.DataFrame or catboost.Pool (for catboost)
            A matrix of samples (# samples x # features) on which to explain the model’s output. 
            (catboost not yet supported)

        number_of_features : int, (default = 5)
        Function will return explanations for the top K number of features
        
        model_type : None (default), or 'Random_Forest'
            If 'Random_Forest' then global explanation is calculated on a sub-sample of the full datatset

        task_type : 'regression' (default) or 'classification'
            Is the tree model a regressor or a classifier
    '''

    def __init__(self, model, X, number_of_features = 5, model_type = None, task_type = 'regression')-> None:
  
        self.X = X
        self.model = model
        self.number_of_features = number_of_features
        self.model_type = model_type
        self.task_type = task_type 

        self.explainer = shap.TreeExplainer(self.model)
    
    

    def _get_data_sample(df, target, sample_length):
        '''
            Sub-samples the given dataframe to provide a smaller, balanced dataframe to make global explainer
        '''
        n_classes = df[target].unique()
        proportion = round(sample_length/len(n_classes))
        
        dfs = []
        full_classes = []
        
        for i in n_classes:
            if df[df[target]==i].shape[0] <= proportion:
                dfs.append(df[df[target]==i])
            else:
                full_classes.append(i)
        df0 = pd.concat(dfs)
        
        #reset dfs
        dfs = [df0]
        
        remaining = sample_length - df0.shape[0]
        remaining_prop = round(remaining/len(full_classes))
            
        for i in full_classes:
            dfs.append(df[df[target]==i].sample(remaining_prop))
        
        sub_sample_df = pd.concat(dfs)
        
        return sub_sample_df

    
    def produce_sample(self, samples):
        '''
            Returns the shap values for the given samples
        '''
        ##need the prediction value with it here, then you can give the output for the actual prediction?
        shap_values = self.explainer.shap_values(self.X.iloc[samples])
        
        return shap_values
    
    
    def produce_global(self):
        """
            Returns global explanation (shapley values)
        """ 

        if self.model_type == 'Random_Forest':
            
            df = self.X.copy()
            df['rf_preds'] = self.model.predict(self.X).astype(int)

            sub_sample = _get_data_sample(df=df, target='rf_preds', sample_length = 1000)
            shap_values = self.explainer.shap_values(sub_sample)
        

        else:            
            shap_values = self.explainer.shap_values(X_test)

        return shap_values

if __name__ == '__main__':

    train_path = 'file:///home/alexmably/datasets/seed_datasets_current/SEMI_1217_click_prediction_small/TRAIN/dataset_TRAIN/tables/learningData.csv'
    test_path = 'file:///home/alexmably/datasets/seed_datasets_current/SEMI_1217_click_prediction_small/SCORE/dataset_SCORE/tables/learningData.csv'

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train.dropna(inplace=True)
    X_train = train.drop(columns = ['d3mIndex','click'])
    y_train = train['click']

    X_test = test.drop(columns = ['d3mIndex','click'])
    y_test = test['click']

    rf = RandomForestClassifier(n_estimators=50, max_depth=5)
    rf.fit(X_train, y_train)

    exp = Tree(model = rf, X = X_test)
    print(exp.produce_sample(samples = [20, 21]))
