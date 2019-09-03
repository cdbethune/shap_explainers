import sys
import os.path
import numpy as np
import pandas as pd
import typing

import shap
import xgboost

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

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

        return None    
    

    def _get_data_sample(self, sample_length):

        '''
            Sub-samples the dataframe to provide a smaller, balanced dataframe to make global explainer
        '''
        
        df = self.X.copy()
        df['pred_target'] = self.model.predict(self.X).astype(int)

        n_classes = df['pred_target'].unique()

        #deal with cases in which the predictions are all one class
        if len(n_classes) ==1:
            return df.sample(1000).drop(columns = ['pred_target'])
        
        else:
            proportion = round(sample_length/len(n_classes))
            
            dfs = []
            full_classes = []
            
            for i in n_classes:
                #dealing with classes that have less than or equal to their proportional representation
                if df[df['pred_target']==i].shape[0] <= proportion:
                    dfs.append(df[df['pred_target']==i])
                else:
                    dfs.append(df[df['pred_target']==i].sample(proportion))
                    
            #try:
             #   sub_sample_df = pd.concat(dfs)
            #except(ValueError):
             #   sub_sample_df = pd.DataFrame(columns = self.X.columns)
        
            # deal with splitting classes that have more than their proportional representation    
           # if len(full_classes) > 0:
                    #reset df
            #    dfs = [sub_sample_df]
                    
             #   remaining = sample_length - sub_sample_df.shape[0]
              #  remaining_prop = round(remaining/len(full_classes))

                        
               # for i in full_classes:
                #    dfs.append(df[df['pred_target']==i].sample(remaining_prop))
                   
            sub_sample_df = pd.concat(dfs)
            
            return sub_sample_df.drop(columns = ['pred_target'])
        

    
    def produce_sample(self, samples):

        '''
            Regression: Returns the shapley values for the given samples
            Classification: Random Forest - returns a list of the explanations for the prediction, in the order that the samples are given
        '''
        
        ##restrict features to most important
        
        shap_values = self.explainer.shap_values(self.X.iloc[samples])
        
       
        if (self.task_type == 'classification') & (self.model_type == 'Random_Forest'):

            values_list = []
            for idx, val in enumerate(samples):
                a = list(self.model.predict_proba(self.X.iloc[[val]]))
                pred_value = a.index(max(a))
                values_list.append(shap_values[pred_value][idx])
                                   
            return values_list
        
        else: 
            return shap_values
    
    
    def produce_global(self):
        """
            Returns global explanation (shapley values)
        """ 

        if (self.model_type == 'Random_Forest') & (self.X.shape[0] > 1500):
            
            sub_sample = self._get_data_sample(sample_length = 1000)
            shap_values = self.explainer.shap_values(sub_sample)
        

        else:            
            shap_values = self.explainer.shap_values(X_test)

        if self.task_type == 'classification':
            return shap_values[1]
        else: 
            return shap_values

if __name__ == '__main__':

    train_path = 'file:///home/alexmably/datasets/seed_datasets_current/26_radon_seed/TRAIN/dataset_TRAIN/tables/learningData.csv'
    test_path = 'file:///home/alexmably/datasets/seed_datasets_current/26_radon_seed/SCORE/dataset_TEST/tables/learningData.csv'
    target = 'log_radon'
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train.dropna(inplace=True)
    
    train = train.drop(columns = ['state','state2','basement','windoor','county'])
    test = test.drop(columns = ['state','state2','basement','windoor','county'])
    X_train = train.drop(columns = ['d3mIndex',target])
    y_train = train[target]

    X_test = test.drop(columns = ['d3mIndex',target])
    y_test = test[target]

    #rf = RandomForestClassifier(n_estimators=50, max_depth=5)
    rf = RandomForestRegressor(n_estimators=50, max_depth=5)
    rf.fit(X_train, y_train)

    exp = Tree(model = rf, X = X_test, model_type = 'Random_Forest', task_type = 'regression')
    #print(exp.produce_global())
    #print(X_test.shape[0])
    print(exp.produce_sample(samples = [10, 44]))
    #print(X_test.iloc[[10,44]])
