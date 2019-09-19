import sys
import os.path
import numpy as np
import pandas as pd
import typing

import shap
import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans

class Tree():
    '''
        Primitive that takes a dataset and a fitted tree-based model and returns explnations. 
        For global explanations returns a tuple containing the datatset that explanations were made on, and the global explanations
        For sample explnations, returns a list of the explantions for the samples provided
    
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

    def __init__(self, model, X, number_of_features = None, model_type = None, task_type = 'regression')-> None:
  
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
        df['cluster_assignment'] = KMeans().fit_predict(self.X).astype(int)

        n_classes = df['cluster_assignment'].unique()

        #deal with cases in which the predictions are all one class
        if len(n_classes) ==1:
            return df.sample(sample_length).drop(columns = ['cluster_assignment'])
        
        else:
            proportion = round(sample_length/len(n_classes))
            
            dfs = []
            
            for i in n_classes:
                #dealing with classes that have less than or equal to their proportional representation
                if df[df['cluster_assignment']==i].shape[0] <= proportion:
                    dfs.append(df[df['cluster_assignment']==i])
                else:
                    dfs.append(df[df['cluster_assignment']==i].sample(proportion))

            sub_sample_df = pd.concat(dfs)
            
            return sub_sample_df.drop(columns = ['cluster_assignment'])

    def _get_top_features(self, shap_values, number_of_features):
        
        df = pd.DataFrame(shap_values, columns= self.X.columns)
        cols = abs(df.mean()).sort_values(ascending=False)[:number_of_features].index
        sorter = np.argsort(self.X.columns)
        imp_features = sorter[np.searchsorted(self.X.columns, cols, sorter = sorter)]

        keep = []
        for row in shap_values:
            keep.append(row[imp_features])
        keep_list = [k.tolist() for k in keep]

        return imp_features, keep_list

        

    
    def produce_sample(self, samples):

        '''
            Returns a dataframe of the shapley values for the given samples
            
        '''
        
        ##restrict features to most important
        
        shap_values = self.explainer.shap_values(self.X.iloc[samples])
        
       
        if (self.task_type == 'classification') & (self.model_type == 'Random_Forest'):
            
            probs = self.explainer.expected_value
            idx = probs.index(max(probs))
                                
            return pd.DataFrame(shap_values[idx], columns = self.X.columns, index = self.X.iloc[samples].index)
        
        else: 
            return pd.DataFrame(shap_values, columns = self.X.columns, index = self.X.iloc[samples].index)
    
    
    def produce_global(self):
        """
            Returns a dataframe of the shap values for each feature, this will be a downsampled dataframe for large datasets 
        """ 
        
        if (self.model_type == 'Random_Forest') & (self.X.shape[0] > 1500):
            df = self._get_data_sample(sample_length = 1500)
            shap_values = self.explainer.shap_values(df)
            df.sort_index(inplace = True)

        else:
            df = self.X
            shap_values = self.explainer.shap_values(df)


        if (self.task_type == 'classification') & (self.model_type == 'Random_Forest'):

            probs = self.explainer.expected_value
            idx = probs.index(max(probs))
                
            shap_values = shap_values[idx]
        
       #if self.number_of_features:
         #   features, shap_values = self._get_top_features(shap_values, self.number_of_features)
          #  return (sub_sample.iloc[:,features], shap_values)    
        
        return (pd.DataFrame(shap_values, columns = df.columns, index = df.index))

if __name__ == '__main__':

    train_path = 'file:///home/alexmably/datasets/seed_datasets_current/SEMI_1217_click_prediction_small/TRAIN/dataset_TRAIN/tables/learningData.csv'
    test_path = 'file:///home/alexmably/datasets/seed_datasets_current/SEMI_1217_click_prediction_small/SCORE/dataset_SCORE/tables/learningData.csv'
    target = 'click'
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train.dropna(inplace=True)
    
    #train = train.drop(columns = ['state','state2','basement','windoor','county'])
    #test = test.drop(columns = ['state','state2','basement','windoor','county'])
    
    ##RF 
    X_train = train.drop(columns = ['d3mIndex',target])
    y_train = train[target]
    X_test = test.drop(columns = ['d3mIndex',target])

    #rf = RandomForestClassifier(n_estimators=50, max_depth=5)
    #rf = RandomForestRegressor(n_estimators=50, max_depth=5)
    #rf.fit(X_train, y_train)
 

    ##XGBOOST
    
    xgtrain = xgb.DMatrix(X_train.values, y_train.values)
    xgtest = xgb.DMatrix(X_test.values)
    param = {'max_depth':2, 'eta':1}
    #xgb = xgboost.XGBRegressor().fit(X_train, y_train)
    model = xgb.train(params = param, dtrain = xgtrain)
    #model = xgb.fit(X_train,y_train)

    exp = Tree(model = model, X = X_test, task_type = 'classification')
    print(exp.produce_global())
    print(exp.produce_sample([12]))
    #print(X_test.shape[0])
    #print(exp.produce_sample(samples = [10, 44]))
    #shap.summary_plot(exp.produce_global()[1][0], exp.produce_global()[0])
    #print(X_test.iloc[[10,44]])
    #print(shap.kmeans(X_test,10))



    #datasets:
    #SEMI_1217_click_prediction_small
    #SEMI_1040_sylva_prior
    #SEMI_1044_eye_movements
    #196_autoMpg
    #26_radon_seed