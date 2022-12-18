#imports
from statmodels.random_forest.abstract.base_randomforest import RandomForest
from statmodels.decisiontrees import DecisionTreeRegressor
import numpy as np
import pandas as pd
#class for random forest regressor
class RandomForestRegressor(RandomForest):
    #initializer
    def __init__(self,n_trees=100,max_depth=None,min_samples_split=2,loss='mse'):
        super().__init__(n_trees)
        self.max_depth             = max_depth
        self.min_samples_split     = min_samples_split
        self.loss                  = loss
        
    #protected function to obtain the right decision tree
    def _make_tree_model(self,features):
        return(DecisionTreeRegressor(self.max_depth,self.min_samples_split,self.loss))
    
    #public function to return model parameters
    def get_params(self, deep = False):
        return {'n_trees':self.n_trees,
                'max_depth':self.max_depth,
                'min_samples_split':self.min_samples_split,
                'loss':self.loss}
    
    #train the ensemble
    def fit(self,X_train,y_train):
        #call the protected training method
        dcOob = self._train(X_train,y_train)
            
    #predict from the ensemble
    def predict(self,X):
        #call the protected prediction method
        ypred = self._predict(X)
        #return the results
        return(ypred)  