#imports
from statmodels.decisiontrees import DecisionTreeRegressor
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.base import clone

## boosting regressor ##
class GradientBoostTreeRegressor(object):
    #initializer
    def __init__(self,  n_elements : int = 100, learning_rate : float = 0.01) -> None:
        self.weak_learner  = DecisionTreeRegressor(max_depth=5)
        self.n_elements    = n_elements
        self.learning_rate = learning_rate
        self.f             = []
        self.residuals     = []
        
    #destructor
    def __del__(self) -> None:
        del self.weak_learner
        del self.n_elements
        del self.learning_rate
        del self.f
        del self.residuals
    
    #public function to return model parameters
    def get_params(self, deep : bool = False) -> Dict:
        return {'weak_learner':self.weak_learner,'n_elements':self.n_elements,'learning_rate':self.learning_rate}
    
    #public function to train the ensemble
    def fit(self, X_train, y_train): #X_train : np.array, y_train : np.array, feature=None) -> None:
        self.features = X_train.columns
        
        #initialize residuals
        r = np.copy(y_train.reshape(-1, 1)).astype(float)
             
        #get the list of features
        self.features_used = pd.DataFrame()
        tree_count =0
        
        #loop through the specified number of iterations in the ensemble
        for _ in range(self.n_elements):
            #make a copy of the weak learner
            model = clone(self.weak_learner)
            #fit the weak learner on the current dataset
            model.fit(X_train.values,r, features=self.features)#,feature)
            #update the residuals
            r -= self.learning_rate*model.predict(X_train.values)
            #append resulting model
            self.f.append(model)
            #-----> additional code: get important features 
            dt_features = pd.DataFrame(model.get_features())
            dt_features.columns = ['featurename','treelevel']
            dt_features["tree"]= tree_count
            self.features_used = pd.concat([self.features_used,dt_features])
            tree_count+=1
            
            #append current mean residual
            self.residuals.append(np.mean(r))    
            
    #public function to return residuals
    def get_residuals(self) -> List:
        return(self.residuals)
    
    #public function to generate predictions
    def predict(self, X_test : np.array) -> np.array:
        #initialize output
        y_pred = np.zeros((X_test.shape[0]))
        #traverse ensemble to generate predictions
        for model in self.f:
            y_pred += self.learning_rate*model.predict(X_test)
        #return predictions
        return(y_pred)
    
    #public function to return decision trees
    def get_decision_trees(self):
        return self.f
    
    #get the list of features
    def get_features(self):
        return self.features_used 