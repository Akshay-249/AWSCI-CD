import os
import sys

from dataclasses import dataclass

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from sklearn.ensemble import (AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class MOdelTrainerConfig:
    train_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = MOdelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("spliting train and test input data")

            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "cat boosting" : CatBoostRegressor(verbose = False),
                "linear Regression" : LinearRegression(),
                "Random Forest" : RandomForestRegressor(),
                "Decision tree" : DecisionTreeRegressor(),
                "gradient bossting" : GradientBoostingRegressor(),
                "k-Neighbours regressor" : KNeighborsRegressor(),
                "AdaBossting regressor" : AdaBoostRegressor(),
                
            }

            model_report:dict = evaluate_models(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, models = models)
            
            #To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            #To get best model_name
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            save_object(
                file_path = self.model_trainer_config.train_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square, best_model_name, model_report
        
        except Exception as e:
            raise CustomException(e, sys)