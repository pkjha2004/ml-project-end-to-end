import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomEXception
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class model_trainer_config:
    trained_model_file_path = os.path.join("artifacts" , "model.pkl")

class model_trainer:
    def __init__(self):
        self.model_trainer_config  = model_trainer_config()
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("split training and testing input data")
            x_train,y_train,x_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
                

            )
            models = {
                "Random_forest":RandomForestRegressor(),
                "Decision tree":DecisionTreeRegressor(),
                "Gradient_boosting":GradientBoostingRegressor(),
                "Linear_Regression":LinearRegression(),
                "k-neighbour regression":KNeighborsRegressor(),
                "XGB Classifier":XGBRegressor(),
                
                "Adaboost regression":AdaBoostRegressor(),
            }
            modelreport:dict = evaluate_models(x_train= x_train,y_train = y_train,x_test = x_test , y_test = y_test,models = models)

            best_model_score = max(sorted(modelreport.values()))

            best_model_name = list(modelreport.keys())[
                list(modelreport.values()).index(best_model_score)

            ]
            best_model = models[best_model_name]
            if best_model_score< 0.6:
                raise CustomEXception("No best model found")
            logging.info("Best model found on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test,predicted)
            return r2_square




        except Exception as e:
            raise CustomEXception(e,sys)




