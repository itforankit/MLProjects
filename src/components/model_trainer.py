import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evalutae_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            logging.info("Data split successfully")
            models = {
                "Random Forest" : RandomForestRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "AdaBoost" : AdaBoostRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "Linear Regression" : LinearRegression(),
                "K-Nearest Neighbors" : KNeighborsRegressor(),
                "XGBClasifier" : XGBRegressor(),
                "catboost classifier" : CatBoostRegressor(verbose=False),
            }

            model_report:dict=evalutae_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))   
            ## to get the best model name from dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]    
            logging.info(f"Best model name: {best_model_name}")
            logging.info(f"Best model score: {best_model_score}")   \
            
            best_model=models[best_model_name]

            if best_model_score <0.6:
                raise CustomException("No best Model found")
            logging.info("Best model found successfully on train and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted=best_model.predict(X_test)
            r2_score_value=r2_score(y_test,predicted)
            logging.info(f"R2 score value: {r2_score_value}")
            return r2_score_value

        except Exception as e:
            logging.error(f"Error in initiating model trainer: {e}")
            raise CustomException("Error in initiating model trainer")


