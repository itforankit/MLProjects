import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingesion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion initiated")
        try:
            df=pd.read_csv('notebook/data/stud.csv') # read data from raw data path
            logging.info("Data read successfully in dataframe df")

            os.makedirs(os.path.dirname(self.ingesion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingesion_config.raw_data_path, index=False,header=True) # save data to train data path
            logging.info("Train/Test Split Initiated")
            
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            
            train_set.to_csv(self.ingesion_config.train_data_path, index=False,header=True) # save data to train data path
            test_set.to_csv(self.ingesion_config.test_data_path, index=False,header=True) # save data to test data path

            logging.info("Data saved successfully to Train/Test Split")
            return(
                self.ingesion_config.train_data_path,
                self.ingesion_config.test_data_path
                #self.ingesion_config.raw_data_path
            ) #train_set,test_set
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data,test_data=data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    model_trainer = ModelTrainer()
    #model_trainer.initiate_model_trainer(train_arr,test_arr)
    print("Model Training Completed")
    print(model_trainer.initiate_model_trainer(train_arr,test_arr))
