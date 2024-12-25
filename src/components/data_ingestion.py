import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artificat',"train.csv")
    test_data_path: str=os.path.join('artificat',"test.csv")
    raw_data_path: str=os.path.join('artificat',"data.csv")

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
    data_ingestion.initiate_data_ingestion()