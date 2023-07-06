import os , sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:
    def __init__(self)-> None:
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()




    def run_pipeline(self):
        try:

            train_path , test_path = self.data_ingestion.initiate_data_ingestion()
            train_arr , test_arr , _ , _ = self.data_transformation.initiate_data_transformation(train_path , test_path)
            self.model_trainer.initiate_model_trainer(train_arr , test_arr)

        except Exception as e:
            logging.info("Exception occured in run_pipeline")
            raise CustomException(e , sys)