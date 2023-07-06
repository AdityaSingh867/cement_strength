import os , sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_model , save_obj
from dataclasses import dataclass
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB


@dataclass

class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts' , 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self , train_array , test_array):
        try:

            logging.info("Splitting dependent and independent variables from train and test data")
            X_train , y_train , X_test , y_test = (
                train_array[: , :-1],
                train_array[: , -1],
                test_array[: , :-1],
                test_array[: , -1]
            )

            models = {
                'xgb_classifire' : XGBClassifier(),
                'gaussian_nb' : GaussianNB()
            }

            model_report : dict = evaluate_model(X_train , y_train , X_test , y_test , models)
            print(model_report)
            print("\n=====================================================\n")
            logging.info(f'model report : {model_report}')

            ## TO GET THE BEST MODEL SCORE FROM DICTIONARY

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            

            best_model = models[best_model_name]

            print(f'best model found : {best_model_name} , best model score : {best_model_score}')
            print("\n==========================================================================\n")
            logging.info(f"best model found : {best_model_name} , best model score : {best_model_score}")

            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )



        except Exception as e:
            logging.info("Exception occured in initiate_model_trainer")
            raise CustomException(e , sys)