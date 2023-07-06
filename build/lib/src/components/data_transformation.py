import os , sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder , StandardScaler
from src.exception import CustomException 
from src.logger import logging
from src.utils import save_obj

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts" , "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Data Transformation initiate")

            scaler_cols = [
                'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2',
       'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
            ]


            scaler_pipeline  = Pipeline(steps=[
                ('scaler' , StandardScaler())
            ])

            preprocessor = ColumnTransformer([
                ('scaler' , scaler_pipeline , scaler_cols)
            ])

            return preprocessor
            logging.info('pipeline is Completed')
        


        except Exception as e:
            logging.info('Exception occured in get_data_transformation_object')
            raise CustomException(e , sys)
        
    def initiate_data_transformation(self , train_path , test_path):
        try:
            ## Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read test and train file is Completed')

            preprocessing_obj = self.get_data_transformation_object()

            target_col_name = ['default payment next month']

            ####### TRANING DATAFRAME

            input_feature_train_df = train_df.drop(target_col_name , axis=1)
            target_feature_train_df = train_df[target_col_name]

            ###### TESTING DATAFRAME

            input_feaature_test_df = test_df.drop(target_col_name , axis=1)
            target_feature_test_df = test_df[target_col_name]

            ## TRANSFORM TRAIN AND TEST DATAFRAME

            transform_input_train_feature = preprocessing_obj.fit_transform(input_feature_train_df)
            transform_input_test_feature = preprocessing_obj.transform(input_feaature_test_df)

            train_arr = np.c_[transform_input_train_feature , np.array(target_feature_train_df)]
            test_arr = np.c_[transform_input_test_feature , np.array(target_feature_test_df)]

            save_obj(
                self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )


            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
                preprocessing_obj
            )
        
        except Exception as e:
            logging.info("Exception occured in initiate_data_transformation")
            raise CustomException(e , sys)