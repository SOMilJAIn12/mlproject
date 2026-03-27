import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
import os

class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.Transformation_Config=DataTransformationConfig()
    def get_data_transformation_object(self):
        try:
            Numerical_Columns = ['reading score', 'writing score']
            Categorical_Columns = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
            num_pipeline=Pipeline(
                steps=[
                ("Imputer",SimpleImputer(strategy="median")),
                ("Scaler",StandardScaler())])
            cat_pipeline=Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder(handle_unknown="ignore")),
                    ("Scaler",StandardScaler(with_mean=False))
                ])
            logging.info(f"Categorical column : {Categorical_Columns}")
            logging.info(f"Numerical column : {Numerical_Columns}")
            preprocessor=ColumnTransformer(
                [("num_pipeline",num_pipeline,Numerical_Columns),
                 ("cat_pipeline",cat_pipeline,Categorical_Columns)]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Read Train and Test Completed")
            logging.info("Obtaining preprocessor object")
            preprocessing_obj=self.get_data_transformation_object()
            target_Column_name='math score'
            numeric_columns=['writing score','reading score']
            input_feature_train_df=train_df.drop(columns=[target_Column_name],axis=1)
            target_feature_train_df=train_df[target_Column_name]

            input_feature_test_df=test_df.drop(columns=[target_Column_name],axis=1)
            target_feature_test_df=test_df[target_Column_name]

            logging.info(
                "Applying the preprocessing object on training and testing dataframes "
            )
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]


            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            logging.info("Saved preprocessing objects")
            save_object(
                file_path=self.Transformation_Config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,

            )
        except Exception as e:
            raise CustomException(e,sys)