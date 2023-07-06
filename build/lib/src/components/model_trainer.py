import os , sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_model , save_obj
from dataclasses import dataclass
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


@dataclass

class ModelTrainerConfig:
    trained_model_file_path = 