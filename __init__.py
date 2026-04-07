"""
Autur: de clankers
Date: 7 april 2026
"""

#from .model import DemoClassifier
import pandas as pd

def data_reader(file_path):
    return pd.read_csv(file_path)

#def model_factory():
#    return DemoClassifier()