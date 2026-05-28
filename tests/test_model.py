import numpy as np 
import os 

def test_sequences_exist():
    """Confirms the feature files were generated"""
    assert os.path.exists("ModelTraining/data/X_train.npy"), "X_train.npy missing"
    assert os.path.exists("ModelTraining/data/X_test.npy"),  "X_test.npy missing"
    assert os.path.exists("ModelTraining"), "y_train.npy missing"
    assert os.path.exists("ModelTraining/data/y_test.npy"),  "y_test.npy missing"

def test_sequences_shapes():
    """Confirm shapes are correct"""

    X_train = np.load("ModelTraining/data/X_train.npy")
    X_test  = np.load("ModelTraining/data/X_test.npy")
    y_train = np.load("ModelTraining/data/y_train.npy")
    y_test  = np.load("ModelTraining/data/y_test.npy")

    assert X_train.ndim == 3,          "X_train should be 3D (samples, window, features)"
    assert X_train.shape[1] == 20,     "Window size should be 20"
    assert X_train.shape[2] == 18,     "Should have 18 features"
    assert len(X_train) == len(y_train), "X and y train length mismatch"
    assert len(X_test)  == len(y_test),  "X and y test length mismatch"


def test_label_balance():
    """Confirm labels are not heavilly skewed"""
    y_train = np.load("ModelTraining/data/y_train.npy")
    balance = y_train.mean()
    assert 0.35 < balance < 0.65, f"Label balance {balance:.2f} is too skewed"

def test_model_exists():
    """confirm model was saved"""
    assert os.path.exists("ModelTraining/lstm_stock.h5"), "Model file missing"