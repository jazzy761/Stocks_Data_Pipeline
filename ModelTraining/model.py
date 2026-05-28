import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os


def build_model(input_shape: tuple)-> tf.keras.Model:

    model = Sequential([
        
        Bidirectional(LSTM(128, return_sequences=True) , input_shape = input_shape),
        BatchNormalization(),
        Dropout(0.4),

        Bidirectional(LSTM(32 , return_sequences= True)),
        BatchNormalization(),
        Dropout(0.3),

        LSTM(32, return_sequences =False),
        BatchNormalization(),
        Dropout(0.3),

        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1 , activation="sigmoid")
    ])

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
        loss = "binary_crossentropy",
        metrics = ["accuracy"]
    )

    return model

def train_model(X_train , y_train , X_test , y_test):
    
    model = build_model(input_shape = (X_train.shape[1] , X_train.shape[2]))
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor =  "val_loss",
            patience = 15, 
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor = "val_loss",
            factor = 0.5,
            patience = 7, 
            min_lr = 1e-6
        )
    ]

    history = model.fit(
        X_train , y_train, 
        validation_data = (X_test , y_test),
        epochs = 250,
        batch_size = 16,
        callbacks= callbacks,
        verbose = 1
    )

    return model, history

def save_model(model , path="ModelTraining/lstm_stock.h5"):
    os.makedirs(os.path.dirname(path) , exist_ok=True)
    model.save(path)
    print(f"Model saved to {path}")


if __name__ == "__main__":
    import numpy as np 

    X_train = np.load("ModelTraining/data/X_train.npy")
    X_test = np.load("ModelTraining/data/X_test.npy")
    y_train = np.load("ModelTraining/data/y_train.npy")
    y_test = np.load("ModelTraining/data/y_test.npy")

    print(f"Loaded X_train:{X_train.shape} , X_test:{X_test.shape}")

    model , history = train_model(X_train , y_train , X_test , y_test)
    save_model(model)

    final_train_acc = history.history["accuracy"][-1]
    final_val_acc   = history.history["val_accuracy"][-1]
    print(f"\nFinal train accuracy: {final_train_acc:.4f}")
    print(f"Final val accuracy:   {final_val_acc:.4f}")