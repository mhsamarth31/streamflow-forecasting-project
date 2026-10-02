import numpy as np
from sklearn.metrics import mean_squared_error

from models.ann_model import build_ann


def train_ann(
    X_train,
    y_train,
    X_valid,
    y_valid,
    epochs=200,
    batch_size=32,
    hidden_size=12,
    learning_rate=0.009,
):
    model = build_ann(
        input_size=X_train.shape[1],
        hidden_size=hidden_size,
        learning_rate=learning_rate,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_valid, y_valid),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    predictions = model.predict(X_valid, verbose=0).ravel()
    rmse = np.sqrt(mean_squared_error(y_valid, predictions))

    return model, history, predictions, rmse
