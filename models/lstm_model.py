from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam


def build_lstm(
    sequence_length: int = 5,
    feature_count: int = 2,
    hidden_size: int = 12,
    learning_rate: float = 0.001,
):
    model = Sequential([
        LSTM(hidden_size, input_shape=(sequence_length, feature_count)),
        Dense(1, activation=None),
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
    )

    return model
