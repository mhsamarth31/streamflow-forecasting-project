from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


def build_ann(input_size: int, hidden_size: int = 12, learning_rate: float = 0.009):
    model = Sequential([
        Dense(hidden_size, activation="sigmoid", input_shape=(input_size,)),
        Dense(1, activation=None),
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
    )

    return model
