import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    GRU,
    Dense,
    Dropout,
    Bidirectional,
    Add,
    Input,
    LayerNormalization,
)
from typing import Literal

ArchType = Literal[
    "lstm",
    "stacked_lstm",
    "residual_lstm",
    "gru",
    "stacked_gru",
    "residual_gru",
    "bi_lstm",
]

def build_neural_model(
    arch: str,
    vocab_size: int,
    input_length: int,
    embedding_dim: int = 128,
    hidden_units: int = 128,
    dropout_rate: float = 0.2,
) -> Model:
    """
    Factory function for recurrent language model architectures,
    now including Residual LSTMs and Residual GRUs with skip connections.
    """
    arch = arch.lower().strip()

    # 1. Residual LSTM Architecture
    if arch == "residual_lstm":
        inputs = Input(shape=(input_length,), dtype="int32", name="token_input")
        x = Embedding(input_dim=vocab_size, output_dim=hidden_units, input_length=input_length)(inputs)

        # Layer 1
        lstm1 = LSTM(hidden_units, return_sequences=True, name="lstm_res_1")(x)
        if dropout_rate > 0:
            lstm1 = Dropout(dropout_rate)(lstm1)
        # Residual skip connection 1
        res1 = Add(name="skip_1")([x, lstm1])
        res1 = LayerNormalization(name="norm_1")(res1)

        # Layer 2
        lstm2 = LSTM(hidden_units, return_sequences=False, name="lstm_res_2")(res1)
        if dropout_rate > 0:
            lstm2 = Dropout(dropout_rate)(lstm2)
        lstm2 = LayerNormalization(name="norm_2")(lstm2)

        # Output projection
        outputs = Dense(vocab_size, activation="softmax", name="vocab_projection")(lstm2)
        return Model(inputs=inputs, outputs=outputs, name="residual_lstm_model")

    # 2. Residual GRU Architecture
    elif arch == "residual_gru":
        inputs = Input(shape=(input_length,), dtype="int32", name="token_input")
        x = Embedding(input_dim=vocab_size, output_dim=hidden_units, input_length=input_length)(inputs)

        # Layer 1
        gru1 = GRU(hidden_units, return_sequences=True, name="gru_res_1")(x)
        if dropout_rate > 0:
            gru1 = Dropout(dropout_rate)(gru1)
        # Residual skip connection 1
        res1 = Add(name="skip_1")([x, gru1])
        res1 = LayerNormalization(name="norm_1")(res1)

        # Layer 2
        gru2 = GRU(hidden_units, return_sequences=False, name="gru_res_2")(res1)
        if dropout_rate > 0:
            gru2 = Dropout(dropout_rate)(gru2)
        gru2 = LayerNormalization(name="norm_2")(gru2)

        # Output projection
        outputs = Dense(vocab_size, activation="softmax", name="vocab_projection")(gru2)
        return Model(inputs=inputs, outputs=outputs, name="residual_gru_model")

    # Standard Sequential Models
    model = Sequential(name=f"seq_{arch}")
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length))

    if arch == "lstm":
        model.add(LSTM(hidden_units))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    elif arch == "stacked_lstm":
        model.add(LSTM(hidden_units, return_sequences=True))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
        model.add(LSTM(hidden_units))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    elif arch == "gru":
        model.add(GRU(hidden_units))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    elif arch == "stacked_gru":
        model.add(GRU(hidden_units, return_sequences=True))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
        model.add(GRU(hidden_units))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    elif arch == "bi_lstm":
        model.add(Bidirectional(LSTM(hidden_units)))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    else:
        raise ValueError(f"Unknown architecture: {arch}. Choose from: lstm, stacked_lstm, residual_lstm, gru, stacked_gru, residual_gru, bi_lstm")

    # Output projection with softmax across vocabulary
    model.add(Dense(vocab_size, activation="softmax"))
    return model
