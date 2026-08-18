from tensorflow import keras
from tensorflow.keras import layers


def build_cnn(window_size, feature_number, parameters=None):
    """Constructs a 1D CNN architecture."""

    parameters = parameters or {}
    filters = parameters.get("filters", [32, 64])
    kernel_sizes = parameters.get("kernel_sizes", [5, 3])
    dense_units = parameters.get("dense_units", 32)
    dropout = parameters.get("dropout", 0.30)

    if len(filters) != 2 or len(kernel_sizes) != 2:
        raise ValueError("CNN filters and kernel_sizes must each contain two values.")

    inputs = keras.Input(shape=(window_size, feature_number), name="plasma_signals")

    x = keras.layers.Conv1D(
        filters=filters[0],
        kernel_size=kernel_sizes[0],
        padding="causal",
        activation="relu",
    )(inputs)

    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv1D(
        filters=filters[1],
        kernel_size=kernel_sizes[1],
        padding="causal",
        activation="relu",
    )(x)

    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dense(dense_units, activation="relu")(x)
    x = keras.layers.Dropout(dropout)(x)

    outputs = keras.layers.Dense(
        1, activation="sigmoid", name="instability_probability"
    )(x)

    return keras.Model(inputs=inputs, outputs=outputs)


def residual_tcn_block(x, filters, kernel_size, dilation_rate, dropout):
    """Apply a residual block of dilated causal convolutions to a sequence."""

    residual = x

    x = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SpatialDropout1D(dropout)(x)

    x = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SpatialDropout1D(dropout)(x)

    if residual.shape[-1] != filters:
        residual = layers.Conv1D(filters=filters, kernel_size=1, padding="same")(
            residual
        )

    return layers.Add()([x, residual])


def build_tcn(window_size, feature_number, parameters=None):
    """Construct TCN architecture."""

    parameters = parameters or {}
    filters = parameters.get("filters", [32, 32, 64, 64])
    dilation_rates = parameters.get("dilation_rates", [1, 2, 4, 8])
    kernel_size = parameters.get("kernel_size", 3)
    block_dropout = parameters.get("block_dropout", 0.20)
    dense_units = parameters.get("dense_units", 32)
    dropout = parameters.get("dropout", 0.30)

    if len(filters) != len(dilation_rates):
        raise ValueError("TCN filters and dilation_rates must have equal lengths.")

    inputs = keras.Input(shape=(window_size, feature_number), name="plasma_signals")

    x = inputs

    for block_filters, dilation_rate in zip(filters, dilation_rates):
        x = residual_tcn_block(
            x,
            filters=block_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            dropout=block_dropout,
        )

        x = layers.Activation("relu")(x)

    x = layers.Cropping1D(cropping=(window_size - 1, 0))(x)
    x = layers.Flatten()(x)

    x = layers.Dense(
        dense_units,
        activation="relu",
        name="prediction_features",
    )(x)

    x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(1, activation="sigmoid", name="instability_probability")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="tcn")


def build_gru(window_size, feature_number, parameters=None):
    """Build and return a gated recurrent unit network."""
    parameters = parameters or {}
    units = parameters.get("units", [64, 32])
    sequence_dropout = parameters.get("sequence_dropout", 0.20)
    summary_dropout = parameters.get("summary_dropout", 0.30)
    dense_units = parameters.get("dense_units", 32)

    if len(units) != 2:
        raise ValueError("GRU units must contain two values.")

    inputs = keras.Input(shape=(window_size, feature_number), name="plasma_signals")

    x = layers.GRU(units[0], return_sequences=True, name="gru_sequence")(inputs)

    x = layers.Dropout(sequence_dropout)(x)

    x = layers.GRU(units[1], name="gru_summary")(x)

    x = layers.Dropout(summary_dropout)(x)
    x = layers.Dense(dense_units, activation="relu")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="instability_probability")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="gru")


def compile_model(model, learning_rate, classification_threshold):
    model.compile(
        optimizer=keras.optimizers.legacy.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(name="roc_auc"),
            keras.metrics.AUC(name="pr_auc", curve="PR"),
            keras.metrics.Precision(
                name="precision", thresholds=classification_threshold
            ),
            keras.metrics.Recall(
                name="recall",
                thresholds=classification_threshold,
            ),
        ],
    )

    return model


def build_model(name, window_size, feature_number, model_parameters=None):
    if name == "cnn":
        return build_cnn(window_size, feature_number, model_parameters)

    if name == "tcn":
        return build_tcn(window_size, feature_number, model_parameters)

    if name == "gru":
        return build_gru(window_size, feature_number, model_parameters)

    raise ValueError(f"Unsupported model {name!r}")
