from tensorflow import keras
from tensorflow.keras import layers


def build_cnn(window_size, feature_number):
    """Constructs a 1D CNN architecture."""

    inputs = keras.Input(shape=(window_size, feature_number), name="plasma_signals")

    x = keras.layers.Conv1D(
        filters=32, kernel_size=5, padding="causal", activation="relu"
    )(inputs)

    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv1D(
        filters=64, kernel_size=3, padding="causal", activation="relu"
    )(x)

    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dropout(0.30)(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    x = keras.layers.Dropout(0.30)(x)

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


def build_tcn(window_size, feature_number):
    """Construct TCN architecture."""

    inputs = keras.Input(shape=(window_size, feature_number), name="plasma_signals")

    x = inputs

    for filters, dilation_rate in [(32, 1), (32, 2), (64, 4), (64, 8)]:
        x = residual_tcn_block(
            x,
            filters=filters,
            kernel_size=3,
            dilation_rate=dilation_rate,
            dropout=0.20,
        )

        x = layers.Activation("relu")(x)

    x = layers.Cropping1D(cropping=(window_size - 1, 0))(x)
    x = layers.Flatten()(x)

    x = layers.Dense(
        32,
        activation="relu",
        name="prediction_features",
    )(x)

    x = layers.Dropout(0.30)(x)

    outputs = layers.Dense(1, activation="sigmoid", name="instability_probability")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="tcn")


def build_gru(window_size, feature_number):
    """Build and return a gated recurrent unit network."""
    inputs = keras.Input(shape=(window_size, feature_number), name="plasma_signals")

    x = layers.GRU(64, return_sequences=True, name="gru_sequence")(inputs)

    x = layers.Dropout(0.20)(x)

    x = layers.GRU(32, name="gru_summary")(x)

    x = layers.Dropout(0.30)(x)
    x = layers.Dense(32, activation="relu")(x)

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


def build_model(name, window_size, feature_number):
    if name == "cnn":
        return build_cnn(window_size, feature_number)

    if name == "tcn":
        return build_tcn(window_size, feature_number)

    if name == "gru":
        return build_gru(window_size, feature_number)

    raise ValueError(f"Unsupported model {name!r}")
