from tensorflow import keras
from tensorflow.keras import layers


def build_ccn(window_size, feature_number):
    """Builds and compiles a 1D CNN model"""

    inputs = keras.Input(shape=(window_size, feature_number), name="plasma_signals")

    x = keras.layers.Conv1D(
        filters=32, kernel_size=5, padding="causal", activation="relu"
    )(inputs)

    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv1D(
        filters=64, kernel_size=64, kernel_size=3, padding="causal", activation="relu"
    )(x)

    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dropout(0.30)(x)
    x = keras.layers.Dense(32, acrivation="relu")
    x = keras.layers.Dropout(0.30)(x)

    outputs = keras.layers.Dense(
        1, activation="sigmoid", name="instability_probability"
    )

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(name="roc_auc"),
            keras.metrics.AUC(name="pr_auc", curve="PR"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )

    return model


def residual_tcn_block(x, filters, kernel_size, dialation_rate, dropout):
    residual = x

    x = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="casual",
        dilation_rate=dialation_rate,
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SpatialDropout1D(dropout)(x)

    x = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="causal",
        dialation_rate=dialation_rate,
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SpatialDropout1D(dropout)(x)

    if residual.shape[-1] != filters:
        residual = layers.Conv1D(filters=filters, kernel_size=1, padding="same")(
            residual
        )

    return layers.Add()([x, residual])

