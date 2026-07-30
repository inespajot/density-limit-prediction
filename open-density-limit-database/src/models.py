from tensorflow import keras
from tensorflow.keras import layers

def build_cnn(window_size,feature_number):
    inputs = keras.Input(shape=(window_size,feature_number),
                         name = "plasma_signals")

    x = keras.layers.Conv1d


def residual_tcn_block(x,filters,kernel_size,dialation_rate,dropout):
    residual = x 

    x = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="casual",
        dilation_rate=dilation_rate)(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SpatialDropout1D(dropout)(x)

    x = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="causal",
        dialation_rate=dialation_rate
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SpatialDropout1D(dropout)(x)

    if residual.shape[-1]!=filters:
        residual = layers.Conv1D(
            filters=filters
            kernel_size=1
            padding="same",)(residual)

    return layers.Add()([x,residual])