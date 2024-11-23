import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore

actions = os.listdir(r"C:\Users\nagad\SanketUvach\keypoint_data")

def lstm_v1(device_name):

    with tf.device(device_name):
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 150)))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        model.add(LSTM(256, return_sequences=False, activation='relu'))

        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(len(actions), activation='softmax'))

    return model


def lstm_v2(device_name):

    with tf.device(device_name):
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 150)))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        model.add(LSTM(256, return_sequences=True, activation='relu'))
        model.add(LSTM(256, return_sequences=False, activation='relu'))

        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(len(actions), activation='softmax'))

    return model

def lstm_v3(device_name):

    with tf.device(device_name):
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=(30, 150)))
        model.add(LSTM(256, return_sequences=True))
        model.add(LSTM(128, return_sequences=False))

        model.add(Dense(1024, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(len(actions), activation='softmax'))

    return model


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def transformer(
        input_shape=(30, 150),
        output_shape=len(actions),
        head_size=512,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=4,
        mlp_units=[128],
        dropout=0.1,
        mlp_dropout=0.1,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(output_shape, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def compile_model(model):
    adam = tf.keras.optimizers.Adam(3e-4)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

def load_model(name='lstm_v3', pretrained=False, training=True, device=None):
    if not pretrained:
        if name == 'lstm_v1':
            model = lstm_v1(device)
        elif name == 'lstm_v2':
            model = lstm_v2(device)
        elif name == 'lstm_v3':
            model = lstm_v3(device)
        elif name == 'transformer':
            model = transformer()
        else:
            raise ValueError(f"Unknown model name: {name}")
        
        if training:
            return compile_model(model)
        return model

    # Handle pretrained model loading
    try:
        # Use os.path.join for proper path construction
        model_dir = os.path.abspath(os.path.join("models", name))
        if not os.path.exists(model_dir):
            raise ValueError(f"Model directory not found: {model_dir}")

        # Find .h5 file in the directory
        h5_files = [f for f in os.listdir(model_dir) if f.endswith('.keras')]
        if not h5_files:
            raise ValueError(f"No .keras files found in {model_dir}")

        model_path = os.path.join(model_dir, h5_files[0])
        print(f"Loading Model from: {model_path}")

        # Load the model
        try:
            # First try loading directly
            model = tf.keras.models.load_model(model_path)
        except OSError as e:
            print(f"Warning: Initial loading failed, attempting alternative loading method: {str(e)}")
            # If direct loading fails, try saving and reloading in SavedModel format
            temp_save_path = os.path.join(model_dir, "temp_saved_model")
            if os.path.exists(temp_save_path):
                import shutil
                shutil.rmtree(temp_save_path)
            
            model = tf.keras.models.load_model(model_path, compile=False)
            model.save(temp_save_path, save_format="tf")
            model = tf.keras.models.load_model(temp_save_path)

        if training:
            return compile_model(model)
        return model

    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")