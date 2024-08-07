from tensorflow.keras.layers import TransformerEncoder, Dense, Input
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.regularizers import l2
import pandas as pd
import numpy as np
y_train = X_train
y_val = X_val

# Load data
numerical_data = pd.read_csv('chroma_features.csv')

# Ensure the data has three dimensions: (num_samples, timesteps, features)
timesteps = 1500  # or whatever number of timesteps you expect
features = 12

# Reshape data
num_samples = len(numerical_data) // timesteps
X = numerical_data.values[:num_samples * timesteps].reshape((num_samples, timesteps, features))

# Split data
X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

# Normalize data
X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())
X_val = (X_val - X_val.min()) / (X_val.max() - X_val.min())

class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, x, training):
        attn_output = self.mha(x, x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
transformer_layer = TransformerEncoderLayer(d_model=64, num_heads=8, dff=256)
x = transformer_layer(input_layer)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = transformer_layer(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = transformer_layer(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
output_layer = Dense(features, activation='softmax')(x)

model_transformer = Model(input_layer, output_layer)
model_transformer.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model_transformer.keras', save_best_only=True, monitor='val_loss')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

model_transformer.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=[early_stopping, model_checkpoint, reduce_lr])
model_transformer.load_weights('best_model_transformer.keras')
