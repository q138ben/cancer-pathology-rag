import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0

def build_efficientnet_model():
    # Load EfficientNetB0 with pre-trained weights (excluding the top layer)
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Add custom layers
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation='softmax')  # Adjust for your number of classes
    ])

    return model