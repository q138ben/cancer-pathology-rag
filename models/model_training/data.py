import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(data_dir):
    """
    Load and preprocess image data from a directory.

    Args:
        data_dir (str): Path to the directory containing training and validation data.

    Returns:
        train_data (tf.data.Dataset): Training dataset.
        val_data (tf.data.Dataset): Validation dataset.
    """
    # Define image size and batch size
    img_size = (224, 224)  # EfficientNet expects 224x224 images
    batch_size = 32

    # Data augmentation and preprocessing for training data
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,  # Normalize pixel values to [0, 1]
        rotation_range=20,  # Randomly rotate images
        width_shift_range=0.2,  # Randomly shift images horizontally
        height_shift_range=0.2,  # Randomly shift images vertically
        shear_range=0.2,  # Apply shear transformations
        zoom_range=0.2,  # Randomly zoom images
        horizontal_flip=True,  # Randomly flip images horizontally
        fill_mode="nearest",  # Fill missing pixels after transformations
    )

    # Preprocessing for validation data (only rescaling)
    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    # Load training data
    train_data = train_datagen.flow_from_directory(
        f"{data_dir}/train",  # Path to training data
        target_size=img_size,  # Resize images to 224x224
        batch_size=batch_size,
        class_mode="categorical",  # For multi-class classification
    )

    # Load validation data
    val_data = val_datagen.flow_from_directory(
        f"{data_dir}/validation",  # Path to validation data
        target_size=img_size,  # Resize images to 224x224
        batch_size=batch_size,
        class_mode="categorical",  # For multi-class classification
    )

    return train_data, val_data