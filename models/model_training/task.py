import argparse
import os
import tensorflow as tf
from .model import build_efficientnet_model
from .data import load_data

def train_model(args):
    # Load data
    train_data, val_data = load_data(args.data_dir)

    # Build the EfficientNet model
    model = build_efficientnet_model()

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_data, validation_data=val_data, epochs=args.epochs, batch_size=args.batch_size)

    # Save the model
    model.save(args.model_dir)

if __name__ == "__main__":
    # Local testing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=f'{os.getenv('MOUNT_FOLDER_PATH')}/data/data')
    parser.add_argument('--model_dir', type=str, default=f'{os.getenv('MOUNT_FOLDER_PATH')}/models')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()
    train_model(args)
