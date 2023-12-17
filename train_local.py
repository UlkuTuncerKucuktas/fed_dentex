import argparse
from ultralytics import YOLO
import torch
import os
import gc

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train YOLO models with different training splits.')
    parser.add_argument('--config', type=str, required=True, help='Path to the YOLO configuration file')
    parser.add_argument('--model', type=str, required=True, help='Path to the YOLO model file')
    parser.add_argument('--epochs', type=int, default=4, help='Number of epochs for training')
    parser.add_argument('--batch', type=int, default=4, help='Batch size for training')
    parser.add_argument('--train_splits', nargs='+', type=str, required=True, help='Paths to different training split YAML configurations')
    return parser.parse_args()

def train_model(config, model_path, epochs, batch, data_config, save_name):
    # Load the model
    model = YOLO(config).load(model_path)

    # Start training
    model.train(data=data_config, epochs=epochs, batch=batch, name=save_name,workers=0)

    model.zero_grad()

def main():
    args = parse_arguments()

    for train_split in args.train_splits:
        # Extracting the base name of the YAML file without the extension
        split_name = os.path.basename(train_split).split('/')[-1][:-5]
        save_name = f'training_{split_name}'
        train_model(args.config, args.model, args.epochs, args.batch, train_split, save_name)
        gc.collect()  # Collect garbage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Empty CUDA cache

if __name__ == "__main__":
    main()
