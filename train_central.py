import argparse
from ultralytics import YOLO
import torch

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a YOLO model with specified parameters.')
    parser.add_argument('--config', type=str, required=True, help='Path to the YOLO configuration file')
    parser.add_argument('--model', type=str, required=True, help='Path to the YOLO model file')
    parser.add_argument('--epochs', type=int, default=4, help='Number of epochs for training')
    parser.add_argument('--batch', type=int, default=4, help='Batch size for training')
    parser.add_argument('--data', type=str, required=True, help='Path to the dataset configuration file')
    parser.add_argument('--save_name', type=str, required=True, help='Name for saving the training results in the runs folder')
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Load the model
    model = YOLO(args.config).load(args.model)

    # Start training
    model.train(data=args.data, epochs=args.epochs, batch=args.batch, name=args.save_name)

if __name__ == "__main__":
    main()
