import argparse
from ultralytics import YOLO
import torch
import os 

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train and validate a YOLO model with multiple validation datasets.')
    parser.add_argument('--config', type=str, required=True, help='Path to the YOLO configuration file')
    parser.add_argument('--model', type=str, required=True, help='Path to the YOLO model file')
    parser.add_argument('--epochs', type=int, default=4, help='Number of epochs for training')
    parser.add_argument('--batch', type=int, default=4, help='Batch size for training')
    parser.add_argument('--train_data', type=str, required=True, help='Path to the training dataset configuration file')
    parser.add_argument('--val_data', type=str, nargs='+', required=True, help='Paths to the validation dataset configuration files')
    parser.add_argument('--save_name', type=str, required=True, help='Name for saving the trained model')
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Load the model
    model = YOLO(args.config).load(args.model)

    # Start training
    model.train(data=args.train_data, epochs=args.epochs, batch=args.batch, name=args.save_name)

    # Validate the model on each validation dataset
    for val_dataset in args.val_data:
        print(f"Validating on dataset: {val_dataset}")
        val_dataset_name = os.path.split(val_dataset)[1].split('.')[0] + '_val'
        model.val(data=val_dataset,name=val_dataset_name)

if __name__ == "__main__":
    main()
