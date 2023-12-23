import os
import copy
import torch
from ultralytics import YOLO
import shutil
import gc
import matplotlib.pyplot as plt
import argparse
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description='Federated Training of YOLO Models with Weighted Averaging')
    parser.add_argument('--config', type=str, required=True, help='Path to the YOLO configuration file')
    parser.add_argument('--model', type=str, required=True, help='Path to the YOLO model file')
    parser.add_argument('--local_epochs', type=int, default=1, help='Number of epochs for training in each local training phase')
    parser.add_argument('--batch', type=int, default=8, help='Batch size for training')
    parser.add_argument('--train_splits', nargs='+', type=str, required=True, help='Paths to different training split YAML configurations')
    parser.add_argument('--num_comm', type=int, default=30, help='Number of communication rounds')
    parser.add_argument('--weighted_avg', action='store_true', help='Use weighted averaging based on validation performance')
    return parser.parse_args()

def delete_weights_folders(directory):
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in dirs:
            if name == 'weights':
                weights_folder_path = os.path.join(root, name)
                print(f"Deleting folder: {weights_folder_path}")
                shutil.rmtree(weights_folder_path)

def average_models(models):
    model_params = [model.model.state_dict() for model in models]
    averaged_params = {}
    for param_name in model_params[0]:
        params = torch.stack([model_params[i][param_name] for i in range(len(models))])
        params = params.to(torch.float16)
        averaged_params[param_name] = torch.mean(params, dim=0)
    return averaged_params

def plot_metrics(metrics, title, filename):
    plt.figure(figsize=(10, 5))
    for metric in metrics:
        plt.plot(metrics[metric], label=metric)
    plt.xlabel('Communication Round')
    plt.ylabel('Metric Value')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def evaluate_model(model_state, data_config, config):
    model = YOLO(config)
    model.model.load_state_dict(model_state)
    results = model.val(data=data_config)
    mean_precision, mean_recall, mean_map50, mean_map_50_95 = results.mean_results()
    return {'precision': mean_precision, 'recall': mean_recall, 'map50': mean_map50, 'map_50_95': mean_map_50_95}

def main():
    args = parse_arguments()

    # Disable WandB
    os.environ['WANDB_DISABLED'] = 'true'
    temp_dir = '/content/fed_dentex/runs'
    split_metrics = {split: {'map50': [], 'precision': [], 'recall': [], 'map_50_95': []} for split in args.train_splits}

    model = YOLO(args.config).load(args.model)

    best_map = 0

    for comm_round in range(args.num_comm):
        print("-----------------------------------------------------------------------------")
        print('Communication Round: ', comm_round)
        print("-----------------------------------------------------------------------------")

        models = [copy.deepcopy(model) for _ in range(len(args.train_splits))]

        for index, dup_model in enumerate(models):
            dup_model.train(data=args.train_splits[index], epochs=args.local_epochs, batch=args.batch, save=True, resume=True, workers=0)
            dup_model.zero_grad()

            if os.path.exists(temp_dir):
                delete_weights_folders(temp_dir)

        ensemble_model_params = average_models(models)
        model.model.load_state_dict(ensemble_model_params)

        # Evaluate the averaged model on all splits
        for split in args.train_splits:
            eval_results = evaluate_model(ensemble_model_params, split, args.config)
            for key in split_metrics[split]:
                split_metrics[split][key].append(eval_results[key])

            if eval_results['map50'] > best_map:
                best_map = eval_results['map50']
                torch.save(model, '/content/drive/MyDrive/federated_results/best.pt')

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Plot and save metrics for each split
    for split in args.train_splits:
        split_name = os.path.basename(split).split('.')[0]
        plot_metrics(split_metrics[split], title=f'Federated Training Metrics - {split_name}', 
                     filename=f'/content/drive/MyDrive/federated_results/metrics_{split_name}.png')

    torch.save(model, '/content/drive/MyDrive/federated_results/final_federated_model.pt')

if __name__ == "__main__":
    main()
