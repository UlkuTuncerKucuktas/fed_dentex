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

def average_models(models):
    model_params = [model.model.state_dict() for model in models]
    averaged_params = {}
    for param_name in model_params[0]:
        params = torch.stack([model_params[i][param_name] for i in range(len(models))])
        params = params.to(torch.float16)
        averaged_params[param_name] = torch.mean(params, dim=0)
    return averaged_params

def plot_metrics(metrics, title='Training Metrics'):
    plt.figure(figsize=(10, 5))
    for metric in metrics:
        plt.plot(metrics[metric], label=metric)
    plt.xlabel('Communication Round')
    plt.ylabel('Metric Value')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'/content/drive/MyDrive/federated_results/{title}.png')

def evaluate_model(model_state, data_config,config):
    model = YOLO(config)
    model.model.load_state_dict(model_state)
    results = model.val(data=data_config)
    mean_precision, mean_recall, mean_map50, mean_map_50_95 = results.mean_results()
    # Using the mean_results method to get mean precision, recall, mAP50, and mAP50-95
    mean_precision, mean_recall, mean_map50, mean_map_50_95 = results.mean_results()
    return {'precision': mean_precision, 'recall': mean_recall, 'map50': mean_map50, 'map_50_95': mean_map_50_95}


def main():
    args = parse_arguments()

    # Disable WandB
    os.environ['WANDB_DISABLED'] = 'true'
    temp_dir = '/content/runs'
    metrics = {'map50': [], 'precision': [], 'recall': [],'map_50_95': []}

    model = YOLO(args.config).load(args.model)

    for comm_round in range(args.num_comm):
        print("-----------------------------------------------------------------------------")
        print('Communication Round: ', comm_round)
        print("-----------------------------------------------------------------------------")

        models = [copy.deepcopy(model) for _ in range(len(args.train_splits))]

        validation_scores = []

        for index, dup_model in enumerate(models):
            dup_model.train(data=args.train_splits[index], epochs=args.local_epochs, batch=args.batch,save=True, resume=True, workers=0)
            dup_model.zero_grad()

            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


        ensemble_model_params = average_models(models)

        model.model.load_state_dict(ensemble_model_params)

        # Evaluate the averaged model
        
        eval_results = evaluate_model(ensemble_model_params, args.train_splits[0],args.config)
        metrics['map50'].append(eval_results['map50'])
        metrics['map_50_95'].append(eval_results['map_50_95'])
        metrics['precision'].append(eval_results['precision'])
        metrics['recall'].append(eval_results['recall'])
        

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    plot_metrics(metrics, title='Federated Training Metrics')
    torch.save(model, '/content/drive/MyDrive/federated_results/final_federated_model.pt')

if __name__ == "__main__":
    main()
