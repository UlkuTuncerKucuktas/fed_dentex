python train_federated.py --config './configs/yolov8x.yaml' --model 'yolov8x.pt' --num_comm 2 --local_epochs 1 --batch 8 --weighted_avg --train_splits './configs/train_15.yaml' './configs/train_20.yaml' './configs/train_25.yaml' './configs/train_40.yaml'
