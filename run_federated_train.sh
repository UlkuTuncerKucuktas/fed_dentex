python train_federated.py --config './configs/yolov8x.yaml' --model 'yolov8x.pt' --num_comm 5 --local_epochs 10 --batch 32 --weighted_avg  --train_splits './configs/train_15.yaml' './configs/train_20.yaml' './configs/train_25.yaml' './configs/train_40.yaml'
