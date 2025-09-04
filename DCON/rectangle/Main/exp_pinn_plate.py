import yaml
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from data import generate_plate_dataloader
from models import DeepONet_plate, Improved_DeepONet_plate, DCON_plate, New_model_plate
from plate_utils import train, test

# define arguements
parser = argparse.ArgumentParser(description='command setting')
parser.add_argument('--phase', type=str, default='predict')
parser.add_argument('--data', type=str, default='Dataset_Rectangle')
parser.add_argument('--model', type=str, default='DCON')
args = parser.parse_args()
print('Model forward phase: {}'.format(args.phase))
print('Using dataset: {}'.format(args.data))
print('Using model: {}'.format(args.model))

# extract configuration
with open(r'./configs/{}_{}.yaml'.format(args.model, args.data), 'r') as stream:
    config = yaml.load(stream, yaml.FullLoader)

# load the data
coors, train_loader, val_loader, test_loader,\
    youngs, nu, num_bc_nodes,\
    flag_BCxy, flag_BCy, flag_load = generate_plate_dataloader(config)

# define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define model
if args.model == 'DCON':
    model = DCON_plate(config)
if args.model == 'DON':
    model = DeepONet_plate(config, num_bc_nodes)
if args.model == 'IDON':
    model = Improved_DeepONet_plate(config, num_bc_nodes)
if args.model == 'self_defined':
    model = New_model_plate(config)

# 根据不同的阶段执行不同的操作
if args.phase == 'train':
    # model training
    train(args, config, model, device, (train_loader, val_loader, test_loader), coors, flag_BCxy, flag_BCy, flag_load,
          [youngs, nu])

elif args.phase == 'predict':
    # 预测阶段 - 泛化检验
    print("Starting prediction phase...")

    # 加载训练好的模型
    model_path = r'../res/saved_models/best_model_{}_{}.pkl'.format(args.data, args.model)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded trained model from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file {model_path} not found. Please train the model first.")
        exit(1)

    model.to(device)

    # 创建预测数据
    predict_loader, num_predict_nodes = create_predict_data(config, coors, flag_load, args.load_value)

    # 进行预测
    predict(model, predict_loader, coors, device, args, args.load_value)

else:
    print("Invalid phase. Use 'train' or 'predict'.")