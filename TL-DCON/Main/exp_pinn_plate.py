import yaml
import torch
import argparse
from data import generate_plate_dataloader, generate_generalization_test_data
from models import DeepONet_plate, Improved_DeepONet_plate, DCON_plate, New_model_plate
from plate_utils import train, predict


# define arguements
parser = argparse.ArgumentParser(description='command setting')
parser.add_argument('--phase', type=str, default='predict', choices=['train', 'predict'])
parser.add_argument('--data', type=str, default='plate_dis_high')
parser.add_argument('--model', type=str, default='DCON')
args = parser.parse_args()
print('Model forward phase: {}'.format(args.phase))
print('Using dataset: {}'.format(args.data))
print('Using model: {}'.format(args.model))

# extract configuration
with open(r'./configs/{}_{}.yaml'.format(args.model, args.data), 'r') as stream:
    config = yaml.load(stream, yaml.FullLoader)

# load the data
if args.phase == 'train':
    coors, train_loader, val_loader, test_loader,\
        youngs, nu, num_bc_nodes,\
        flag_BCxy, flag_BCy, flag_load = generate_plate_dataloader(config)
elif args.phase == 'predict':
    # For generalization test: new BCs (free top/bottom, fixed hole, 2mm displacement on left/right)
    coors, test_loader, youngs, nu, num_bc_nodes,\
        flag_BCxy, flag_BCy, flag_load = generate_generalization_test_data(config)

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

# model training or prediction
if args.phase == 'train':
    train(args, config, model, device, (train_loader, val_loader, test_loader),
          coors, flag_BCxy, flag_BCy, flag_load, [youngs, nu])
elif args.phase == 'predict':
    predict(args, config, model, device, test_loader, coors, flag_BCxy, flag_BCy, flag_load, [youngs, nu])