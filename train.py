## PyTorch libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed import rpc
from torch.distributed.pipeline.sync import Pipe
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

## Custom libraries
import maxvit
import pipe_maxvit

## Other libraries
import torchgpipe
import numpy as np
import wandb
import argparse
import subprocess as sp
from threading import Timer
import time
import logging
import tqdm
import os
import tempfile

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataset():
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                                     transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True,
                                    transform=transform)

    return train_dataset, test_dataset


def train(train_loader, model, epoch_number, learning_rate, device):
    # Train the MaxViT model
    num_epochs = epoch_number
    # Set the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        for i, (images, labels) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        epoch_time = time.time() - start_time
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] {epoch_time}s, Loss: {running_loss / len(train_loader):.4f}")
    return model

###################### Distributed Data Parallelism Function ##########################
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def train_dp(rank, world_size, epoch_number, learning_rate, batch_size):
    # Setup process group
    setup(rank, world_size)

    # Load the training dataset
    train_dataset, _ = get_dataset()

    # Initialize the MaxViT model and move it to the current process
    model = maxvit.max_vit_tiny_224(num_classes=10).to(rank)

    # Wrap the model in DistributedDataParallel and assign it to this process
    model = DDP(model, device_ids=[rank])

    # Create a data loader for training data with DistributedSampler
    train_loader = DataLoader(train_dataset, batch_size= int(batch_size / world_size), shuffle=False,
                              num_workers=2, sampler=DistributedSampler(train_dataset))

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize previous_loss variable
    previous_loss = 0.0

    # Set the number of epochs
    num_epochs = epoch_number
    start_dev_id = 2
    # Set the devices to use for the model parallelism
    devices = [start_dev_id + i for i in range(world_size)]

    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0

        # Loop over batches of data
        for i, (images, labels) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            # Move data to the current process
            images, labels = images.to(rank), labels.to(rank)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch + 1}/{num_epochs}] {epoch_time}s, Loss: {running_loss / len(train_loader):.4f}")

    return model

###################### Pipeline Parallelism Function ################################################

def train_pipeline(train_loader, model, epoch_number, learning_rate):
    # set the number of epochs, input and output devices
    num_epochs = epoch_number
    in_device = model.devices[0]
    out_device = model.devices[-1]

    # set the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # initialize previous_loss
    previous_loss = 0.0

    # loop over the epochs
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0

        # loop over the batches in the training set
        for i, (images, labels) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            images, labels = images.to(in_device), labels.to(out_device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch + 1}/{num_epochs}] {epoch_time}s, Loss: {running_loss / len(train_loader):.4f}")
        previous_loss = running_loss

    # return the trained model
    return model

###################### Distributed Data Parallelism + Pipeline Parallelism Functions ##########################
def init_rpc(rank):
    tmpfile = tempfile.NamedTemporaryFile()
    rpc.init_rpc(
        name="worker",
        rank=0,
        world_size=1,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            init_method="file://{}".format(tmpfile.name),
            _transports=["ibv", "uv"],
            _channels=["cuda_ipc", "cuda_basic"],
        )
    )

def print_model_parameters(model, print_with_rank):
    def get_total_params(module: torch.nn.Module):
        total_params = 0
        for param in module.parameters():
            total_params += param.numel()
        return total_params

    print_with_rank('Total parameters in model: {:,}'.format(get_total_params(model)))

def create_model(rank):
    layers = pipe_maxvit.maxvit(num_classes=10, array=True)
    tmp_list = [
        layers[0].to(2 * rank + 0),
        layers[1].to(2 * rank + 0),
        layers[2].to(2 * rank + 0),
        layers[3].to(2 * rank + 1),
        layers[4].to(2 * rank + 1),
        layers[5].to(2 * rank + 1)
    ]
    module_list = nn.Sequential(*tmp_list)

    # Build the pipeline
    model = Pipe(module_list, chunks=8, checkpoint='never')
    return model

def init_ddp(rank, world_size, model):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    model = DDP(model)
    return model

def train_dp_pp_model(rank, num_epochs, model, train_loader, optimizer, criterion, print_with_rank):
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for i, (images, labels) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            images, labels = images.to(2 * rank), labels.to(2 * rank)
            optimizer.zero_grad()
            # Since the Pipe is only within a single host and process the ``RRef``
            # returned by forward method is local to this node and can simply
            # retrieved via ``RRef.local_value()``.
            output = model(images).local_value()
            # Need to move targets to the device where the output of the
            # pipeline resides.
            loss = criterion(output.cuda(2 * rank + 1), labels.cuda(2 * rank + 1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            running_loss += loss.item()
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch}/{num_epochs}] {epoch_time}s, Loss: {running_loss / len(train_loader):.4f}")

def train_dp_pipeline(rank, world_size, epoch_number, batch_size, gpu):
    def print_with_rank(msg):
        print('[RANK {}]: {}'.format(rank, msg))
    
    # Initialize RPC
    init_rpc(rank)

    # Create model
    num_gpus = 2
    model = create_model(rank)

    # Initialize process group and wrap model in DDP
    model = init_ddp(rank, world_size, model)

    # Print model parameters
    print_model_parameters(model, print_with_rank)
    
    # Set up training components
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Load and prepare the dataset
    train_dataset, _ = get_dataset()
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size= int(batch_size / num_gpus), shuffle=False, num_workers=2, sampler=train_sampler)

    # Training loop
    train_dp_pp_model(rank, epoch_number, model, train_loader, optimizer, criterion, print_with_rank)

if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", "-e", required=True, type=int, help="Number of epochs")
    parser.add_argument("--batch_size", "-b", required=True, type=int, help="Batch size")
    parser.add_argument("--model", "-m", required=True, type=str, choices=['baseline', 'PP', 'DP', 'DP_PP'], help="Model type")
    parser.add_argument('--gpu', '-g', required=True, type=int, help='Number of GPUs')
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    # extract the arguments from the parser
    epoch_number = args.epoch
    batch_size = args.batch_size
    model_type = args.model
    gpus = args.gpu
    learning_rate = 0.0005

    # create train dataset
    print('Creating Dataset')
    train_dataset, _ = get_dataset()

    # create train data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # initialize the model
    if model_type == 'baseline':
        device = 'cuda:2' # specify your GPU
        model = maxvit.max_vit_tiny_224(num_classes=10).to(device)
        model = train(train_loader, model, epoch_number, learning_rate, device)
    if model_type == 'PP':
        # use pipeline parallelism
        model = pipe_maxvit.maxvit(num_classes=10)
        if gpus == 4:
            devices = [0, 1, 2, 3] # specify the list of GPUs
            model = torchgpipe.GPipe(model, [2, 1, 1, 3], devices=devices, chunks=8)
        elif gpus == 2:
            devices = [2, 3] # specify the list of GPUs
            model = torchgpipe.GPipe(model, [3, 4], devices=devices, chunks=8)
        else:
            print('Invalid number of GPUs')
            exit(1)

        # train the model
        model = train_pipeline(train_loader, model, epoch_number, learning_rate)

    elif model_type == 'DP':
        # use data parallelism
        world_size = gpus # specify the number of GPUs here
        mp.spawn(train_dp, args=(world_size, epoch_number, learning_rate, batch_size), nprocs=world_size)
        dist.destroy_process_group()

    elif model_type == 'DP_PP':
        # The number of groups you want can be determined using this parameter. 
        # The dataset will be split into the specified number of groups, each consisting of two GPUs. 
        # To utilize this training method, you will need to have at least four GPUs
        # Also, the number of GPUs must be even.
        world_size = 2 
        mp.spawn(train_dp_pipeline, args=(world_size, epoch_number, batch_size, gpus), nprocs=world_size, join=True)
    else:
        # invalid model type
        print('Invalid model type')
        exit(1)