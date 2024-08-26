import os
import nvtx
import argparse
import numpy as np
import torch.optim as optim
import torch.utils.data as torchdata

import scripts.utils as utils
from scripts.models import *
from scripts.CaloDiffu import *


if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Config file with training parameters')
    parser.add_argument('--steps', type=int, default=20, help='Number of steps to profile')
    parser.add_argument('--warmup', type=int, default=10, help='Number of steps to run before profiling')
    parser.add_argument('--batch-size', type=int, help='Number of samples per batch')
    parser.add_argument('--device', type=str, choices=["cpu", "cuda"], help='Type of device (cpu/cuda) to use for benchmark')
    args = parser.parse_args()

    dataset_config = utils.load_config(args.config, args.batch_size, args.device)
    device = torch.device(dataset_config['DEVICE'])
    batch_size = dataset_config['BATCH']
    training_obj = dataset_config.get('TRAINING_OBJ', 'noise_pred')
    loss_type = dataset_config.get("LOSS_TYPE", "l2")
    dataset_num = dataset_config.get('DATASET_NUM', 2)
    shower_embed = dataset_config.get('SHOWER_EMBED', '')
    orig_shape = ('orig' in shower_embed)
    energy_loss_scale = dataset_config.get('ENERGY_LOSS_SCALE', 0.0)

    data = []
    energies = []

    for i, dataset in enumerate(dataset_config['FILES']):
        data_, e_ = utils.DataLoader(
            os.path.join("../datasets", dataset),
            dataset_config['SHAPE_PAD'],
            emax = dataset_config['EMAX'],emin = dataset_config['EMIN'],
            max_deposit=dataset_config['MAXDEP'], #noise can generate more deposited energy than generated
            logE=dataset_config['logE'],
            showerMap = dataset_config['SHOWERMAP'],
            dataset_num  = dataset_num,
            orig_shape = orig_shape
        )

        if i==0: 
            data = data_
            energies = e_
        else:
            data = np.concatenate((data, data_))
            energies = np.concatenate((energies, e_))

    if 'NN' in shower_embed:
        particle = "photon" if dataset_num == 1 else "pion"
        bins = XMLHandler(particle, f"../CaloChallenge/code/binning_dataset_1_{particle}s.xml")
        NN_embed = NNConverter(bins=bins).to(device=device)
    else:
        NN_embed = None

    torch_data_tensor = torch.from_numpy(data)
    torch_E_tensor = torch.from_numpy(energies)
    del data, energies

    torch_dataset  = torchdata.TensorDataset(torch_E_tensor, torch_data_tensor)
    loader_train = iter(torchdata.DataLoader(torch_dataset, batch_size=batch_size, shuffle=True))
    del torch_data_tensor, torch_E_tensor, torch_dataset

    shape = dataset_config['SHAPE_PAD'][1:] if not orig_shape else dataset_config['SHAPE_ORIG'][1:]
    model = CaloDiffu(shape, config=dataset_config, NN_embed=NN_embed).to(device=device)
    optimizer = optim.Adam(model.parameters(), lr = float(dataset_config["LR"]))
    model.train()

    def step():
        step_rng = nvtx.start_range(message="Step")

        rng = nvtx.start_range(message="Zero grad")
        model.zero_grad()
        optimizer.zero_grad()
        nvtx.end_range(rng)
        
        rng = nvtx.start_range(message="Get batch")
        E, data = loader_train.__next__()
        data = data.to(device = device)
        E = E.to(device = device)
        nvtx.end_range(rng)

        rng = nvtx.start_range(message="Random t")
        t = torch.randint(0, model.nsteps, (data.size()[0],), device=device).long()
        nvtx.end_range(rng)

        rng = nvtx.start_range(message="Random noise")
        noise = torch.randn_like(data)
        nvtx.end_range(rng)

        rng = nvtx.start_range(message="Forward pass")
        batch_loss = model.compute_loss(data, E, noise = noise, t = t, loss_type = loss_type, energy_loss_scale = energy_loss_scale)
        nvtx.end_range(rng)

        rng = nvtx.start_range(message="Backward pass")
        batch_loss.backward()
        nvtx.end_range(rng)

        rng = nvtx.start_range(message="Update weights")
        optimizer.step()
        nvtx.end_range(rng)

        del data, E, noise, batch_loss
        nvtx.end_range(step_rng)
    
    # Warm-up
    for _ in range(args.warmup):
        step()

    # Start profiler
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(args.steps):
        step()
    torch.cuda.cudart().cudaProfilerStop()