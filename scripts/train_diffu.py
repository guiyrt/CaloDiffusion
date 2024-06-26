import os
import argparse
import numpy as np
import torch.optim as optim
import torch.utils.data as torchdata

import scripts.utils as utils
from scripts.models import *
from scripts.CaloDiffu import *


if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    print("TRAIN DIFFU")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', default='../datasets', help='Folder containing data and MC files')
    parser.add_argument('-c', '--config', default='configs/test.json', help='Config file with training parameters')
    parser.add_argument('--nevts', type=int, default=-1, help='Number of events to load')
    parser.add_argument('--frac', type=float, default=0.85, help='Fraction of total events used for training')
    parser.add_argument('--load', action='store_true', default=False, help='Load pretrained weights to continue the training')
    parser.add_argument('--seed', type=int, default=1234, help='Pytorch seed')
    parser.add_argument('--reset_training', action='store_true', default=False,help='Retrain')
    args = parser.parse_args()

    dataset_config = utils.LoadJson(args.config)

    print("TRAINING OPTIONS")
    print(dataset_config, flush = True)

    torch.manual_seed(args.seed)

    nholdout  = dataset_config.get('HOLDOUT', 0)

    batch_size = dataset_config['BATCH']
    num_epochs = dataset_config['MAXEPOCH']
    early_stop = dataset_config['EARLYSTOP']
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
            os.path.join(args.data_folder, dataset),
            dataset_config['SHAPE_PAD'],
            emax = dataset_config['EMAX'],emin = dataset_config['EMIN'],
            nevts = args.nevts,
            max_deposit=dataset_config['MAXDEP'], #noise can generate more deposited energy than generated
            logE=dataset_config['logE'],
            showerMap = dataset_config['SHOWERMAP'],
            nholdout = nholdout if i == (len(dataset_config['FILES'])-1) else 0,
            dataset_num  = dataset_num,
            orig_shape = orig_shape
        )

        if(i ==0): 
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

    num_data = data.shape[0]
    print("Data Shape " + str(data.shape))
    #print("Pre-processed shower mean %.2f std dev %.2f" % (np.mean(data), np.std(data)))
    torch_data_tensor = torch.from_numpy(data)
    torch_E_tensor = torch.from_numpy(energies)
    del data

    torch_dataset  = torchdata.TensorDataset(torch_E_tensor, torch_data_tensor)
    nTrain = int(round(args.frac * num_data))
    nVal = num_data - nTrain
    train_dataset, val_dataset = torch.utils.data.random_split(torch_dataset, [nTrain, nVal])

    loader_train = torchdata.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    loader_val = torchdata.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    del torch_data_tensor, torch_E_tensor, train_dataset, val_dataset
    checkpoint_folder = '../models/{}/'.format(dataset_config['CHECKPOINT_NAME'])
    os.makedirs(checkpoint_folder, exist_ok=True)

    checkpoint = dict()
    checkpoint_path = os.path.join(checkpoint_folder, "checkpoint.pth")
    if(args.load and os.path.exists(checkpoint_path)): 
        print("Loading training checkpoint from %s" % checkpoint_path, flush = True)
        checkpoint = torch.load(checkpoint_path, map_location = device)
        print(checkpoint.keys())



    shape = dataset_config['SHAPE_PAD'][1:] if not orig_shape else dataset_config['SHAPE_ORIG'][1:]
    model = CaloDiffu(shape, config=dataset_config, training_obj=training_obj, NN_embed=NN_embed, nsteps=dataset_config['NSTEPS']).to(device=device)


    #sometimes save only weights, sometimes save other info
    if('model_state_dict' in checkpoint.keys()): model.load_state_dict(checkpoint['model_state_dict'])
    elif(len(checkpoint.keys()) > 1): model.load_state_dict(checkpoint)

    early_stopper = EarlyStopper(patience = dataset_config['EARLYSTOP'], mode = 'diff', min_delta = 1e-5)
    if('early_stop_dict' in checkpoint.keys() and not args.reset_training): early_stopper.__dict__ = checkpoint['early_stop_dict']
    print(early_stopper.__dict__)
    

    criterion = nn.MSELoss().to(device=device)

    optimizer = optim.Adam(model.parameters(), lr = float(dataset_config["LR"]))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer, factor = 0.1, patience = 15, verbose = True) 
    if('optimizer_state_dict' in checkpoint.keys() and not args.reset_training): optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if('scheduler_state_dict' in checkpoint.keys() and not args.reset_training): scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    training_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    start_epoch = 0
    min_validation_loss = 99999.
    if('train_loss_hist' in checkpoint.keys() and not args.reset_training): 
        training_losses = checkpoint['train_loss_hist']
        val_losses = checkpoint['val_loss_hist']
        start_epoch = checkpoint['epoch'] + 1

    #training loop
    for epoch in range(start_epoch, num_epochs):
        print("Beginning epoch %i" % epoch, flush=True)
        for i, param in enumerate(model.parameters()):
            break
        train_loss = 0

        model.train()
        for i, (E,data) in tqdm(enumerate(loader_train, 0), unit="batch", total=len(loader_train)):
            model.zero_grad()
            optimizer.zero_grad()

            data = data.to(device = device)
            E = E.to(device = device)

            t = torch.randint(0, model.nsteps, (data.size()[0],), device=device).long()
            noise = torch.randn_like(data)

            batch_loss = model.compute_loss(data, E, noise = noise, t = t, loss_type = loss_type, energy_loss_scale = energy_loss_scale)
            batch_loss.backward()

            optimizer.step()
            train_loss+=batch_loss.item()

            del data, E, noise, batch_loss

        train_loss = train_loss/len(loader_train)
        training_losses[epoch] = train_loss
        print("loss: "+ str(train_loss))

        val_loss = 0
        model.eval()
        for i, (vE, vdata) in tqdm(enumerate(loader_val, 0), unit="batch", total=len(loader_val)):
            vdata = vdata.to(device=device)
            vE = vE.to(device = device)

            t = torch.randint(0, model.nsteps, (vdata.size()[0],), device=device).long()
            noise = torch.randn_like(vdata)

            batch_loss = model.compute_loss(vdata, vE, noise = noise, t = t, loss_type = loss_type, energy_loss_scale = energy_loss_scale)

            val_loss+=batch_loss.item()
            del vdata,vE, noise, batch_loss

        val_loss = val_loss/len(loader_val)
        val_losses[epoch] = val_loss
        print("val_loss: "+ str(val_loss), flush = True)

        scheduler.step(torch.tensor([train_loss]))

        if(val_loss < min_validation_loss):
            torch.save(model.state_dict(), os.path.join(checkpoint_folder, 'best_val.pth'))
            min_validation_loss = val_loss

        if(early_stopper.early_stop(val_loss - train_loss)):
            print("Early stopping!")
            break

        # save the model
        model.eval()
        print("SAVING")
        
        # save full training state so can be resumed
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss_hist': training_losses,
            'val_loss_hist': val_losses,
            'early_stop_dict': early_stopper.__dict__,
            }, checkpoint_path)

        with open(checkpoint_folder + "/training_losses.txt","w") as tfileout:
            tfileout.write("\n".join("{}".format(tl) for tl in training_losses)+"\n")
        with open(checkpoint_folder + "/validation_losses.txt","w") as vfileout:
            vfileout.write("\n".join("{}".format(vl) for vl in val_losses)+"\n")


    print("Saving to %s" % checkpoint_folder, flush=True)
    torch.save(model.state_dict(), os.path.join(checkpoint_folder, 'final.pth'))

    with open(checkpoint_folder + "/training_losses.txt","w") as tfileout:
        tfileout.write("\n".join("{}".format(tl) for tl in training_losses)+"\n")
    with open(checkpoint_folder + "/validation_losses.txt","w") as vfileout:
        vfileout.write("\n".join("{}".format(vl) for vl in val_losses)+"\n")