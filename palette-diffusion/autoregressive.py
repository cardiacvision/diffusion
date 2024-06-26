import argparse
import os
import warnings
import torch
import torch.multiprocessing as mp

from core.logger import VisualWriter, InfoLogger
import core.praser as Praser
import core.util as Util
from data import define_dataloader
from models import create_model, define_network, define_loss, define_metric
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from models.network import DDIMNetwork
def main_worker(gpu, ngpus_per_node, opt):
    """  threads running on each GPU """
    if 'local_rank' not in opt:
        opt['local_rank'] = opt['global_rank'] = gpu

    '''set seed and and cuDNN environment '''
    torch.backends.cudnn.enabled = True
    warnings.warn('You have chosen to use cudnn for accleration. torch.backends.cudnn.enabled=True')
    Util.set_seed(opt['seed'])

    ''' set logger '''
    phase_logger = InfoLogger(opt)
    phase_writer = VisualWriter(opt, phase_logger)  
    phase_logger.info('Create the log file in directory {}.\n'.format(opt['path']['experiments_root']))

    '''set networks and dataset'''
    phase_loader, val_loader = define_dataloader(phase_logger, opt) # val_loader is None if phase is test.
    networks = [define_network(phase_logger, opt, item_opt) for item_opt in opt['model']['which_networks']]

    ''' set metrics, loss, optimizer and  schedulers '''
    metrics = [define_metric(phase_logger, item_opt) for item_opt in opt['model']['which_metrics']]
    losses = [define_loss(phase_logger, item_opt) for item_opt in opt['model']['which_losses']]

    model = create_model(
        opt = opt,
        networks = networks,
        phase_loader = phase_loader,
        val_loader = val_loader,
        losses = losses,
        metrics = metrics,
        logger = phase_logger,
        writer = phase_writer
    )

    phase_logger.info('Begin model {}.'.format(opt['phase']))

    model.netG.eval()
    model.netG = DDIMNetwork(model.netG.denoise_fn, model.netG.beta_schedule)
    model.netG.set_new_noise_schedule(phase="train")


    model.test_metrics.reset()
    iterator = iter(model.phase_loader)
    for i in range(0):
        next(iterator)
    curr_dict = next(iterator)
    starter = curr_dict["cond_image"]
    all_data = []
    gt_data = []
    with torch.no_grad():
        for i in tqdm(range(28)):
            model.set_input(curr_dict)
            model.output, model.visuals = model.netG.restoration(model.cond_image, sample_num=model.sample_num)

            all_data.append(model.output.detach().cpu().numpy()[0])
            gt_data.append(curr_dict["gt_image"].detach().cpu().numpy()[0])
            starter = model.output.detach()
            for i in range(5):
                curr_dict = next(iterator)
            curr_dict["cond_image"] = starter
    print(starter.shape)
    outputs = np.concatenate(all_data, axis=0)
    gts = np.concatenate(gt_data, axis=0)
    np.save("experiments_next_timestep/predicted_sim.npy", outputs)
    np.save("experiments_next_timestep/gt_sim.npy", gts)


    gts = np.load("experiments_next_timestep/gt_sim.npy")
    preds = np.load("experiments_next_timestep/predicted_sim.npy")
    # import ipdb;ipdb.set_trace()
    skip = 5
    fig, ax = plt.subplots(2, len(preds)//skip)
    for i in range(0, len(preds), skip):
        ax[0][i//skip].imshow(gts[i], cmap=mpl.colormaps["magma"])
        ax[1][i//skip].imshow(preds[i], cmap=mpl.colormaps["magma"])
        ax[0][i//skip].set_axis_off()
        ax[1][i//skip].set_axis_off()
        ax[0][i//skip].set_ylabel(f"{i}")

    fig.set_size_inches(100//skip, 2)
    # fig.tight_layout()
    fig.savefig("u_autoregressive_output.png")
    # fig.savefig("t_autoregressive_output.png")
    
    phase_writer.close()
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/colorization_mirflickr25k.json', help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train','test'], help='Run train or test', default='train')
    parser.add_argument('-b', '--batch', type=int, default=None, help='Batch size in every gpu')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-P', '--port', default='21012', type=str)

    ''' parser configs '''
    args = parser.parse_args()
    opt = Praser.parse(args)
    
    ''' cuda devices '''
    gpu_str = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    print('export CUDA_VISIBLE_DEVICES={}'.format(gpu_str))

    ''' use DistributedDataParallel(DDP) and multiprocessing for multi-gpu training'''
    # [Todo]: multi GPU on multi machine
    opt['world_size'] = 1 
    main_worker(0, 1, opt)