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
from glob import glob
from models.network import DDIMNetwork
from torch.utils.data import DataLoader

from data.dataset import NextTimeStep2DNPY
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
    mode = "simple"
    avg_rmse = np.zeros(2495)
    for file in glob(f"../data/next-timestep-eval/{mode}*"):
        phase_loader = DataLoader(NextTimeStep2DNPY(data_path=f"{file}/u_data.npy", image_size=64), **opt['datasets'][opt['phase']]['dataloader']['args'])
        # iterator = iter(phase_loader)
        # curr_dict = next(iterator)
        # starter = curr_dict["cond_image"]
        all_data = []
        gt_data = []
        prev_output = None
        with torch.no_grad():
            for i, curr_dict in enumerate(tqdm(phase_loader)):
                if i % 5 != 0:
                    continue
                if prev_output is not None:
                    curr_dict["cond_image"] = prev_output
                model.set_input(curr_dict)
                model.output, model.visuals = model.netG.restoration(model.cond_image, sample_num=model.sample_num)

                all_data.append(model.output.detach().cpu().numpy()[0])
                gt_data.append(curr_dict["gt_image"].detach().cpu().numpy()[0])
                prev_output = model.output.detach()
        
        outputs = np.concatenate(all_data, axis=0)
        gts = np.concatenate(gt_data, axis=0)

        skip = 30

        fig, ax = plt.subplots(3, len(outputs)//skip)
        for i in range(0, len(outputs) - skip, skip):
            ax[0][i//skip].imshow(gts[i], cmap=mpl.colormaps["magma"])
            ax[1][i//skip].imshow(outputs[i], cmap=mpl.colormaps["magma"])
            ax[2][i//skip].imshow(np.abs(gts[i] - outputs[i]), cmap=mpl.colormaps["magma"])
            
            ax[0][i//skip].set_axis_off()
            ax[1][i//skip].set_axis_off()
            ax[2][i//skip].set_axis_off()

        fig.set_size_inches(len(outputs)//skip, 3)
        # fig.tight_layout()
        fig.savefig(f"u-autoreg/u_test_autoreg_{mode[0]}{file[-1]}.png")
        np.save(f"u-autoreg/u_autoreg_data_{mode[0]}{file[-1]}_output.npy", outputs)
        np.save(f"u-autoreg/u_autoreg_data_{mode[0]}{file[-1]}_gts.npy", gts)

        rmse = ((gts - outputs) ** 2).sum(axis=(1, 2)) ** 0.5
        avg_rmse += rmse
    # import ipdb;ipdb.set_trace()
    
    fig, ax = plt.subplots()
    ax.plot(range(len(rmse)), rmse/5/(64**2))
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Test RMSE Score")
    ax.set_title(f"{mode} Data")
    # fig.set_size_inches(100//skip, 2)
    # fig.tight_layout()
    fig.savefig(f"u-autoreg/u_{mode}_next_timestep_rmse.png")
    np.save(f"u-autoreg/u_{mode}_rmse.npy", rmse/5/(64**2))
    # fig.savefig("t_autoregressive_output.png")
    
    phase_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/next_timestep.json', help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train','test'], help='Run train or test', default='test')
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