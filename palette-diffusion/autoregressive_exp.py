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

    curr_dict = next(iterator)

    all_data = []
    # import ipdb; ipdb.set_trace()
    with torch.no_grad():
        lens = []
        for idx, curr_dict in enumerate(tqdm(phase_loader)):
            trajectory = []
            non_end_indices = list(range(len(curr_dict["cond_image"])))
            removed = []
            # import ipdb; ipdb.set_trace()
            for i in range(2000 // 5):
                model.set_input(curr_dict)
                model.output, model.visuals = model.netG.restoration(model.cond_image, sample_num=model.sample_num)
                trajectory.append(model.output.detach().cpu().numpy().clip(-1, 1) / 2 + 0.5)
                curr_dict["cond_image"] = model.output.detach()
                # curr_dict["cond_image"][curr_dict["cond_image"] < 0.001] = 0
                traj = model.output.detach().cpu().numpy().clip(-1, 1) / 2 + 0.5
                percent_under_thresh = (traj < 0.1).mean((2, 3))

                # minimum_retval = curr_dict["cond_image"][non_end_indices].max(2).values.max(2).values / 2 + 0.5
                for j in range(len(percent_under_thresh)):
                    if percent_under_thresh[j].min() < 0.05 and j in non_end_indices:
                        non_end_indices.remove(j)
                        removed.append(j)
                        lens.append((i+1)*5 + (percent_under_thresh[j] < 0.05).astype(int).argmax().item())
                print(f"removed {len(removed)} indices")
                if not non_end_indices:
                    break
            else:
                lens.extend([2500 for i in range(len(non_end_indices))])
            
            all_data.append(trajectory)
            

    np.save('./diffusion_exp_diffusion_evolved_trajs.npy', np.array(all_data, dtype=object), allow_pickle=True)
    np.save("./diffusion_exp_diffusion_evolved_lens.npy", lens)
    


    phase_writer.close()
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/colorization_mirflickr25k.json', help='JSON file for configuration')
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