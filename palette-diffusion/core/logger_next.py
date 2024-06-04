import os
from PIL import Image
import importlib
from datetime import datetime
import logging
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

import core.util as Util

class InfoLogger():
    """
    use logging to record log, only work on GPU 0 by judging global_rank
    """
    def __init__(self, opt):
        self.opt = opt
        self.rank = opt['global_rank']
        self.phase = opt['phase']

        self.setup_logger(None, opt['path']['experiments_root'], opt['phase'], level=logging.INFO, screen=False)
        self.logger = logging.getLogger(opt['phase'])
        self.infologger_ftns = {'info', 'warning', 'debug'}

    def __getattr__(self, name):
        if self.rank != 0: # info only print on GPU 0.
            def wrapper(info, *args, **kwargs):
                pass
            return wrapper
        if name in self.infologger_ftns:
            print_info = getattr(self.logger, name, None)
            def wrapper(info, *args, **kwargs):
                print_info(info, *args, **kwargs)
            return wrapper
    
    @staticmethod
    def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False):
        """ set up logger """
        l = logging.getLogger(logger_name)
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
        log_file = os.path.join(root, '{}.log'.format(phase))
        fh = logging.FileHandler(log_file, mode='a+')
        fh.setFormatter(formatter)
        l.setLevel(level)
        l.addHandler(fh)
        if screen:
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            l.addHandler(sh)

class VisualWriter():
    """ 
    use tensorboard to record visuals, support 'add_scalar', 'add_scalars', 'add_image', 'add_images', etc. funtion.
    Also integrated with save results function.
    """
    def __init__(self, opt, logger):
        log_dir = opt['path']['tb_logger']
        self.result_dir = opt['path']['results']
        enabled = opt['train']['tensorboard']
        self.rank = opt['global_rank']

        self.writer = None
        self.selected_module = ""

        if enabled and self.rank==0:
            log_dir = str(log_dir)

            # Retrieve vizualization writer.
            succeeded = False
            for module in ["tensorboardX", "torch.utils.tensorboard"]:
                try:
                    self.writer = importlib.import_module(module).SummaryWriter(log_dir)
                    succeeded = True
                    break
                except ImportError:
                    succeeded = False
                self.selected_module = module

            if not succeeded:
                message = "Warning: visualization (Tensorboard) is configured to use, but currently not installed on " \
                    "this machine. Please install TensorboardX with 'pip install tensorboardx', upgrade PyTorch to " \
                    "version >= 1.1 to use 'torch.utils.tensorboard' or turn off the option in the 'config.json' file."
                logger.warning(message)

        self.epoch = 0
        self.iter = 0
        self.phase = ''

        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding'
        }
        self.tag_mode_exceptions = {'add_histogram', 'add_embedding'}
        self.custom_ftns = {'close'}
        self.timer = datetime.now()

    def set_iter(self, epoch, iter, phase='train'):
        self.phase = phase
        self.epoch = epoch
        self.iter = iter

    def save_images(self, results):
        result_path = os.path.join(self.result_dir, self.phase)
        os.makedirs(result_path, exist_ok=True)
        result_path = os.path.join(result_path, str(self.epoch))
        os.makedirs(result_path, exist_ok=True)

        ''' get names and corresponding images from results[OrderedDict] '''
        try:
            names = results['name']
            outputs = Util.postprocess(results['result'])
            # for i in range(len(names)): 
            #     Image.fromarray(outputs[i]).save(os.path.join(result_path, names[i]))
            gt = []
            mask = []
            out = []
            name_indicies = []
            cond = []
            for i in range(len(names)):
                if "GT" in names[i]:
                    gt.append((outputs[i]).astype(np.int16))
                    name_indicies.append(names[i][3:])
                elif "Out" in names[i]:
                    out.append((outputs[i]).astype(np.int16))
                elif "Mask" in names[i]:
                    mask.append(outputs[i].astype(np.int16))
                elif "Cond" in names[i]:
                    cond.append((outputs[i]).astype(np.int16))
            for i in range(len(gt)):
                # fig, (axu, axv) = plt.subplots(2, 4)
                # vmin = 0
                # vmax = 255
                # axu[0].imshow(gt[i][:, :, 0], vmin=vmin, vmax=vmax, interpolation="none")
                # axu[0].set_title("Ground Truth U")
                # axu[1].imshow(mask[i][:, :, 0], vmin=vmin, vmax=vmax, interpolation="none")
                # axu[1].set_title("Mask U")
                # axu[2].imshow(out[i][:, :, 0], vmin=vmin, vmax=vmax, interpolation="none")
                # axu[2].set_title("Output U")
                # axu[3].imshow(np.abs(out[i] - gt[i])[:, :, 0], vmin=vmin, vmax=vmax, interpolation="none")
                # axu[3].set_title("Difference U")

                # axv[0].imshow(gt[i][:, :, 1], vmin=vmin, vmax=vmax, interpolation="none")
                # axv[0].set_title("Ground Truth V")
                # axv[1].imshow(mask[i][:, :, 1], vmin=vmin, vmax=vmax, interpolation="none")
                # axv[1].set_title("Mask V")
                # axv[2].imshow(out[i][:, :, 1], vmin=vmin, vmax=vmax, interpolation="none")
                # axv[2].set_title("Output V")
                # axv[3].imshow(np.abs(out[i] - gt[i])[:, :, 1], vmin=vmin, vmax=vmax, interpolation="none")
                # axv[3].set_title("Difference V")

                fig, axu = plt.subplots(5, 4)
                for j in range(5):
                    vmin = 0
                    vmax = 255
                    axu[j][1].imshow(gt[i][:, :, j], vmin=vmin, vmax=vmax, interpolation="none", cmap=mpl.colormaps["magma"])
                    axu[j][1].set_title("Ground Truth U")
                    axu[j][1].set_axis_off()
                    axu[j][0].imshow(cond[i][:, :, j], vmin=vmin, vmax=vmax, interpolation="none", cmap=mpl.colormaps["magma"])
                    axu[j][0].set_title("Prev U")
                    axu[j][0].set_axis_off()
                    axu[j][2].imshow(out[i][:, :, j], vmin=vmin, vmax=vmax, interpolation="none", cmap=mpl.colormaps["magma"])
                    axu[j][2].set_title("Output U")
                    axu[j][2].set_axis_off()
                    axu[j][3].imshow(np.abs(out[i] - gt[i])[:, :, j], vmin=vmin, vmax=vmax, interpolation="none", cmap=mpl.colormaps["magma"])
                    axu[j][3].set_title("Difference U")
                    axu[j][3].set_axis_off()

                fig.savefig(os.path.join(result_path, f"All_{name_indicies[i]}.png"))
                plt.close(fig)
            try:
                arr = np.load(os.path.join(result_path, "results.npy"))
                new_arr = np.array(out[:len(gt)])
                arr = np.concatenate([arr, new_arr])

                old_gt = np.load(os.path.join(result_path, "gts.npy"))
                new_gt = np.array(gt)
                full_gt = np.concatenate([old_gt, new_gt])

                np.save(os.path.join(result_path, "gts.npy"), full_gt)
                np.save(os.path.join(result_path, "results.npy"), arr)
            except:
                arr = np.array(out[:len(gt)])
                np.save(os.path.join(result_path, "results.npy"), arr)

                new_gt = np.array(gt)
                np.save(os.path.join(result_path, "gts.npy"), new_gt)
                    
        except:
            raise NotImplementedError('You must specify the context of name and result in save_current_results functions of model.')

    def close(self):
        self.writer.close()
        print('Close the Tensorboard SummaryWriter.')

        
    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)
            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    # add phase(train/valid) tag
                    if name not in self.tag_mode_exceptions:
                        tag = '{}/{}'.format(self.phase, tag)
                    add_data(tag, data, self.iter, *args, **kwargs)
            return wrapper
        else:
            # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object '{}' has no attribute '{}'".format(self.selected_module, name))
            return attr


class LogTracker:
    """
    record training numerical indicators.
    """
    def __init__(self, *keys, phase='train'):
        self.phase = phase
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return {'{}/{}'.format(self.phase, k):v for k, v in dict(self._data.average).items()}
