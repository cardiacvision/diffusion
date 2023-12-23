#%%
import napari
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from tqdm import trange

parser = ArgumentParser()
parser.add_argument('path', type=Path, default=Path('.'), nargs='+')
parser.add_argument('--Nz', type=int, default=24)
parser.add_argument('--theme', type=str, default='dark') #light
parser.add_argument('--screenshot', action='store_true')
parser.add_argument('--skip', type=int, default=1)
parser.add_argument('--secondhalf', action='store_true')
parser.add_argument('--tstart', type=int, default=0)
parser.add_argument('--tend', type=int, default=-1)
args = parser.parse_args()
#%%
Nx = 64
Ny = 64
Nz = args.Nz

def dat_loader(path):
    data = np.memmap(path, dtype=np.float32, mode='r', order='C')
    Nt = data.shape[0] // ((Nx+2)*(Ny+2)*(Nz+2))
    data = data.reshape((Nt, Nz+2, Ny+2, Nx+2))
    data = data[:, 1:-1, 1:-1, 1:-1]
    import pdb; pdb.set_trace()
    return data

def npy_loader(path):
    data = np.load(path, mmap_mode='r')
    if len(data.shape) == 4:
        data = np.moveaxis(data, -1, 1)
    global Nz
    Nz = data.shape[1]
    return data

def loader(path):
    if path.suffix == '.dat':
        return dat_loader(path)
    elif path.suffix == '.npy':
        return npy_loader(path)
    else:
        raise ValueError(f'Unknown file type: {path}')

datas = []
for path in args.path:
    if path.is_dir():
        path = path / 'prefix_u.dat'

    data = loader(path)
    print(data.shape)

    data = data[args.tstart:args.tend:args.skip]
    # data = np.clip(data, 0.000001, 1)
    if args.secondhalf:
        data = data[:, Nz//2-1:, :, :] + 0.00001
    else:
        data = data + 0.0001

    datas.append((data, path.stem))

#%%
def add_lines(viewer, Nz, edge_width=0.25, name=None):
    if name is None:
        name = str(Nz)
    lines = [
        np.array([[0, 127.5, 127.5], [Nz-0.5, 127.5, 127.5]]),
        np.array([[Nz-0.5, 127.5, 127.5], [Nz-0.5, 0, 127.5]]),
        np.array([[0, 127.5, 127.5], [0, 0, 127.5]]),
        np.array([[Nz-0.5, 0, 127.5], [Nz-0.5, -0.5, -0.5]]),
        np.array([[0, 0, 127.5], [0, -0.5, -0.5]]),
        np.array([[0, -0.5, -0.5], [Nz-0.5, -0.5, -0.5]]),
        np.array([[0, 0, 127.5], [Nz-0.5, 0, 127.5]]),
        np.array([[0, -0.5, -0.5], [0, 127.5, -0.5]]),
        np.array([[0, 127.5, -0.5], [-0.5, 127.5, 127.5]]),
    ]
    if args.secondhalf:
        lines = [line - np.array([Nz//2-1, 0, 0]) for line in lines]

    for i, line in enumerate(lines):
        viewer.add_shapes(line, shape_type='line', edge_color='dimgray', edge_width=edge_width, opacity=1, name=f'{name}-line{i}')

def set_camera(viewer):
    viewer.camera.center = (30 - float(args.secondhalf) * (Nz//2 - 1), 80, 50)
    viewer.camera.zoom = 8.
    viewer.camera.angles = (-130, 44, 48)
    viewer.camera.perspective = 0.0

#%%
import pdb; pdb.set_trace()
viewer = napari.Viewer()
for data, name in datas:
    viewer.add_image(data, name=name, colormap='magma', gamma=1, attenuation=0.085, contrast_limits=(0, 1))
for layer in viewer.layers:
    layer.rendering = 'attenuated_mip'
add_lines(viewer, Nz, name='data')
set_camera(viewer)
viewer.theme = args.theme
if not args.screenshot:
    napari.run()
else:
    OUT_DIR = Path('napari')
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    names = [name for _, name in datas]
    xlim = min([x[0].shape[0] for x in datas])
    for t in trange(0, xlim):
        viewer.dims.set_current_step(axis=0, value=t)
        for name in names:
            for layer in viewer.layers:
                if layer.name in names and layer.name != name:
                    layer.visible = False
                else:
                    layer.visible = True
            folder = name.split('_u')[0]
            if args.secondhalf:
                folder += '_secondhalf'
            fn = OUT_DIR / folder / f"{folder}_{t+args.tstart:04d}.png"
            fn.parent.mkdir(parents=True, exist_ok=True)
            viewer.screenshot(fn, canvas_only=True, flash=False)
#%%