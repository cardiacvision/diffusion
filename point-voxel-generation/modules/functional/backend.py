import os

from torch.utils.cpp_extension import load

os.environ["CC"] = "/usr/bin/gcc-11"
os.environ["CXX"] = "/usr/bin/g++-11"

_src_path = os.path.dirname(os.path.abspath(__file__))
_backend = load(name='_pvcnn_backend',
                extra_cflags=['-O3', '-std=c++17'],
                extra_cuda_cflags=['--compiler-bindir=/usr/bin/gcc-11'],
                sources=[os.path.join(_src_path,'src', f) for f in [
                    'ball_query/ball_query.cpp',
                    'ball_query/ball_query.cu',
                    'grouping/grouping.cpp',
                    'grouping/grouping.cu',
                    'interpolate/neighbor_interpolate.cpp',
                    'interpolate/neighbor_interpolate.cu',
                    'interpolate/trilinear_devox.cpp',
                    'interpolate/trilinear_devox.cu',
                    'sampling/sampling.cpp',
                    'sampling/sampling.cu',
                    'voxelization/vox.cpp',
                    'voxelization/vox.cu',
                    'bindings.cpp',
                ]], extra_ldflags=["-L /home/tanish/miniconda3/envs/pvd/lib/"]
                )

__all__ = ['_backend']
