import argparse
import re
import sys
import time
from pathlib import Path
from threading import Thread

import numpy as np
import pyvista as pv
from matplotlib import cm
from tqdm import tqdm


def _natural_sort_path_key(path: Path, _nsre=re.compile("([0-9]+)")):
    return [
        int(text) if text.isdigit() else text.lower() for text in _nsre.split(path.name)
    ]


def dir_path_validator(value):
    p = Path(value)
    if not p.exists() or not p.is_dir():
        msg = "{value} is not valid path to an existing directory".format(
            value=value)
        raise argparse.ArgumentTypeError(msg)
    return p.resolve()


def discrete_cmap(N, base_cmap=None):
    """
    Create an N-bin discrete colormap from the specified input map
    https://gist.github.com/jakevdp/91077b0cae40f8f8244a
    """
    base = cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def path_validator(value):
    p = Path(value)
    if not p.exists():
        raise argparse.ArgumentTypeError(f"path {value} does not exist")
    return p.resolve()


class ScreenExporter:
    def __init__(self, video_fps=20):
        self.make_pngs = False
        self.make_movie = False

        self.export_path = None
        self.max_t = None
        self.video_fps = video_fps
        self.path = None
        self.movie_path = None
        self.progress = None

    def start_movie(self, plotter):
        if not self.export_path.parent.exists():
            self.export_path.parent.mkdir(exist_ok=True, parents=True)

        self.make_movie = True
        self.movie_path = self.export_path.with_suffix(".mp4")
        plotter.open_movie(str(self.movie_path), framerate=self.video_fps)
        print(f"Video export '{self.movie_path.resolve()}' started ...")
        self.progress = tqdm(desc='Video export', unit='frames', total=self.max_t)

    def start_png_sequence(self):
        self.make_pngs = True
        if not self.export_path.exists():
            self.export_path.mkdir(exist_ok=True, parents=True)

    def stop(self, plotter=None):
        if self.make_movie:
            if plotter is None:
                raise RuntimeError("plotter is None")
            plotter.mwriter.close()
            self.make_movie = False
            print(f"Video export '{self.movie_path.absolute()}' completed")
        if self.make_pngs:
            if plotter is None:
                raise RuntimeError("plotter is None")
            self.make_pngs = False
            print(f"Screenshots completed")
        if self.progress is not None:
            self.progress.close()
            self.progress = None

    def tick(self, t, plotter, autostop):
        if self.make_pngs:
            self._screenshot(self.export_path / f"{t:04d}.png", plotter)
            if t == self.max_t - 1:
                self.stop(plotter)
        if self.make_movie:
            plotter.write_frame()
            self.progress.update()
            if t == self.max_t - 1:
                self.stop(plotter)
        if autostop and t == self.max_t - 1:
            sys.exit()

    def toggle_png_sequence(self):
        if self.make_pngs:
            self.stop()
            self.make_pngs = False
        else:
            self.start_png_sequence()

    def toggle_movie(self, plotter):
        if self.make_movie:
            self.stop()
            self.make_movie = False
        else:
            self.start_movie(plotter)

    def set_params(self, export_dir: Path, size: int):
        self.export_path = export_dir
        self.max_t = size

    def _screenshot(self, png_file: Path, plotter):
        plotter.screenshot(png_file, transparent_background=False)
        print(f"Saved screenshot to {png_file}")


def get_opacity_transfer_function(name, n_colors=256):
    half_n_colors = n_colors // 2
    d = np.zeros(n_colors, dtype=np.uint8)
    if name == "linear_c":
        d[half_n_colors:] = np.linspace(0, 255, n_colors // 2, dtype=np.uint8)
        d[:half_n_colors] = d[half_n_colors:][::-1]
    elif name == "geom_c":
        # take the second half of the colormap instead of every second one
        d = np.geomspace(1e-6, 255, n_colors, dtype=np.uint8)
        d[:half_n_colors] = d[half_n_colors:][::-1]
    return d


def wireframe_color(x, cmap, transfer_func, default_opacity=0.5):
    x = np.array(x)
    cmap = cm.get_cmap(cmap)
    default_color = list(cmap(0))
    default_color[3] = default_opacity

    rgba = cmap(x)
    transfer = pv.opacity_transfer_function(transfer_func, 256)
    idxs = np.array(np.clip(x, 0, 1) * 255, dtype=np.int64)
    opacity = transfer[idxs]
    for i, o in enumerate(opacity):
        if o < 0.8:
            rgba[i] = default_color
        else:
            rgba[i, 3] = o
    rgba = (rgba * 255).astype(np.uint8)
    return rgba


def orbit_on_path(
        plotter,
        path=None,
        focus=None,
        step=0.5,
        viewup=None,
        write_frames=False,
        threaded=False,
        progress_bar=False,
    ):
        """Orbit on the given path focusing on the focus point.

        Parameters
        ----------
        path : pyvista.PolyData
            Path of orbital points. The order in the points is the order of
            travel.

        focus : list(float) of length 3, optional
            The point of focus the camera.

        step : float, optional
            The timestep between flying to each camera position.

        viewup : list(float), optional
            The normal to the orbital plane.

        write_frames : bool, optional
            Assume a file is open and write a frame on each camera
            view during the orbit.

        threaded : bool, optional
            Run this as a background thread.  Generally used within a
            GUI (i.e. PyQt).

        progress_bar : bool, optional
            Show the progress bar when proceeding through the path.
            This can be helpful to show progress when generating
            movies with ``off_screen=True``.

        Examples
        --------
        Plot an orbit around the earth.  Save the gif as a temporary file.

        >>> import os
        >>> from tempfile import mkdtemp
        >>> import pyvista
        >>> from pyvista import examples
        >>> filename = os.path.join(mkdtemp(), 'orbit.gif')
        >>> plotter = pyvista.Plotter(window_size=[300, 300])
        >>> _ = plotter.add_mesh(examples.load_globe(), smooth_shading=True)
        >>> plotter.open_gif(filename)
        >>> viewup = [0, 0, 1]
        >>> orbit = plotter.generate_orbital_path(factor=2.0, n_points=24,
        ...                                       shift=0.0, viewup=viewup)
        >>> plotter.orbit_on_path(orbit, write_frames=True, viewup=viewup,
        ...                       step=0.02)

        See :ref:`orbiting_example` for a full example using this method.

        """
        if focus is None:
            focus = plotter.center
        if viewup is None:
            viewup = plotter._theme.camera['viewup']
        if path is None:
            path = plotter.generate_orbital_path(viewup=viewup)
        points = path.points

        # Make sure the whole scene is visible
        plotter.camera.thickness = path.length

        def orbit():
            """Define the internal thread for running the orbit."""
            if progress_bar:
                points_seq = tqdm(points, desc="Orbiting")
            else:
                points_seq = points

            for point in points_seq:
                tstart = time.time()  # include the render time in the step time
                plotter.set_position(point, render=False)
                plotter.set_focus(focus, render=False)
                plotter.set_viewup(viewup, render=False)
                plotter.renderer.ResetCameraClippingRange()
                plotter.render()
                if write_frames:
                    plotter.write_frame()
                sleep_time = step - (time.time() - tstart)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        if threaded:
            thread = Thread(target=orbit)
            thread.start()
        else:
            orbit()