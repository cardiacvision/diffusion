import argparse
import signal
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
from matplotlib import cm
from matplotlib.colors import ListedColormap
from PyQt5 import Qt, QtGui, QtWidgets
from pyvistaqt import QtInteractor

from .utils import (
    ScreenExporter,
    _natural_sort_path_key,
    get_opacity_transfer_function,
    orbit_on_path,
    path_validator,
    wireframe_color,
    discrete_cmap,
)
from .visualize_ui import Ui_MainWindow

import pyacvd
def remesh(mesh, nsub=3, nclus=20000):
    clus = pyacvd.Clustering(mesh)
    # mesh is not dense enough for uniform remeshing
    clus.subdivide(nsub)
    clus.cluster(nclus)

    remesh = clus.create_mesh()
    return remesh

pv.set_plot_theme("document")
pv.global_theme.multi_samples = 8
EXPORT_BASE_DIR = Path("./sphviz")
MUTED_GREY_CMAP = ListedColormap(
    cm.get_cmap("Greys")(np.linspace(0, 0.7, 255)), "MutedGreys"
)
cm.register_cmap(cmap=MUTED_GREY_CMAP)

CAMERA_POSITIONS = {
    "front": [
        (-60, 30, -200),  # camera position
        (0, 1, 0),  # up direction
    ],
    "front_side": [
        (10, -15, -200),  # camera position
        (0.03, 0.99, -0.14),  # up direction
    ],
    "back": [
        (-60, 30, 200),  # camera position
        (0, 1, 0),  # up direction
    ],
    "back_side": [
        (10, -15, 200),  # camera position
        (0, 1, 0),  # up direction
    ],
    "top": [
        (0, 200, 0),  # camera position
        (0, 0, 1),  # up direction
    ],
}

DEFAULT_CMAPS = {
    "None": "None",
    "Activation": "coolwarm",
    "APD": "coolwarm",
    "Voltage": "MutedGreys",
    "ElectricalHeterogeneity": "PiYG",
    "Conductivity": "PiYG",
    "PredictedElectricalHeterogeneity": "PiYG",
    "Fiber": "twilight",
    "Sheet": "twilight",
    "Scarred": "Reds",
    "PredictedVoltage": "MutedGreys",
    "PredictedVoltageDifference": "PiYG",
}
CMAP_MAPPINGS = {
    "None": None,
    "Greys": "Greys",
    "MutedGreys": "MutedGreys",
    "viridis": "viridis",
    "inferno": "inferno",
    "PiYG": "PiYG",
    "PRGn": "PRGn",
    "twilight": "twilight",
    "phase": "phase",
    "Reds": "Reds",
    "coolwarm": "coolwarm",
    "coolwarm_d": discrete_cmap(8, "coolwarm"),
}
DEFAULT_RANGE = {
    "None": None,
    "Activation": None,
    "APD": None,
    "Voltage": None,
    "PredictedVoltage": (0, 1),
    "PredictedVoltageDifference": (-1, 1),
    "Conductivity": (0, 2),
    "ElectricalHeterogeneity": (0, 2),
    "PredictedElectricalHeterogeneity": (0, 2),
}

# Opacity of the mesh. If a single float value is given, it will be the global opacity of the mesh and uniformly
# applied everywhere - should be between 0 and 1. A string can also be specified to map the scalars range to a
# predefined opacity transfer function (options include: 'linear', 'linear_r', 'geom', 'geom_r').
OPACITY_MAPPINGS = {
    "None": None,
    "0.5": 0.5,
    "0.8": 0.8,
    "linear": "linear",
    "linear_r": "linear_r",
    "linear_c": get_opacity_transfer_function("linear_c"),
    "geom": "geom",
    "geom_r": "geom_r",
    "geom_c": get_opacity_transfer_function("geom_c"),
    "sigmoid": "sigmoid",
    "sigmoid_3": "sigmoid_3",
    "sigmoid_6": "sigmoid_6",
}


class MainWindow(Qt.QMainWindow):
    point_size = 15
    stimuli_radius = 3
    frame_delay_msec = 100
    default_input_scalar1 = "Voltage"
    default_input_scalar2 = "PredictedVoltage"
    GLYPH_ELEM_SKIP = 32
    GLYPH_SCALE_FACTOR = 16
    GLYPH_COLOR_SCALAR = "Magnitude"  # "Magnitude", "XY", "XZ", or "YZ"
    STIMULI_VISIBLE_FRAMES = 30  # number of frames the stimuli will be visible

    def __init__(
        self,
        path1=None,
        path2=None,
        scalar=None,
        camera_position="front",
        show_glyph=False,
        show_wireframe=True,
        make_pngs=False,
        make_video=False,
        offscreen=False,
        offscreen_width=2048,
        offscreen_height=1024,
        less_lights=False,
    ):
        super(MainWindow, self).__init__()
        self.setupUi(off_screen=offscreen)
        self.t = 0
        self.autostop = make_pngs | make_video
        self.timer = Qt.QTimer()
        self.timer.timeout.connect(self.load_next_frame)
        if not offscreen:
            self.timer.setInterval(self.frame_delay_msec)
        else:
            self.timer.setInterval(1)
        self.show_tlabel = True
        self.tlabel = None

        self.files1 = None
        self.files2 = None
        self.point_cloud1 = None
        self.point_cloud2 = None
        self.actor_pcls = []
        self.opacity1 = "None"
        self.opacity2 = "None"
        self.cmap1 = None
        self.cmap2 = None
        self.active_scalar1 = self.default_input_scalar1
        if scalar is not None:
            self.active_scalar2 = scalar
        else:
            self.active_scalar2 = self.default_input_scalar2
        self.active_vector_field = None

        self.available_input_scalars1 = []
        self.available_input_scalars2 = []
        self.available_input_vector_fields = []

        self.show_glyph = show_glyph
        self.hide_glyph_sliders = show_glyph
        self.actor_glyph = None
        self.glyph_data = None
        self.hide_bar = offscreen

        self.screen_exporter = ScreenExporter()

        self.show_stimuli = False
        self.stimuli_actors = []

        self.clipping = False
        self.clip_show_plane = False
        self.clip_normal = (0.0, 0.0, 1.0)
        self.clip_origin = None

        if show_wireframe:
            self.show_wireframe = True
            self.opacity1 = "geom"
            self.opacity2 = "geom"
            DEFAULT_CMAPS["Voltage"] = "Greys"
            DEFAULT_CMAPS["PredictedVoltage"] = "Greys"
        else:
            self.show_wireframe = False

        if not less_lights:
            # Add more lights
            # [print(l) for l in self.plotter.renderer.lights]
            diffuse_color = (0.9998, 0.9998, 0.9998)
            specular_color = (0.9998, 0.9998, 0.9998)
            light = pv.Light(
                light_type="scene light", position=(0, -100, -100), intensity=0.214286
            )
            light.diffuse_color = diffuse_color
            light.specular_color = specular_color
            self.plotter.add_light(light)
            light = light.copy()
            light.position = (0, -100, +100)
            self.plotter.add_light(light)

        # Set camera position from name or the string representation of the camera position
        cam_pos = CAMERA_POSITIONS.get(camera_position, None)
        if cam_pos is None:
            cam_pos = eval(camera_position.strip("\n"))
        if type(cam_pos) is str:
            self.plotter.renderer.camera_position = cam_pos
        else:
            self.plotter.camera.position = cam_pos[0]
            self.plotter.camera.up = cam_pos[1]

        if path1:
            self.load_dirs(path1, path2)
        if show_glyph:
            self.show_glyph = False
            self.toggle_glyph("")
        if make_pngs:
            self.toggle_png_sequence()
        if make_video:
            self.toggle_video_recording()

        if not offscreen:
            self.plotter.subplot(0, 0)
            if self.show_tlabel:
                self.tlabel = self.plotter.add_text("", position=(0.05, 0.047), font_size=10, viewport=True)
            self.plotter.subplot(0, 1)
            self.plotter.add_checkbox_button_widget(self.toggle_glyph, size=25)
            self.show()
        else:
            self.plotter.window_size = offscreen_width, offscreen_height

    def setupUi(self, off_screen):
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # add the pyvista interactor object
        # noinspection PyTypeChecker
        self.plotter = QtInteractor(
            self.ui.frame,
            shape=(1, 2),
            lighting="light kit",
            auto_update=False,
            off_screen=off_screen,
        )  # "none"
        # Force same view for both subplots
        self.plotter.renderers[1].camera = self.plotter.renderers[0].camera
        # self.plotter.renderer.enable_eye_dome_lighting()
        # camerastyle = vtk.vtkInteractorStyleTrackballCamera()
        # self.plotter.iren.SetInteractorStyle(camerastyle) #https://vtk.org/Wiki/VTK/Examples/Python/InteractorStyleTrackballCamera
        self.ui.centralwidget.layout().addWidget(self.plotter.interactor)

        self.ui.left_cmap.addItems(CMAP_MAPPINGS.keys())
        self.ui.right_cmap.addItems(CMAP_MAPPINGS.keys())
        self.ui.left_opacity.addItems(OPACITY_MAPPINGS.keys())
        self.ui.right_opacity.addItems(OPACITY_MAPPINGS.keys())

        self.ui.actionLoad_Directory.triggered.connect(self.load_dir_dialog)
        self.ui.actionLoad_Second_Directory.triggered.connect(
            self.load_dir2_dialog)
        self.ui.actionPrint_Camera_Position.triggered.connect(
            self.print_camera_position
        )

        self.ui.playpause.clicked.connect(self.toggle_playpause)
        shorcut = QtWidgets.QShortcut("space", self.ui.playpause)
        shorcut.activated.connect(self.toggle_playpause)
        self.ui.actionExit.triggered.connect(self.close)
        self.ui.framenr.setTickInterval(0)
        self.ui.framenr.valueChanged.connect(self.set_t)
        shortcut = QtWidgets.QShortcut(QtGui.QKeySequence.MoveToNextChar, self)
        shortcut.activated.connect(self.load_next_frame)
        shortcut = QtWidgets.QShortcut(QtGui.QKeySequence.MoveToPreviousChar, self)
        shortcut.activated.connect(self.load_prev_frame)

        self.ui.framedelay.setValue(int(1000 / self.frame_delay_msec))
        self.ui.framedelay.setMinimum(1)
        self.ui.framedelay.setMaximum(60)
        self.ui.framedelay.setTickInterval(1)
        self.ui.framedelay.valueChanged.connect(self.set_frame_delay)

        self.ui.pointsize.setValue(self.point_size)
        self.ui.pointsize.setMinimum(1)
        self.ui.pointsize.setMaximum(40)
        self.ui.pointsize.valueChanged.connect(self.set_point_size)

        self.ui.left_scalar.activated.connect(self.set_active_scalar)
        self.ui.right_scalar.activated.connect(self.set_active_scalar)
        self.ui.left_opacity.activated.connect(self.set_opacity)
        self.ui.right_opacity.activated.connect(self.set_opacity)
        self.ui.left_cmap.activated.connect(self.set_cmap)
        self.ui.right_cmap.activated.connect(self.set_cmap)

        self.ui.actionToggle_Scale_Bar.triggered.connect(self.toggle_scale_bar)
        self.ui.actionToggle_Stimuli_Points.triggered.connect(self.toggle_stimuli)
        self.ui.actionToggle_Clipping.triggered.connect(self.toggle_clipping)
        self.ui.actionToggle_Plane.triggered.connect(self.toggle_clipping_plane)

        self.ui.actionToggle_PNG_Screenshots.triggered.connect(self.toggle_png_sequence)
        self.ui.actionCreate_Movie.triggered.connect(self.toggle_video_recording)

        self.ui.actionToggle_Glyphs.triggered.connect(self.toggle_glyph)
        self.ui.actionGlyphMagnitude.triggered.connect(self.set_glyph_colorscalar)
        self.ui.actionGlyphXY.triggered.connect(self.set_glyph_colorscalar)
        self.ui.actionGlyphXZ.triggered.connect(self.set_glyph_colorscalar)
        self.ui.actionGlyphYZ.triggered.connect(self.set_glyph_colorscalar)

        self.ui.actionOrbit.triggered.connect(lambda x: self.orbit(x, False))
        self.ui.actionOrbit_Video.triggered.connect(lambda x: self.orbit(x, True))

    def locate_files(self, path: Path):
        if path.is_file() and path.suffix == ".npz":
            files = [path]
        else:
            files = sorted(path.glob("*.npz"), key=_natural_sort_path_key)
        if not files:
            msg = Qt.QMessageBox()
            msg.setIcon(Qt.QMessageBox.Critical)
            msg.setText("Unable to find any suitable files")
            msg.setWindowTitle("Error")
            msg.exec_()
        return files

    # noinspection PyTypeChecker
    def load_dirs(self, path: Path, path2: Path = None):
        def detect_shape(data: np.lib.npyio.NpzFile):
            scalars = ["None"]
            fields = []
            keys = data.files[1:]
            for key in keys:
                shape = data[key].shape
                if len(shape) == 2:
                    if shape[1] == 1:
                        scalars.append(key)
                    elif shape[1] == 3:
                        fields.append(key)
            return scalars, fields

        print(f"Loading {path} ...")
        self.files1 = self.locate_files(path)
        d = np.load(self.files1[0], mmap_mode="r")
        self.point_cloud1 = pv.PolyData(d["XYZ"])
        # noinspection PyUnresolvedReferences
        scalars1, self.available_input_vector_fields = detect_shape(d)
        self.available_input_scalars1 = scalars1

        # focus camera on center of point cloud
        self.plotter.camera.focal_point = self.point_cloud1.center

        self.point_cloud2 = None
        if path2:
            self.files2 = self.locate_files(path2)
            assert len(self.files1) == len(self.files2)
            d2 = np.load(self.files2[0], mmap_mode="r")
            self.point_cloud2 = pv.PolyData(d2["XYZ"])
            # noinspection PyUnresolvedReferences
            self.available_input_scalars2 = detect_shape(d2)[0]
            self.active_scalar2 = self.active_scalar1

            self.ui.leftBox.setTitle(f"Left: {path.name}")
            self.ui.rightBox.setTitle(f"Right: {path2.name}")
        else:
            self.available_input_scalars2 = (
                scalars1 + self.available_input_vector_fields
            )
            self.point_cloud2 = pv.PolyData(self.point_cloud1.points)

        if self.active_scalar1 not in self.available_input_scalars1:
            self.active_scalar1 = self.available_input_scalars1[0]
        if self.active_scalar2 not in self.available_input_scalars2:
            self.active_scalar2 = self.available_input_scalars2[0]
        self.ui.left_scalar.clear()
        self.ui.right_scalar.clear()
        self.ui.left_scalar.addItems(self.available_input_scalars1)
        self.ui.right_scalar.addItems(self.available_input_scalars2)
        self.ui.left_scalar.setCurrentIndex(
            self.available_input_scalars1.index(self.active_scalar1)
        )
        self.ui.right_scalar.setCurrentIndex(
            self.available_input_scalars2.index(self.active_scalar2)
        )

        self.cmap1 = DEFAULT_CMAPS.get(
            self.active_scalar1, DEFAULT_CMAPS["Voltage"])
        self.cmap2 = DEFAULT_CMAPS.get(self.active_scalar2, None)
        self.ui.left_cmap.setCurrentIndex(
            self.ui.left_cmap.findText(self.cmap1))
        self.ui.right_cmap.setCurrentIndex(
            self.ui.right_cmap.findText(self.cmap2))

        self.ui.framenr.setMaximum(len(self.files1))

        if self.t >= len(self.files1):
            self.t = 0
        self.load(self.t, force_mesh_reload=True)
        self.screen_exporter.set_params(
            EXPORT_BASE_DIR / self.files1[0].parent.name, len(self.files1)
        )

        self.timer.start()

    def load(self, t, force_mesh_reload=False):
        """The main update function. Load the new data & update canvas.

        This function is a mess (sorry) and is in need of refactoring.
        """

        if (t < 0) or (t >= len(self.files1)):
            t = 0
        
        # print(f"Loading frame {t} ... {self.files1[t]}")
        data1 = np.load(self.files1[t], mmap_mode="r")
        if self.files2:
            data2 = np.load(self.files2[t], mmap_mode="r")
        else:
            data2 = data1

        # this is such ugly code, sorry ...
        def get_scalar(idx, scalar, transform=True):
            d = None
            if idx == 2 and not self.files2:
                idx = 1
            if idx == 1:
                # static fields are only saved in the first file in the directory
                if scalar not in data1:
                    d = np.load(self.files1[0], mmap_mode="r")[scalar]
                else:
                    d = data1[scalar]
            elif idx == 2:
                if scalar not in data2:
                    d = np.load(self.files2[0], mmap_mode="r")[scalar]
                else:
                    d = data2[scalar]
            if transform and scalar == "Voltage":
                return 100 * d - 80
            else:
                return d

        # Create our two point cloud objects
        pc1 = pv.PolyData(data1["XYZ"])
        pc2 = pv.PolyData(data2["XYZ"])
        if self.active_scalar1 != "None":
            pc1.point_data[self.active_scalar1] = get_scalar(
                1, self.active_scalar1, False
            )
        if self.active_scalar2 != "None":
            pc2.point_data[self.active_scalar2] = get_scalar(
                2, self.active_scalar2, False
            )

        # If enabled, clip the first one
        if self.clip_origin is None:
            self.clip_origin = pc1.center
        if self.clipping:
            pc1 = pc1.clip(self.clip_normal, self.clip_origin, invert=False)

        # Either update displayed render objects or create new ones.
        # We could always call self.reload_mesh(), but it can be very slow.
        # Check if the point cloud has the the same number of points, then we can
        # just update the render object data directly.
        if (
            force_mesh_reload
            or pc1.points.shape != self.point_cloud1.points.shape
            or pc2.points.shape != self.point_cloud2.points.shape
        ):
            self.point_cloud1 = pc1
            self.point_cloud2 = pc2
            self.reload_mesh(reload_glyph=False)
        else:
            self.point_cloud1.points = pc1.points
            self.point_cloud2.points = pc2.points
            if self.active_scalar1 != "None":
                self.point_cloud1.point_data[self.active_scalar1] = pc1.point_data[
                    self.active_scalar1
                ]
            if self.active_scalar2 != "None":
                self.point_cloud2.point_data[self.active_scalar2] = pc2.point_data[
                    self.active_scalar2
                ]

        # Add/remove stimuli points
        self.tick_stimuli(t, data1)

        if self.show_glyph:
            if t + 1 < len(self.files1) and t > 0:
                if self.active_vector_field is None:
                    if "DXYZ" in data1:
                        vectors = data1["DXYZ"][0]
                    else:
                        points_t1 = np.load(
                            self.files1[t + 1], mmap_mode="r")["XYZ"]
                        vectors = points_t1 - data1["XYZ"]
                else:
                    vectors = get_scalar(1, self.active_vector_field, False)

                # uniform sampling for more organic distribution of points
                size = data1["XYZ"].shape[0]
                idx = np.random.randint(0, size, size // self.GLYPH_ELEM_SKIP)
                positions = data1["XYZ"][idx]
                vectors = vectors[idx]

                pc = pv.PolyData(positions)
                pc["vectors"] = vectors
                pc.active_vectors_name = "vectors"
                pc.point_data["vector_scales"] = np.linalg.norm(
                    vectors, axis=1)

                # set glyph coloring
                if self.GLYPH_COLOR_SCALAR == "Magnitude":
                    v = pc.point_data["vector_scales"]
                    pc.point_data["colors"] = (v - min(v)) / np.ptp(v)
                else:
                    # scale [-Pi, Pi] to [0, 1]
                    def scale_pi(x):
                        return x / np.pi * 0.5 + 0.5

                    if self.GLYPH_COLOR_SCALAR == "XY":
                        pc.point_data["colors"] = scale_pi(
                            np.arctan2(vectors[:, 0], vectors[:, 1])
                        )
                    elif self.GLYPH_COLOR_SCALAR == "XZ":
                        pc.point_data["colors"] = scale_pi(
                            np.arctan2(vectors[:, 0], vectors[:, 2])
                        )
                    elif self.GLYPH_COLOR_SCALAR == "YZ":
                        pc.point_data["colors"] = scale_pi(
                            np.arctan2(vectors[:, 1], vectors[:, 2])
                        )

                self.glyph_data = pc.glyph(
                    scale="vector_scales", factor=self.GLYPH_SCALE_FACTOR
                )
                self.plotter.subplot(0, 1)
                if self.actor_glyph is not None:
                    self.plotter.remove_actor(
                        self.actor_glyph, reset_camera=False, render=False
                    )
                self.actor_glyph = self.plotter.add_mesh(
                    self.glyph_data,
                    scalars="colors",
                    cmap=CMAP_MAPPINGS.get(self.cmap2, None),
                    # WARNING: data has to be in [0, 1], because we call SetInputData()
                    clim=(0, 1),
                    show_scalar_bar=not self.hide_bar,
                    scalar_bar_args=dict(
                        fmt="%.1f",
                        title=self.GLYPH_COLOR_SCALAR,
                        # interactive=True
                    ),
                    render=False,
                    reset_camera=False,
                )
                # self.actor_glyph.GetMapper().SetInputData(self.glyph_data)
        else:
            self.plotter.remove_actor(
                self.actor_glyph, reset_camera=False, render=False
            )
            self.actor_glyph = None

        if self.tlabel is not None and 'physical_time' in data1:
            self.tlabel.SetInput(f"{(data1['physical_time'][0] * 12.9) / 1000.0:.2f} s")

        self.ui.framenr.blockSignals(True)
        self.ui.framenr.setValue(t)
        self.ui.framenr.blockSignals(False)

        self.plotter.render()

    def load_next_frame(self):
        last_t = self.t
        self.t += 1
        if (self.t < 0) or (self.t >= len(self.files1)):
            self.t = 0
        if self.t != last_t:
            self.load(self.t, force_mesh_reload=self.show_wireframe)
            self.screen_exporter.tick(self.t, self.plotter, self.autostop)

    def load_prev_frame(self):
        self.t -= 1
        if (self.t < 0) or (self.t >= len(self.files1)):
            self.t = 0
        self.load(self.t)

    def reload_mesh(self, reload_glyph=False):
        if self.actor_glyph:
            self.plotter.remove_actor(self.actor_glyph, render=False)
            self.actor_glyph = None
        for actor in self.actor_pcls:
            self.plotter.remove_actor(actor, render=False)
        self.actor_pcls = []

        def add_pointcloud(pcl, scalar, cmap, default_cmap, opacity):
            default_color = (0.95, 0.95, 0.95)
            return self.plotter.add_mesh(
                pcl,
                render_points_as_spheres=True,
                point_size=self.point_size,
                lighting=True,
                opacity=OPACITY_MAPPINGS.get(opacity, None),
                color=default_color if scalar == "None" else None,
                cmap=CMAP_MAPPINGS.get(cmap, default_cmap),
                clim=DEFAULT_RANGE.get(scalar, (0, 0)),
                show_scalar_bar=not self.hide_bar,
                scalar_bar_args=dict(
                    fmt="%.1f",
                    color="black",
                    # interactive=True
                ),
                render=False,
                reset_camera=False,
            )

        def add_wireframe(pcl, scalar, cmap, opacity):
            # volume = pcl.delaunay_3d(alpha=1)
            volume = pcl.delaunay_3d(alpha=2)
            mesh = volume.extract_surface()
            # print(self.t)
            # mesh = self.point_cloud1.reconstruct_surface(nbr_sz=10, sample_spacing=1, progress_bar=True)
            colors = wireframe_color(
                mesh[scalar],
                CMAP_MAPPINGS.get(cmap, MUTED_GREY_CMAP),
                OPACITY_MAPPINGS[opacity],
            )
            ## mesh = mesh.clean(tolerance=1e-6)
            # mesh = remesh(mesh)
            return self.plotter.add_mesh(
                mesh,
                style="wireframe",
                scalars=colors,
                rgb=True,
                render=False,
                reset_camera=False,
                # scalars=scalar,
                # cmap=cmap,
                # edge_color='red',
                # show_edges=False,
                ## smooth_shading=True
                show_scalar_bar=False,
            )

        self.plotter.subplot(0, 0)
        self.actor_pcls.append(
            add_pointcloud(
                pcl=self.point_cloud1,
                scalar=self.active_scalar1,
                cmap=self.cmap1,
                default_cmap=MUTED_GREY_CMAP,
                opacity=self.opacity1,
            )
        )
        if self.opacity1 != "None" and self.show_wireframe:
            self.actor_pcls.append(
                add_wireframe(
                    pcl=self.point_cloud1,
                    scalar=self.active_scalar1,
                    cmap=self.cmap1,
                    opacity=self.opacity1,
                )
            )

        # self.plotter.scalar_bar.SetAnnotationLeaderPadding(16)
        # self.plotter.scalar_bar.SetTitle("Membrane Potential [mV]")

        self.plotter.subplot(0, 1)
        if not self.show_glyph:
            self.actor_pcls.append(
                add_pointcloud(
                    pcl=self.point_cloud2,
                    scalar=self.active_scalar2,
                    cmap=self.cmap2,
                    default_cmap=None,
                    opacity=self.opacity2,
                )
            )
            if self.opacity2 != "None" and self.show_wireframe:
                self.actor_pcls.append(
                    add_wireframe(
                        pcl=self.point_cloud2,
                        scalar=self.active_scalar2,
                        cmap=self.cmap2,
                        opacity=self.opacity2,
                    )
                )
        else:
            if reload_glyph:
                self.load(self.t)

    def set_t(self):
        self.t = self.ui.framenr.value()
        self.load(self.t)

    def set_frame_delay(self, val):
        self.frame_delay_msec = 1000.0 / val
        self.timer.setInterval(int(self.frame_delay_msec))

    def set_point_size(self, val):
        self.point_size = val
        for actor in self.actor_pcls:
            actor.GetProperty().SetPointSize(val)

    def set_active_scalar(self, _):
        self._remove_scale_bars()

        self.active_scalar1 = self.ui.left_scalar.currentText()
        self.active_scalar2 = self.ui.right_scalar.currentText()
        if self.active_scalar2 in self.available_input_vector_fields:
            self.active_vector_field = self.active_scalar2
            if not self.show_glyph:
                self.toggle_glyph("", reload=False)
        else:
            self.active_vector_field = None

        self.cmap1 = DEFAULT_CMAPS.get(
            self.active_scalar1, DEFAULT_CMAPS["Voltage"])
        self.cmap2 = DEFAULT_CMAPS.get(
            self.active_scalar2, DEFAULT_CMAPS["Voltage"])
        self.ui.left_cmap.setCurrentIndex(
            self.ui.left_cmap.findText(self.cmap1))
        self.ui.right_cmap.setCurrentIndex(
            self.ui.right_cmap.findText(self.cmap2))

        self.point_cloud1 = pv.PolyData(self.point_cloud1.points)
        self.point_cloud2 = pv.PolyData(self.point_cloud2.points)
        self.load(self.t, force_mesh_reload=True)

    def set_opacity(self, _):
        self.opacity1 = self.ui.left_opacity.currentText()
        self.opacity2 = self.ui.right_opacity.currentText()
        self.reload_mesh()

    def set_cmap(self, _):
        self.cmap1 = self.ui.left_cmap.currentText()
        self.cmap2 = self.ui.right_cmap.currentText()
        self.reload_mesh(reload_glyph=False)

    def set_glyph_scale(self, val):
        self.GLYPH_SCALE_FACTOR = val
        self.reload_mesh(reload_glyph=True)

    def set_glyph_skip(self, val):
        self.GLYPH_ELEM_SKIP = int(val)
        self.reload_mesh(reload_glyph=True)

    def toggle_playpause(self):
        if self.timer.isActive():
            self.timer.stop()
            self.ui.playpause.setText("Play")
        elif self.actor_pcls:
            self.timer.setInterval(int(self.frame_delay_msec))
            self.timer.start()
            self.ui.playpause.setText("Pause")

    def toggle_video_recording(self):
        self.screen_exporter.toggle_movie(self.plotter)
        if self.screen_exporter.make_movie:
            self.t = -1

    def toggle_png_sequence(self):
        self.screen_exporter.toggle_png_sequence()
        if self.screen_exporter.make_pngs:
            self.t = -1

    def load_dir_dialog(self):
        self.timer.stop()
        name = Qt.QFileDialog.getExistingDirectory(self, "Select Directory")
        if name:
            self.t = 0
            self.load_dirs(Path(name))
        elif self.actor_pcls:
            self.timer.start()

    def load_dir2_dialog(self):
        if not self.files1:
            msg = Qt.QMessageBox()
            msg.setIcon(Qt.QMessageBox.Critical)
            msg.setText("First load a directory")
            msg.setWindowTitle("Error")
            msg.exec_()
        self.timer.stop()
        name = Qt.QFileDialog.getExistingDirectory(self, "Select Directory")
        if name:
            self.t = 0
            self.load_dirs(self.files1[0].parent, Path(name))
        elif self.actor_pcls:
            self.timer.start()

    def print_camera_position(self):
        self.plotter.subplot(0, 0)
        self.plotter.add_axes()
        self.plotter.add_bounding_box()
        self.plotter.show_grid()
        self.plotter.add_points(
            np.zeros((1, 3)),
            render_points_as_spheres=True,
            point_size=10.0,
            color="red",
        )
        print([self.plotter.camera.position, self.plotter.camera.up])

    def toggle_scale_bar(self):
        self.hide_bar = not self.hide_bar
        if self.hide_bar:
            self._remove_scale_bars()
        self.reload_mesh()

    def _remove_scale_bars(self):
        # remove scale bars from both plots
        self.plotter.subplot(0, 0)
        for bar in list(self.plotter.scalar_bars.keys()):
            self.plotter.remove_scalar_bar(title=bar, render=False)
        self.plotter.subplot(0, 1)
        for bar in list(self.plotter.scalar_bars.keys()):
            self.plotter.remove_scalar_bar(title=bar, render=False)

    def toggle_stimuli(self):
        self.show_stimuli = not self.show_stimuli

    def tick_stimuli(self, t, data):
        # Remove outdated stimuli
        def func(x):
            t_stim, actor = x
            if (
                t <= t_stim
                or t >= t_stim + self.STIMULI_VISIBLE_FRAMES
                or not self.show_stimuli
            ):
                self.plotter.remove_actor(
                    actor, reset_camera=False, render=False)
                return False
            else:
                return True

        self.stimuli_actors = list(filter(func, self.stimuli_actors))

        # Add new stimuli
        if self.show_stimuli and "stimuli" in data:
            self.process_stimuli(t, data["stimuli"].tolist())

    def process_stimuli(self, t, stimuli):
        self.plotter.subplot(0, 0)
        for stimulus in stimuli:
            if len(stimulus) == 3:
                print(f"Stimulus found in file #{t} at position {stimulus}")
                s = pv.Sphere(radius=self.stimuli_radius, center=stimulus)
                actor = self.plotter.add_mesh(
                    s, color="red", render=False, reset_camera=False
                )
                self.stimuli_actors.append((t, actor))
        self.plotter.subplot(0, 1)  # because it fixes blinking bug

    def toggle_clipping(self):
        self.clipping = not self.clipping
        if not self.clipping:
            self.plotter.clear_plane_widgets()

    def toggle_clipping_plane(self):
        self.plotter.subplot(0, 0)
        self.clip_show_plane = not self.clip_show_plane
        if self.clip_show_plane:
            self.clipping = True
            self.plotter.add_plane_widget(
                self.set_clip, self.clip_normal, self.clip_origin
            )  # , implicit=False
        else:
            self.plotter.clear_plane_widgets()
        self.plotter.subplot(0, 1)  # because it fixes blinking bug

    def set_clip(self, normal, origin):
        self.clip_normal = normal
        self.clip_origin = origin

    def toggle_glyph(self, _, reload=True):
        self.show_glyph = not self.show_glyph

        if self.show_glyph:
            self._remove_scale_bars()

            if self.active_vector_field is None:
                self.GLYPH_COLOR_SCALAR = "Magnitude"
                self.cmap2 = "Reds"
            else:
                self.GLYPH_COLOR_SCALAR = "XY"
                self.cmap2 = "twilight"
            self.ui.right_cmap.setCurrentIndex(
                self.ui.right_cmap.findText(self.cmap2))

            self.plotter.subplot(0, 1)
            if not self.hide_glyph_sliders:
                self.plotter.add_slider_widget(
                    self.set_glyph_scale,
                    (0, 50),
                    value=self.GLYPH_SCALE_FACTOR,
                    title="Glyph Scale",
                    pointa=(0.1, 0.92),
                    pointb=(0.5, 0.92),
                    style="modern",
                )
                self.plotter.add_slider_widget(
                    self.set_glyph_skip,
                    (1, 100),
                    value=self.GLYPH_ELEM_SKIP,
                    title="Glyph Skip",
                    pointa=(0.55, 0.92),
                    pointb=(0.95, 0.92),
                    style="modern",
                    fmt="%.0f",
                )
            self.plotter.subplot(0, 0)
        else:
            self.plotter.clear_slider_widgets()
            # for widget in self.glyph_widgets: widget.Off()
            # self.glyph_widgets = []

        if reload:
            self.reload_mesh(reload_glyph=False)

    def set_glyph_colorscalar(self):
        if self.sender() == self.ui.actionGlyphMagnitude:
            self.GLYPH_COLOR_SCALAR = "Magnitude"
        else:
            if self.sender() == self.ui.actionGlyphXY:
                self.GLYPH_COLOR_SCALAR = "XY"
            elif self.sender() == self.ui.actionGlyphXZ:
                self.GLYPH_COLOR_SCALAR = "XZ"
            elif self.sender() == self.ui.actionGlyphYZ:
                self.GLYPH_COLOR_SCALAR = "YZ"
            self.cmap2 = "twilight"
            self.ui.right_cmap.setCurrentIndex(self.ui.right_cmap.findText(self.cmap2))

        self._remove_scale_bars()
        self.load(self.t, force_mesh_reload=True)

    def orbit(self, _, create_video=True):
        # compute orbit radius
        pos = np.array(self.plotter.camera.position)
        radius = np.linalg.norm(pos - self.point_cloud1.center)

        viewup = self.plotter.camera.up
        # camera orbit path
        path = pv.Polygon(
            center=self.point_cloud1.center, radius=radius, normal=viewup, n_sides=180
        )
        # self.plotter.add_mesh(path, style='wireframe', color='b')

        if create_video:
            self.screen_exporter.toggle_movie(self.plotter)
        orbit_on_path(
            self.plotter,
            path,
            step=0.1,
            viewup=viewup,
            focus=self.point_cloud1.center,
            threaded=not create_video,
            write_frames=create_video,
            progress_bar=True,
        )
        if create_video:
            self.screen_exporter.stop(self.plotter)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dir",
        nargs="?",
        type=path_validator,
        default=None,
        help="Directory to load or path to a single .npz file",
    )
    parser.add_argument(
        "dir2",
        nargs="?",
        type=path_validator,
        default=None,
        help="Optinal second directory to load or path to a single .npz file",
    )
    parser.add_argument(
        "--cam_pos",
        help=f"One of {list(CAMERA_POSITIONS.keys())} or the 3x2 matrix from 'Print Camera Position'",
        type=str,
        default="front",
    )
    parser.add_argument("-g", "--show_glyph", action="store_true")
    parser.add_argument("--scalar", type=str, default=None)
    parser.add_argument("--point_size", type=float, default=None)
    parser.add_argument(
        "--wireframe",
        action="store_true",
        help="show wireframe with volume rendering (VERY SLOW)",
    )
    parser.add_argument("--offscreen", "--hide", action="store_true", dest="offscreen")
    parser.add_argument("--pngs", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--export_dir", type=Path, default=None)
    parser.add_argument(
        "--offscreen_width",
        type=int,
        default=2048,
        help="output width, requires --hide",
    )
    parser.add_argument(
        "--offscreen_height",
        type=int,
        default=1024,
        help="output height, requires --hide",
    )
    parser.add_argument(
        "--less_lights",
        action="store_true",
        help="remove additional lights from the scene",
    )
    parser.add_argument(
        "--orbit_video",
        action="store_true",
        help="Create an orbit video only (do not use --video)",
    )
    args, unparsed_args = parser.parse_known_args()

    if args.export_dir is not None:
        global EXPORT_BASE_DIR
        EXPORT_BASE_DIR = args.export_dir

    app = Qt.QApplication(sys.argv[:1] + unparsed_args)
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # noinspection PyUnusedLocal
    window = MainWindow(
        path1=args.dir,
        path2=args.dir2,
        camera_position=args.cam_pos,
        scalar=args.scalar,
        show_glyph=args.show_glyph,
        show_wireframe=args.wireframe,
        offscreen=args.offscreen,
        make_pngs=args.pngs,
        make_video=args.video,
        offscreen_width=args.offscreen_width,
        offscreen_height=args.offscreen_height,
        less_lights=args.less_lights,
    )
    if args.point_size:
        window.set_point_size(args.point_size)
    if args.orbit_video:
        window.orbit(True, create_video=True)
        sys.exit(0)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
