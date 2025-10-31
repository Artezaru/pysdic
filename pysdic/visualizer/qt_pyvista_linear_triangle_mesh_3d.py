# Copyright 2025 Artezaru
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union, Optional, Dict
import numpy
import pyvista
from PyQt5 import QtWidgets, QtCore, QtGui
from pyvistaqt import QtInteractor
import matplotlib.pyplot as plt

from ..geometry import LinearTriangleMesh3D, IntegrationPoints

class QtPyvistaLinearTriangleMesh3D(QtWidgets.QMainWindow):
    r"""
    QT application to visualize a :class:`pysdic.geometry.LinearTriangleMesh3D` with PyVista.

    The windows contains a ComboBox to select the property to visualize.

    - The properties stored into the :attr:`pysdic.geometry.LinearTriangleMesh3D._vertices_properties`.
    - The additional properties given in the ``property_arrays`` argument.

    The window contains a ComboBox to select the texture to apply on the mesh.

    The window can also display integration points given in ``points_clouds`` argument.

    Parameters
    ----------
    mesh : LinearTriangleMesh3D
        The 3D linear triangle mesh to visualize.

    property_arrays : dict[str, numpy.ndarray], optional
        A dictionary mapping property names to numpy arrays containing the property values at each vertex.

    textures : dict[str, numpy.ndarray], optional
        A dictionary mapping texture names to numpy arrays representing images to apply on the mesh.

    integration_points : dict[str, IntegrationPoints], optional
        A dictionary mapping names to IntegrationPoints instances to visualize alongside the mesh.

    """
    def __init__(self, 
        mesh: LinearTriangleMesh3D, 
        property_arrays: Optional[Dict[str, numpy.ndarray]] = None,
        textures: Optional[Dict[str, numpy.ndarray]] = None,
        integration_points: Optional[Dict[str, IntegrationPoints]] = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Visualisation LinearTriangleMesh3D with pyvista")

        # ----- Save the inputs -------
        self.mesh = mesh
        self.textures = textures
        self.integration_points = integration_points
        self.property_arrays = property_arrays

        if self.property_arrays is None:
            self.property_arrays = {}
        if self.textures is None:
            self.textures = {}
        if self.integration_points is None:
            self.integration_points = {}

        # ----- Check the inputs -------
        if not isinstance(self.mesh, LinearTriangleMesh3D):
            raise TypeError("mesh must be an instance of LinearTriangleMesh3D.")
        if self.mesh.n_vertices == 0:
            raise ValueError("mesh must have at least one vertex.")
        if self.mesh.n_elements == 0:
            raise ValueError("mesh must have at least one element.")

        if len(self.property_arrays) > 0:
            if not isinstance(self.property_arrays, dict):
                raise TypeError("property_arrays must be a dictionary mapping strings to numpy arrays.")
            for key, array in self.property_arrays.items():
                if not isinstance(key, str):
                    raise TypeError("All keys in property_arrays must be strings.")
                if not isinstance(array, numpy.ndarray):
                    raise TypeError(f"property_arrays[{key}] must be a numpy.ndarray.")
                if not array.ndim == 1 and not array.ndim == 2:
                    raise ValueError(f"property_arrays[{key}] must be a 1D or 2D numpy array.")
                if array.shape[0] != self.mesh.n_vertices:
                    raise ValueError(f"property_arrays[{key}] must have shape ({self.mesh.n_vertices},) or ({self.mesh.n_vertices}, A) where A is the number of attributes.")
                if key in self.mesh.list_vertices_properties():
                    raise ValueError(f"property_arrays key '{key}' already exists in mesh vertices properties.")

        if len(self.textures) > 0:
            if self.mesh.elements_uvmap is None:
                raise ValueError("The mesh must have the 'uvmap' property set to visualize texture.")
            if not isinstance(self.textures, dict):
                raise TypeError("textures must be a dictionary mapping strings to numpy arrays.")
            for key, array in self.textures.items():
                if not isinstance(key, str):
                    raise TypeError("All keys in textures must be strings.")
                if not isinstance(array, numpy.ndarray):
                    raise TypeError(f"textures[{key}] must be a numpy.ndarray.")
                if array.ndim != 2 and array.ndim != 3:
                    raise ValueError(f"textures[{key}] must be a 2D or 3D numpy array representing an image.")
                if array.ndim == 3 and array.shape[2] not in [1, 3]:
                    raise ValueError(f"textures[{key}] must have 1 (L) or 3 (RGB) channels.")
                if array.dtype != numpy.uint8:
                    raise ValueError(f"textures[{key}] must have dtype numpy.uint8 with values in [0, 255].")
                

        if len(self.integration_points) > 0:
            if not isinstance(self.integration_points, dict):
                raise TypeError("integration_points must be a dictionary mapping strings to IntegrationPoints instances.")
            for key, points in self.integration_points.items():
                if not isinstance(key, str):
                    raise TypeError("All keys in integration_points must be strings.")
                if not isinstance(points, IntegrationPoints):
                    raise TypeError(f"integration_points[{key}] must be an instance of IntegrationPoints.")
                if points.n_dimensions != self.mesh._n_dimensions:
                    raise ValueError(f"integration_points must have {self.mesh._n_dimensions} dimensions.")

        # ----- Create the PyVista widget -------
        self.plotter = QtInteractor(self)
        
        # ----- Create the pyvista mesh (for property visualization) -------
        self.pv_mesh_property = pyvista.PolyData(self.mesh.vertices.points, numpy.hstack((numpy.full((self.mesh.n_elements, 1), 3), self.mesh.connectivity)).astype(numpy.int64))

        # ----- Create the pyvista mesh (for texture visualization) -------
        if self.textures is not None:
            # Duplicate points per face
            fictive_vertices = numpy.zeros((self.mesh.n_elements * 3, 3), dtype=numpy.float64)
            fictive_vertices[0::3, :] = self.mesh.vertices.points[self.mesh.connectivity[:, 0], :]
            fictive_vertices[1::3, :] = self.mesh.vertices.points[self.mesh.connectivity[:, 1], :]
            fictive_vertices[2::3, :] = self.mesh.vertices.points[self.mesh.connectivity[:, 2], :]
            fictive_connectivity = numpy.arange(self.mesh.n_elements * 3, dtype=numpy.int64).reshape(self.mesh.n_elements, 3)

            self.pv_mesh_texture = pyvista.PolyData(fictive_vertices, numpy.hstack((numpy.full((self.mesh.n_elements, 1), 3), fictive_connectivity)).astype(numpy.int64))
            self.pv_mesh_texture.active_texture_coordinates = numpy.zeros((self.mesh.n_elements * 3, 2), dtype=numpy.float64)
            self.pv_mesh_texture.active_texture_coordinates[0::3, 0] = self.mesh.elements_uvmap[:, 0]  # u1
            self.pv_mesh_texture.active_texture_coordinates[0::3, 1] = self.mesh.elements_uvmap[:, 1]  # v1
            self.pv_mesh_texture.active_texture_coordinates[1::3, 0] = self.mesh.elements_uvmap[:, 2]  # u2
            self.pv_mesh_texture.active_texture_coordinates[1::3, 1] = self.mesh.elements_uvmap[:, 3]  # v2
            self.pv_mesh_texture.active_texture_coordinates[2::3, 0] = self.mesh.elements_uvmap[:, 4]  # u3
            self.pv_mesh_texture.active_texture_coordinates[2::3, 1] = self.mesh.elements_uvmap[:, 5]  # v3

        # ----- Add properties to the pyvista mesh -------
        self.all_properties = {}

        for key in self.mesh.list_vertices_properties():
            property_axis = {}
            property_axis[f"(Magnitude)"] = numpy.linalg.norm(self.mesh.get_vertices_property(key), axis=1)
            for axis in range(self.mesh.get_vertices_property(key).shape[1]):
                property_axis[f"(Axis {axis})"] = self.mesh.get_vertices_property(key)[:, axis]
            self.all_properties[key] = property_axis

        if len(self.property_arrays) > 0:
            for key, array in self.property_arrays.items():
                property_axis = {}
                if array.ndim == 1:
                    array = array.reshape(-1, 1)
                property_axis[f"(Magnitude)"] = numpy.linalg.norm(array, axis=1)
                for axis in range(array.shape[1]):
                    property_axis[f"(Axis {axis})"] = array[:, axis]
                self.all_properties[key] = property_axis



        # ----- Create the principal layout -------
        central_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout(central_widget)

        display_layout = QtWidgets.QHBoxLayout()
        display_layout.addWidget(self.plotter.interactor)
        main_layout.addLayout(display_layout)

        controls_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(controls_layout)

        self.setCentralWidget(central_widget)

        # ------ Mode Layout -------
        control_groupbox = QtWidgets.QGroupBox("Mode Controls")
        control_groupbox.setStyleSheet("QGroupBox { font-weight: bold; font-size: 18px; }")
        control_groupbox.setMinimumWidth(300)
        control_groupbox.setMaximumWidth(400)
        control_groupbox.setMaximumHeight(170)
        controls_layout.addWidget(control_groupbox)
        groupbox_layout = QtWidgets.QVBoxLayout(control_groupbox)

        self.mode_combo = QtWidgets.QComboBox()

        modes = ["Geometry"]
        if len(self.all_properties) > 0:
            modes.append("Property")
        if len(self.textures) > 0:
            modes.append("Texture")
        if len(self.integration_points) > 0:
            modes.append("Integration Points")

        self.mode_combo.addItems(modes)
        self.mode_combo.currentTextChanged.connect(self._on_change_mode)
        groupbox_layout.addWidget(self.mode_combo)

        self.vertices_property_combo = QtWidgets.QComboBox()
        if len(self.all_properties) > 0:
            self.vertices_property_combo.addItems(self.all_properties.keys())
            self.vertices_property_combo.currentTextChanged.connect(self._on_change_vertices_property)
            groupbox_layout.addWidget(self.vertices_property_combo)

        self.vertices_property_axis_combo = QtWidgets.QComboBox()
        if len(self.all_properties) > 0:
            self.vertices_property_axis_combo.addItems(self.all_properties[list(self.all_properties.keys())[0]].keys())
            self.vertices_property_axis_combo.currentTextChanged.connect(self._on_change_vertices_property_axis)
            groupbox_layout.addWidget(self.vertices_property_axis_combo)

        self.texture_combo = QtWidgets.QComboBox()
        if len(self.textures) > 0:
            self.texture_combo.addItems(self.textures.keys())
            self.texture_combo.currentTextChanged.connect(self._on_change_texture)
            groupbox_layout.addWidget(self.texture_combo)

        self.integration_points_combo = QtWidgets.QComboBox()
        if len(self.integration_points) > 0:
            self.integration_points_combo.addItems(self.integration_points.keys())
            self.integration_points_combo.currentTextChanged.connect(self._on_change_integration_points)
            groupbox_layout.addWidget(self.integration_points_combo)

        
        # ------ Create the Scroll Area for controls -------
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumWidth(400)
        scroll_area.setMaximumHeight(1100)
        controls_layout.addWidget(scroll_area)

        scroll_area_widget = QtWidgets.QWidget()
        scroll_area.setWidget(scroll_area_widget)
        scroll_area_layout = QtWidgets.QVBoxLayout(scroll_area_widget)

        # ------ Displaying Layout -------
        control_groupbox = QtWidgets.QGroupBox("Display Controls")
        control_groupbox.setStyleSheet("QGroupBox { font-weight: bold; font-size: 18px; }")
        control_groupbox.setMinimumWidth(300)
        control_groupbox.setMaximumWidth(400)
        scroll_area_layout.addWidget(control_groupbox)
        groupbox_layout = QtWidgets.QVBoxLayout(control_groupbox)

        subgroupbox = QtWidgets.QGroupBox("Geometry Display Settings")
        subgroupbox.setMaximumHeight(600)
        subgroupbox.setStyleSheet("QGroupBox { font-weight: bold; font-size: 14px; }")
        groupbox_layout.addWidget(subgroupbox)
        subgroupbox_layout = QtWidgets.QVBoxLayout(subgroupbox)

        self.show_vertices_checkbox = QtWidgets.QCheckBox("Show Vertices")
        self.show_vertices_checkbox.setChecked(True)
        self.show_vertices_checkbox.stateChanged.connect(self._on_change_show_vertices)
        subgroupbox_layout.addWidget(self.show_vertices_checkbox)
        self._current_show_vertices = True

        label = QtWidgets.QLabel("Vertices Color:")
        subgroupbox_layout.addWidget(label)
        self.vertices_color_combo = QtWidgets.QComboBox()
        self.vertices_color_combo.addItems(["black", "white", "gray", "red", "green", "blue", "yellow", "cyan", "magenta", "orange"])
        self.vertices_color_combo.currentTextChanged.connect(self._on_change_vertices_color)
        subgroupbox_layout.addWidget(self.vertices_color_combo)
        self._current_vertices_color = "black"

        label = QtWidgets.QLabel("Vertices Size:")
        subgroupbox_layout.addWidget(label)
        self.vertices_size_slider = QtWidgets.QSlider(orientation=QtCore.Qt.Horizontal)
        self.vertices_size_slider.setMinimum(1)
        self.vertices_size_slider.setMaximum(50)
        self.vertices_size_slider.setValue(5)
        self.vertices_size_slider.valueChanged.connect(self._on_change_vertices_size)
        self._current_vertices_size = 5
        self.vertice_size_label = QtWidgets.QLabel(f"{self._current_vertices_size} px")
        self.vertice_size_label.setFixedWidth(70)
        self.vertice_size_label.setAlignment(QtCore.Qt.AlignRight)
        horizontal_layout = QtWidgets.QHBoxLayout()
        horizontal_layout.addWidget(self.vertices_size_slider)
        horizontal_layout.addWidget(self.vertice_size_label)
        subgroupbox_layout.addLayout(horizontal_layout)


        label = QtWidgets.QLabel("Vertices Opacity (%):")
        subgroupbox_layout.addWidget(label)
        self.vertices_size_slider = QtWidgets.QSlider(orientation=QtCore.Qt.Horizontal)
        self.vertices_size_slider.setMinimum(0)
        self.vertices_size_slider.setMaximum(100)
        self.vertices_size_slider.setValue(100)
        self.vertices_size_slider.valueChanged.connect(self._on_change_vertices_opacity)
        self._current_vertices_opacity = 1.0
        self.vertice_opacity_label = QtWidgets.QLabel(f"{self._current_vertices_opacity * 100:.0f} %")
        self.vertice_opacity_label.setFixedWidth(70)
        self.vertice_opacity_label.setAlignment(QtCore.Qt.AlignRight)
        horizontal_layout = QtWidgets.QHBoxLayout()
        horizontal_layout.addWidget(self.vertices_size_slider)
        horizontal_layout.addWidget(self.vertice_opacity_label)
        subgroupbox_layout.addLayout(horizontal_layout)

        self.show_edges_checkbox = QtWidgets.QCheckBox("Show Edges")
        self.show_edges_checkbox.setChecked(True)
        self.show_edges_checkbox.stateChanged.connect(self._on_change_show_edges)
        subgroupbox_layout.addWidget(self.show_edges_checkbox)
        self._current_show_edges = True

        label = QtWidgets.QLabel("Edges Color:")
        subgroupbox_layout.addWidget(label)
        self.edges_color_combo = QtWidgets.QComboBox()
        self.edges_color_combo.addItems(["black", "white", "gray", "red", "green", "blue", "yellow", "cyan", "magenta", "orange"])
        self.edges_color_combo.currentTextChanged.connect(self._on_change_edges_color)
        subgroupbox_layout.addWidget(self.edges_color_combo)
        self._current_edges_color = "black"

        label = QtWidgets.QLabel("Edges Width:")
        subgroupbox_layout.addWidget(label)
        self.edge_width_slider = QtWidgets.QSlider(orientation=QtCore.Qt.Horizontal)
        self.edge_width_slider.setMinimum(1)
        self.edge_width_slider.setMaximum(50)
        self.edge_width_slider.setValue(2)
        self.edge_width_slider.valueChanged.connect(self._on_change_edge_width)
        self._current_edge_width = 2
        self.edge_width_label = QtWidgets.QLabel(f"{self._current_edge_width} px")
        self.edge_width_label.setFixedWidth(70)
        self.edge_width_label.setAlignment(QtCore.Qt.AlignRight)
        horizontal_layout = QtWidgets.QHBoxLayout()
        horizontal_layout.addWidget(self.edge_width_slider)
        horizontal_layout.addWidget(self.edge_width_label)
        subgroupbox_layout.addLayout(horizontal_layout)

        label = QtWidgets.QLabel("Edges Opacity (%):")
        subgroupbox_layout.addWidget(label)
        self.edge_opacity_slider = QtWidgets.QSlider(orientation=QtCore.Qt.Horizontal)
        self.edge_opacity_slider.setMinimum(0)
        self.edge_opacity_slider.setMaximum(100)
        self.edge_opacity_slider.setValue(100)
        self.edge_opacity_slider.valueChanged.connect(self._on_change_edge_opacity)
        self._current_edge_opacity = 1.0
        self.edge_opacity_label = QtWidgets.QLabel(f"{self._current_edge_opacity * 100:.0f} %")
        self.edge_opacity_label.setFixedWidth(70)
        self.edge_opacity_label.setAlignment(QtCore.Qt.AlignRight)
        horizontal_layout = QtWidgets.QHBoxLayout()
        horizontal_layout.addWidget(self.edge_opacity_slider)
        horizontal_layout.addWidget(self.edge_opacity_label)
        subgroupbox_layout.addLayout(horizontal_layout)

        self.show_faces_checkbox = QtWidgets.QCheckBox("Show Faces")
        self.show_faces_checkbox.setChecked(True)
        self.show_faces_checkbox.stateChanged.connect(self._on_change_show_faces)
        subgroupbox_layout.addWidget(self.show_faces_checkbox)
        self._current_show_faces = True

        label = QtWidgets.QLabel("Faces Color:")
        subgroupbox_layout.addWidget(label)
        self.faces_color_combo = QtWidgets.QComboBox()
        self.faces_color_combo.addItems(["lightgray", "white", "gray", "red", "green", "blue", "yellow", "cyan", "magenta", "orange"])
        self.faces_color_combo.currentTextChanged.connect(self._on_change_faces_color)
        subgroupbox_layout.addWidget(self.faces_color_combo)
        self._current_faces_color = "lightgray"
        
        label = QtWidgets.QLabel("Faces Opacity (%):")
        subgroupbox_layout.addWidget(label)
        self.faces_opacity_slider = QtWidgets.QSlider(orientation=QtCore.Qt.Horizontal)
        self.faces_opacity_slider.setMinimum(0)
        self.faces_opacity_slider.setMaximum(100)
        self.faces_opacity_slider.setValue(100)
        self.faces_opacity_slider.valueChanged.connect(self._on_change_faces_opacity)
        self._current_faces_opacity = 1.0
        self.faces_opacity_label = QtWidgets.QLabel(f"{self._current_faces_opacity * 100:.0f} %")
        self.faces_opacity_label.setFixedWidth(70)
        self.faces_opacity_label.setAlignment(QtCore.Qt.AlignRight)
        horizontal_layout = QtWidgets.QHBoxLayout()
        horizontal_layout.addWidget(self.faces_opacity_slider)
        horizontal_layout.addWidget(self.faces_opacity_label)
        subgroupbox_layout.addLayout(horizontal_layout)


        subgroupbox = QtWidgets.QGroupBox("Property Display Settings")
        subgroupbox.setMaximumHeight(250)
        subgroupbox.setStyleSheet("QGroupBox { font-weight: bold; font-size: 14px; }")
        subgroupbox.setVisible(len(self.all_properties) > 0)
        groupbox_layout.addWidget(subgroupbox)
        subgroupbox_layout = QtWidgets.QVBoxLayout(subgroupbox)

        label = QtWidgets.QLabel("Property Color Map:")
        subgroupbox_layout.addWidget(label)
        colormaps = plt.colormaps()
        model = QtGui.QStandardItemModel()
        for cmap in colormaps:
            item = QtGui.QStandardItem()
            icon = self._create_colormap_icon(cmap)
            item.setIcon(icon)
            item.setText(cmap)
            model.appendRow(item)
        self.property_color_map_combo = QtWidgets.QComboBox()
        self.property_color_map_combo.setModel(model)
        self.property_color_map_combo.setIconSize(QtCore.QSize(100, 12))  # Preview size
        self.property_color_map_combo.currentTextChanged.connect(self._on_change_property_color_map)
        subgroupbox_layout.addWidget(self.property_color_map_combo)
        self._current_property_color_map = colormaps[0]

        self.log_scale_checkbox = QtWidgets.QCheckBox("Logarithmic Scale")
        self.log_scale_checkbox.setChecked(False)
        self.log_scale_checkbox.stateChanged.connect(self._on_change_log_scale)
        subgroupbox_layout.addWidget(self.log_scale_checkbox)
        self._current_log_scale = False

        self.automatic_clim = QtWidgets.QCheckBox("Automatic Color Limits")
        self.automatic_clim.setChecked(True)
        self.automatic_clim.stateChanged.connect(self._on_change_automatic_clim)
        subgroupbox_layout.addWidget(self.automatic_clim)
        self._current_automatic_clim = True

        label = QtWidgets.QLabel("Manual Color Limits (min, max):")
        subgroupbox_layout.addWidget(label)
        self.manual_clim_min = QtWidgets.QLineEdit()
        self.manual_clim_min.setPlaceholderText("Min")
        self.manual_clim_min.setFixedWidth(80)
        self.manual_clim_max = QtWidgets.QLineEdit()
        self.manual_clim_max.setPlaceholderText("Max")
        self.manual_clim_max.setFixedWidth(80)
        clim_layout = QtWidgets.QHBoxLayout()
        clim_layout.addWidget(self.manual_clim_min)
        clim_layout.addWidget(self.manual_clim_max)
        subgroupbox_layout.addLayout(clim_layout)
        self._current_manual_clim = (None, None)

        self.button_apply_clim = QtWidgets.QPushButton("Apply clim")
        self.button_apply_clim.clicked.connect(self._on_change_manual_clim)
        self.button_apply_clim.setEnabled(False)
        subgroupbox_layout.addWidget(self.button_apply_clim)


        subgroupbox = QtWidgets.QGroupBox("Texture Display Settings")
        subgroupbox.setMaximumHeight(80)
        subgroupbox.setStyleSheet("QGroupBox { font-weight: bold; font-size: 14px; }")
        subgroupbox.setVisible(len(self.textures) > 0)
        groupbox_layout.addWidget(subgroupbox)
        subgroupbox_layout = QtWidgets.QVBoxLayout(subgroupbox)

        self.use_RGB_texture_checkbox = QtWidgets.QCheckBox("Use RGB Texture")
        self.use_RGB_texture_checkbox.setChecked(True)
        self.use_RGB_texture_checkbox.stateChanged.connect(self._on_change_use_RGB_texture)
        subgroupbox_layout.addWidget(self.use_RGB_texture_checkbox)
        self._current_use_RGB_texture = True


        subgroupbox = QtWidgets.QGroupBox("integration points Display Settings")
        subgroupbox.setMaximumHeight(200)
        subgroupbox.setStyleSheet("QGroupBox { font-weight: bold; font-size: 14px; }")
        subgroupbox.setVisible(len(self.integration_points) > 0)
        groupbox_layout.addWidget(subgroupbox)
        subgroupbox_layout = QtWidgets.QVBoxLayout(subgroupbox)

        label = QtWidgets.QLabel("integration points Color:")
        subgroupbox_layout.addWidget(label)
        self.integration_points_color_combo = QtWidgets.QComboBox()
        self.integration_points_color_combo.addItems(["black", "white", "gray", "red", "green", "blue", "yellow", "cyan", "magenta", "orange"])
        self.integration_points_color_combo.currentTextChanged.connect(self._on_change_integration_points_color)
        subgroupbox_layout.addWidget(self.integration_points_color_combo)
        self._current_integration_points_color = "black"

        label = QtWidgets.QLabel("integration points Size:")
        subgroupbox_layout.addWidget(label)
        self.integration_points_size_slider = QtWidgets.QSlider(orientation=QtCore.Qt.Horizontal)
        self.integration_points_size_slider.setMinimum(1)
        self.integration_points_size_slider.setMaximum(50)
        self.integration_points_size_slider.setValue(10)
        self.integration_points_size_slider.valueChanged.connect(self._on_change_integration_points_size)
        self._current_integration_points_size = 10
        self.integration_points_size_label = QtWidgets.QLabel(f"{self._current_integration_points_size} px")
        self.integration_points_size_label.setFixedWidth(70)
        self.integration_points_size_label.setAlignment(QtCore.Qt.AlignRight)
        horizontal_layout = QtWidgets.QHBoxLayout()
        horizontal_layout.addWidget(self.integration_points_size_slider)
        horizontal_layout.addWidget(self.integration_points_size_label)
        subgroupbox_layout.addLayout(horizontal_layout)

        label = QtWidgets.QLabel("integration points Opacity (%):")
        subgroupbox_layout.addWidget(label)
        self.integration_points_opacity_slider = QtWidgets.QSlider(orientation=QtCore.Qt.Horizontal)
        self.integration_points_opacity_slider.setMinimum(0)
        self.integration_points_opacity_slider.setMaximum(100)
        self.integration_points_opacity_slider.setValue(100)
        self.integration_points_opacity_slider.valueChanged.connect(self._on_change_integration_points_opacity)
        self._current_integration_points_opacity = 1.0
        self.integration_points_opacity_label = QtWidgets.QLabel(f"{self._current_integration_points_opacity * 100:.0f} %")
        self.integration_points_opacity_label.setFixedWidth(70)
        self.integration_points_opacity_label.setAlignment(QtCore.Qt.AlignRight)
        horizontal_layout = QtWidgets.QHBoxLayout()
        horizontal_layout.addWidget(self.integration_points_opacity_slider)
        horizontal_layout.addWidget(self.integration_points_opacity_label)
        subgroupbox_layout.addLayout(horizontal_layout)


        # ---- Update continuously control ----
        group_box = QtWidgets.QGroupBox("Render Update Controls")
        group_box.setStyleSheet("QGroupBox { font-weight: bold; font-size: 18px; }")
        group_box.setMinimumWidth(300)
        group_box.setMaximumWidth(400)
        group_box.setMaximumHeight(120)
        controls_layout.addWidget(group_box)
        groupbox_layout = QtWidgets.QVBoxLayout(group_box)

        self.update_continuously_checkbox = QtWidgets.QCheckBox("Update Continuously")
        self.update_continuously_checkbox.setChecked(True)
        groupbox_layout.addWidget(self.update_continuously_checkbox)
        self._update_continuously = True
        self.update_continuously_checkbox.stateChanged.connect(self._on_change_update_continuously)

        self.update_render_button = QtWidgets.QPushButton("Update Render")
        self.update_render_button.clicked.connect(self._on_click_update_render)
        groupbox_layout.addWidget(self.update_render_button)
        self.update_render_button.setEnabled(not self._update_continuously)


        # ----- Initial plot -------
        self._current_mode = "Geometry"
        self._current_property = list(self.all_properties.keys())[0] if len(self.all_properties) != 0 else None
        self._current_property_axis = list(self.all_properties[self._current_property].keys())[0] if len(self.all_properties) != 0 else None
        self._current_texture = list(textures.keys())[0] if textures is not None else None
        self._current_integration_points = list(integration_points.keys())[0] if integration_points is not None else None
        self._on_change_mode(self._current_mode)

    def _hide_show_mode(self) -> None:
        if self._current_mode == "Geometry":
            self.vertices_property_combo.setVisible(False)
            self.vertices_property_axis_combo.setVisible(False)
            self.texture_combo.setVisible(False)
            self.integration_points_combo.setVisible(False)
    
        elif self._current_mode == "Property":
            self.vertices_property_combo.setVisible(True)
            self.vertices_property_axis_combo.setVisible(True)
            self.texture_combo.setVisible(False)
            self.integration_points_combo.setVisible(False)

        elif self._current_mode == "Texture":
            self.vertices_property_combo.setVisible(False)
            self.vertices_property_axis_combo.setVisible(False)
            self.texture_combo.setVisible(True)
            self.integration_points_combo.setVisible(False)

        elif self._current_mode == "Integration Points":
            self.vertices_property_combo.setVisible(False)
            self.vertices_property_axis_combo.setVisible(False)
            self.texture_combo.setVisible(False)
            self.integration_points_combo.setVisible(True)

    def _on_change_mode(self, mode: str) -> None:
        self._current_mode = mode
        self._hide_show_mode()

        if mode == "Geometry":
            self._plot_geometry_mode()
        elif mode == "Property":
            self._plot_property_mode()
        elif mode == "Texture":
            self._plot_texture_mode()
        elif mode == "Integration Points":
            self._plot_integration_points_mode()

    def _on_change_vertices_property(self, property_name: str) -> None:
        self._current_property = property_name

        self.vertices_property_axis_combo.blockSignals(True)
        self.vertices_property_axis_combo.clear()
        self.vertices_property_axis_combo.addItems(self.all_properties[self._current_property].keys())
        if self._current_property_axis in self.all_properties[self._current_property].keys():
            index = list(self.all_properties[self._current_property].keys()).index(self._current_property_axis)
            self.vertices_property_axis_combo.setCurrentIndex(index)
        else:
            self.vertices_property_axis_combo.setCurrentIndex(0)
            self._current_property_axis = list(self.all_properties[self._current_property].keys())[0]
        self.vertices_property_axis_combo.blockSignals(False)
        
        self._plot_property_mode()

    def _on_change_vertices_property_axis(self, property_axis: str) -> None:
        self._current_property_axis = property_axis
        self._plot_property_mode()

    def _on_change_texture(self, texture_name: str) -> None:
        self._current_texture = texture_name
        self._plot_texture_mode()

    def _on_change_integration_points(self, points_name: str) -> None:
        self._current_integration_points = points_name
        self._plot_integration_points_mode()

    def _clear_plot(self) -> None:
        self.plotter.clear()
        self._remove_scalars()

    def _render(self) -> None:
        self.plotter.show_axes()
        self.plotter.show_grid()
        self.plotter.render()
    
    def _remove_scalars(self) -> None:
        self.pv_mesh_property.clear_point_data()

    def _plot_edges(self) -> None:
        if self._current_show_edges:
            self.plotter.add_mesh(
                self.pv_mesh_property.extract_all_edges(), 
                color=self._current_edges_color,
                line_width=self._current_edge_width,
                opacity=self._current_edge_opacity,
            )

    def _plot_vertices(self) -> None:
        if self._current_show_vertices:
            self.plotter.add_points(
                self.mesh.vertices.points, 
                color=self._current_vertices_color, 
                opacity=self._current_vertices_opacity,
                point_size=self._current_vertices_size,
                render_points_as_spheres=True,
            )

    def _plot_mesh(self) -> None:
        if self._current_show_faces:
            self.plotter.add_mesh(
                self.pv_mesh_property, 
                color=self._current_faces_color, 
                opacity=self._current_faces_opacity,
            )

    def _plot_geometry_mode(self) -> None:
        self._clear_plot()
        self._plot_mesh()   
        self._plot_edges()
        self._plot_vertices()
        self._render()

    def _plot_property_mode(self) -> None:
        self._clear_plot()

        scalars = self.all_properties[self._current_property][self._current_property_axis]

        if self._current_automatic_clim:
            vmin = scalars.min()
            vmax = scalars.max()
        if not self._current_automatic_clim and self._current_manual_clim[0] is not None:
            vmin = self._current_manual_clim[0]
        if not self._current_automatic_clim and self._current_manual_clim[1] is not None:
            vmax = self._current_manual_clim[1]
        if not self._current_automatic_clim and self._current_manual_clim[0] is None:
            vmin = scalars.min()
        if not self._current_automatic_clim and self._current_manual_clim[1] is None:
            vmax = scalars.max()
        if vmin > vmax and self._current_manual_clim[0] is None:
            vmin = vmax
        if vmin > vmax and self._current_manual_clim[1] is None:
            vmax = vmin

        self.pv_mesh_property.point_data[self._current_property + " " + self._current_property_axis] = scalars
        self.plotter.add_mesh(
            self.pv_mesh_property,
            scalars=self._current_property + " " + self._current_property_axis,
            cmap=self._current_property_color_map,
            clim=[vmin, vmax],
            log_scale=self._current_log_scale,
            opacity=self._current_faces_opacity,
        )
        self._plot_edges()
        self._plot_vertices()
        self._render()

    def _plot_texture_mode(self) -> None:
        self._clear_plot()

        texture_array = self.textures[self._current_texture]
        if texture_array.ndim == 2:
            color_texture = numpy.repeat(texture_array[:, :, numpy.newaxis], 3, axis=2).astype(numpy.uint8)
        elif texture_array.ndim == 3 and texture_array.shape[2] == 1:
            color_texture = numpy.repeat(texture_array, 3, axis=2).astype(numpy.uint8)
        elif texture_array.ndim == 3 and self._current_use_RGB_texture and texture_array.shape[2] == 3:
            color_texture = texture_array
        elif texture_array.ndim == 3 and not self._current_use_RGB_texture and texture_array.shape[2] == 3:
            gray_texture = numpy.round(numpy.dot(texture_array[..., :3], [0.2989, 0.5870, 0.1140])).astype(numpy.uint8)
            color_texture = numpy.repeat(gray_texture[:, :, numpy.newaxis], 3, axis=2).astype(numpy.uint8)
        else:
            raise ValueError("Invalid texture array shape.")

        texture = pyvista.Texture(color_texture)

        self.plotter.add_mesh(
            self.pv_mesh_texture, 
            texture=texture, 
            opacity=self._current_faces_opacity,
        )
        self._plot_edges()
        self._plot_vertices()
        self._render()

    def _plot_integration_points_mode(self) -> None:
        self._clear_plot()
        self._plot_mesh() 
        self._plot_edges()
        self._plot_vertices()

        global_coords = self.mesh.integration_points_to_global_coordinates(self.integration_points[self._current_integration_points])
        self.plotter.add_points(
            global_coords, 
            color=self._current_integration_points_color, 
            point_size=self._current_integration_points_size,
            render_points_as_spheres=True,
        )

        self._render()

    # ---- Callbacks for control widgets ----
    def _on_change_vertices_color(self, color: str) -> None:
        self._current_vertices_color = color
        if self._update_continuously:
            self._on_change_mode(self._current_mode)

    def _on_change_vertices_size(self, value: int) -> None:
        self._current_vertices_size = value
        self.vertice_size_label.setText(f"{self._current_vertices_size} px")
        if self._update_continuously:
            self._on_change_mode(self._current_mode)
    
    def _on_change_vertices_opacity(self, value: int) -> None:
        self._current_vertices_opacity = value / 100.0
        self.vertice_opacity_label.setText(f"{self._current_vertices_opacity * 100:.0f} %")
        if self._update_continuously:
            self._on_change_mode(self._current_mode)
    
    def _on_change_edges_color(self, color: str) -> None:
        self._current_edges_color = color
        if self._update_continuously:
            self._on_change_mode(self._current_mode)

    def _on_change_faces_color(self, color: str) -> None:
        self._current_faces_color = color
        if self._update_continuously:
            self._on_change_mode(self._current_mode)

    def _on_change_faces_opacity(self, value: int) -> None:
        self._current_faces_opacity = value / 100.0
        self.faces_opacity_label.setText(f"{self._current_faces_opacity * 100:.0f} %")
        if self._update_continuously:
            self._on_change_mode(self._current_mode)

    def _on_change_edge_opacity(self, value: int) -> None:
        self._current_edge_opacity = value / 100.0
        self.edge_opacity_label.setText(f"{self._current_edge_opacity * 100:.0f} %")
        if self._update_continuously:
            self._on_change_mode(self._current_mode)

    def _on_change_edge_width(self, value: int) -> None:
        self._current_edge_width = value
        self.edge_width_label.setText(f"{self._current_edge_width} px")
        if self._update_continuously:
            self._on_change_mode(self._current_mode)

    def _on_change_show_vertices(self, state: int) -> None:
        self._current_show_vertices = self.show_vertices_checkbox.isChecked()
        if self._update_continuously:
            self._on_change_mode(self._current_mode)

    def _on_change_show_edges(self, state: int) -> None:
        self._current_show_edges = self.show_edges_checkbox.isChecked()
        if self._update_continuously:
            self._on_change_mode(self._current_mode)

    def _on_change_show_faces(self, state: int) -> None:
        self._current_show_faces = self.show_faces_checkbox.isChecked()
        if self._update_continuously:
            self._on_change_mode(self._current_mode)

    def _on_change_integration_points_color(self, color: str) -> None:
        self._current_integration_points_color = color
        if self._update_continuously:
            self._on_change_mode(self._current_mode)

    def _on_change_integration_points_size(self, value: int) -> None:
        self._current_integration_points_size = value
        self.integration_points_size_label.setText(f"{self._current_integration_points_size} px")
        if self._update_continuously:
            self._on_change_mode(self._current_mode)

    def _on_change_integration_points_opacity(self, value: int) -> None:
        self._current_integration_points_opacity = value / 100.0
        self.integration_points_opacity_label.setText(f"{self._current_integration_points_opacity * 100:.0f} %")
        if self._update_continuously:
            self._on_change_mode(self._current_mode)

    def _on_change_property_color_map(self, color_map: str) -> None:
        self._current_property_color_map = color_map
        if self._update_continuously:
            self._on_change_mode(self._current_mode)

    def _on_change_automatic_clim(self, state: int) -> None:
        self._current_automatic_clim = self.automatic_clim.isChecked()
        self.button_apply_clim.setEnabled(not self._current_automatic_clim)
        if self._update_continuously:
            self._on_change_mode(self._current_mode)

    def _on_change_manual_clim(self) -> None:
        try:
            min_val = float(self.manual_clim_min.text()) if self.manual_clim_min.text() != "" else None
            max_val = float(self.manual_clim_max.text()) if self.manual_clim_max.text() != "" else None
            if min_val is not None and max_val is not None and min_val >= max_val:
                raise ValueError("Minimum clim must be less than maximum clim.")
            self._current_manual_clim = (min_val, max_val)
            if self._update_continuously:
                self._on_change_mode(self._current_mode)
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Please enter valid numeric values for manual color limits. Floating-point with vmin < vmax.")

    def _on_change_log_scale(self, state: int) -> None:
        self._current_log_scale = self.log_scale_checkbox.isChecked()
        if self._update_continuously:
            self._on_change_mode(self._current_mode)

    def _on_change_use_RGB_texture(self, state: int) -> None:
        self._current_use_RGB_texture = self.use_RGB_texture_checkbox.isChecked()
        if self._update_continuously:
            self._on_change_mode(self._current_mode)

    def _on_change_integration_points_size(self, value: int) -> None:
        self._current_integration_points_size = value
        self.integration_points_size_label.setText(f"{self._current_integration_points_size} px")
        if self._update_continuously:
            self._on_change_mode(self._current_mode)

    def _on_change_update_continuously(self, state: int) -> None:
        self._update_continuously = self.update_continuously_checkbox.isChecked()
        self.update_render_button.setEnabled(not self._update_continuously)
        if self._update_continuously:
            self._on_change_mode(self._current_mode)

    def _on_click_update_render(self) -> None:
        self._on_change_mode(self._current_mode)

    def _create_colormap_icon(self, cmap_name: str, width: int = 100, height: int = 12) -> QtGui.QIcon:
        gradient = numpy.linspace(0, 1, width)
        gradient = numpy.tile(gradient, (height, 1))
        cmap = plt.get_cmap(cmap_name)
        rgba = (cmap(gradient)[:, :, :3] * 255).astype(numpy.uint8)
        image = QtGui.QImage(rgba.data, rgba.shape[1], rgba.shape[0], QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(image)
        return QtGui.QIcon(pixmap)



def visualize_qt_pyvista_linear_triangle_mesh_3d(
    mesh: LinearTriangleMesh3D, 
    property_arrays: Optional[Dict[str, numpy.ndarray]] = None,
    textures: Optional[Dict[str, numpy.ndarray]] = None,
    integration_points: Optional[Dict[str, IntegrationPoints]] = None,
) -> None:
    """
    Launch the QT application to visualize a :class:`pysdic.geometry.LinearTriangleMesh3D` with ``pyvista`` and ``pyqt5``.

    The windows contains a ComboBox to select the property to visualize.

    - The properties stored into the :attr:`pysdic.geometry.LinearTriangleMesh3D._vertices_properties`.
    - The additional properties given in the ``property_arrays`` argument.

    The window contains a ComboBox to select the texture to apply on the mesh.

    The window can also display integration points given in ``points_clouds`` argument.

    Parameters
    ----------
    mesh : LinearTriangleMesh3D
        The 3D linear triangle mesh to visualize.

    property_arrays : dict[str, numpy.ndarray], optional
        A dictionary mapping property names to numpy arrays containing the property values at each vertex.
        Each array must have shape (n_vertices,) or (n_vertices, 1) for scalar properties or (n_vertices, n_components) for vectorial properties.

    textures : dict[str, numpy.ndarray], optional
        A dictionary mapping texture names to numpy arrays representing images to apply on the mesh.
        Each array must have shape (height, width, 3) for RGB textures or (height, width) for grayscale textures.
        Integer arrays with values in [0, 255] with dtype ``numpy.uint8``.

    integration_points : dict[str, IntegrationPoints], optional
        A dictionary mapping names to IntegrationPoints instances to visualize alongside the mesh.
        Each IntegrationPoints instance must have a valid point cloud to visualize.

        
    Examples
    --------

    Lets create a 3D linear triangle mesh representing a heightmap surface.
    
    .. code-block:: python

        from pysdic.geometry import create_linear_triangle_heightmap
        import numpy as np

        surface_mesh = create_linear_triangle_heightmap(
            height_function=lambda x, y: 0.5 * np.sin(np.pi * x) * np.cos(np.pi * y),
            x_bounds=(-1.0, 1.0),
            y_bounds=(-1.0, 1.0),
            n_x=50,
            n_y=50,
        )

    Add some properties. They can be scalar or vectorial properties and can be stored directly in the mesh or provided as separate arrays.

    .. code-block:: python

        # Create some property arrays
        height = surface_mesh.vertices.points[:, 2].reshape(-1, 1)  # Use the z-coordinate as a property
        surface_mesh.set_vertices_property("Height", height) # Store in the mesh

        displacement = np.zeros_like(surface_mesh.vertices.points)
        displacement[:, 0] = 0.1 * np.sin(2 * np.pi * surface_mesh.vertices.points[:, 0])
        displacement[:, 1] = 0.1 * np.cos(2 * np.pi * surface_mesh.vertices.points[:, 1])
        displacement[:, 2] = 0.1 * np.sin(2 * np.pi * surface_mesh.vertices.points[:, 0]) * np.cos(2 * np.pi * surface_mesh.vertices.points[:, 1])
        properties = {
            "Displacement": displacement,  # Vectorial property stored as a separate array
        }

    Select some integration points to visualize. They can be constructed using the :class:`pysdic.geometry.IntegrationPoints` class.
    
    .. code-block:: python

        # Create some rays to cast
        ray_origins = np.random.uniform(-1, 1, (100, 3))
        ray_origins[:, 2] = 3.0  # Start above the surface
        ray_directions = np.tile(np.array([[0, 0, -1]]), (100, 1))  # Pointing downwards

        intersection_points = surface_mesh.cast_rays(ray_origins, ray_directions)
        intersection_points = {
            "Ray Intersections": intersection_points,
        }

    Select some textures can be applied on the mesh. The mesh must have valid UV mapping for texture application.

    .. code-block:: python

        # Create some textures
        u = np.linspace(0, 1, 50)
        v = np.linspace(0, 1, 50)
        U, V = np.meshgrid(u, v)

        texture_image_1 = np.round(255/2 + 255/2 * np.sin(U * 4 * np.pi)).astype(np.uint8)  # Example texture image with shape (50, 50)
        texture_image_2 = np.round(255/2 + 255/2 * np.sin(V * 4 * np.pi)).astype(np.uint8)  # Another example texture image with shape (50, 50)
        textures = {
            "Sine Texture": texture_image_1,
            "Cosine Texture": texture_image_2,
            "Coconut Texture": np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8),  # Random RGB texture with shape (16, 16, 3)
        }

    Visualize the mesh with the properties, textures and integration points.

    .. code-block:: python   

        # Visualize using the Qt PyVista viewer
        from pysdic.visualizer import visualize_qt_pyvista_linear_triangle_mesh_3d

        visualize_qt_pyvista_linear_triangle_mesh_3d(
            mesh=surface_mesh,
            property_arrays=properties,
            integration_points=intersection_points,
            textures=textures,
        )

    The open window allows to select the property or texture to visualize using ComboBoxes.

    .. figure:: ../../../pysdic/resources/qt_pyvista_linear_triangle_mesh_3d_example_geometry.png
        :align: center
        :width: 500px
        :alt: QT PyVista Linear Triangle Mesh 3D Example (Geometry Mode)

    .. figure:: ../../../pysdic/resources/qt_pyvista_linear_triangle_mesh_3d_example_property.png
        :align: center
        :width: 500px
        :alt: QT PyVista Linear Triangle Mesh 3D Example (Property Mode)

    .. figure:: ../../../pysdic/resources/qt_pyvista_linear_triangle_mesh_3d_example_texture.png
        :align: center
        :width: 500px
        :alt: QT PyVista Linear Triangle Mesh 3D Example (Texture Mode)

    .. figure:: ../../../pysdic/resources/qt_pyvista_linear_triangle_mesh_3d_example_integration_points.png
        :align: center
        :width: 500px
        :alt: QT PyVista Linear Triangle Mesh 3D Example (integration points Mode)    

    """
    app = QtWidgets.QApplication([])
    window = QtPyvistaLinearTriangleMesh3D(
        mesh=mesh,
        property_arrays=property_arrays,
        textures=textures,
        integration_points=integration_points,
    )
    window.show()
    app.exec_()

    



        