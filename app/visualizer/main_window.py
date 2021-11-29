import math
import time
import os

import PyQt5.QtWidgets as QtWidgets
import PyQt5.QtCore as Qt
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtk
from utils.config import *
from visualizer.file_dialog import FileDialog
from handlers.file_reader import FileReader

file_reader = FileReader()
#


class MainWindow(QtWidgets.QMainWindow, QtWidgets.QApplication):
    """
    """

    def __init__(self, app):
        """ Initialize UI components and show"""
        QtWidgets.QMainWindow.__init__(self, None)
        self.mask_file = ""
        self.brain_loaded = False
        self.slicer_widgets = []
        self.file_dialog = FileDialog()

        self.mask_opacity_sp = self.create_new_picker(
            1.0, 0.0, 0.1, MASK_OPACITY, self.mask_opacity_vc)
        self.mask_smoothness_sp = self.create_new_picker(
            1000, 100, 100, MASK_SMOOTHNESS, self.mask_smoothness_vc)
        self.mask_label_cbs = []

        # set layout and show
        self.setup_vtk()
        self.setup_ui_component()
        self.render_window.Render()
        self.setWindowTitle(APPLICATION_TITLE)
        self.frame.setLayout(self.grid)
        self.setCentralWidget(self.frame)
        self.interactor.Initialize()
        self.show()

    def setup_ui_component(self):
        # create frames and grid
        self.frame = QtWidgets.QFrame()
        self.frame.setAutoFillBackground(True)
        self.grid = QtWidgets.QGridLayout()
        # add widgets and window
        self.add_functions_settings_widget()
        self.add_view_settings_widget()
        self.add_mask_settings_widget()
        self.add_vtk_window_widget()

    def setup_vtk(self):
        """ set up vtk """
        renderer = vtk.vtkRenderer()
        vtk_widget = QVTKRenderWindowInteractor()
        interactor = vtk_widget.GetRenderWindow().GetInteractor()
        render_window = vtk_widget.GetRenderWindow()

        vtk_widget.GetRenderWindow().AddRenderer(renderer)
        render_window.AddRenderer(renderer)
        interactor.SetRenderWindow(render_window)
        interactor.SetInteractorStyle(
            vtk.vtkInteractorStyleTrackballCamera())

        self.renderer, self.vtk_widget, self.interactor, self.render_window = renderer, vtk_widget, interactor, render_window

    def add_brain_settings_widget(self):
        """ add settings for brain settings group"""
        brain_group_box = QtWidgets.QGroupBox("Brain Settings")
        brain_group_layout = QtWidgets.QGridLayout()
        brain_group_layout.addWidget(self.create_new_separator(), 5, 0, 1, 3)
        brain_group_layout.addWidget(QtWidgets.QLabel("Axial Slice"), 6, 0)
        brain_group_layout.addWidget(QtWidgets.QLabel("Coronal Slice"), 7, 0)
        brain_group_layout.addWidget(QtWidgets.QLabel("Sagittal Slice"), 8, 0)
        # order is important

        slicer_funcs = [self.brain_axial_slice_changed,
                        self.brain_coronal_slice_changed, self.brain_sagittal_slice_changed]
        current_label_row = 6
        # data extent is array [xmin, xmax, ymin, ymax, zmin, zmax)
        # we want all the max values for the range
        extent_index = 5
        for func in slicer_funcs:
            slice_widget = QtWidgets.QSlider(Qt.Qt.Horizontal)
            slice_widget.setEnabled(True)
            self.slicer_widgets.append(slice_widget)
            brain_group_layout.addWidget(
                slice_widget, current_label_row, 1, 1, 2)
            slice_widget.valueChanged.connect(func)
            slice_widget.setRange(
                self.brain.extent[extent_index - 1], self.brain.extent[extent_index])
            slice_widget.setValue(self.brain.extent[extent_index] / 2)
            current_label_row += 1
            extent_index -= 2

        brain_group_box.setLayout(brain_group_layout)
        self.grid.addWidget(brain_group_box, 8, 0, 1, 2)

    def reset_slicers(self):
        extent_index = 5
        for slicer in self.slicer_widgets:
            slicer.setValue(self.brain.extent[extent_index] / 2)
            extent_index -= 2

    def add_mask_settings_widget(self):
        """ add settings for mask settings group"""
        mask_settings_group_box = QtWidgets.QGroupBox("Mask Settings")
        mask_settings_layout = QtWidgets.QGridLayout()
        mask_settings_layout.addWidget(QtWidgets.QLabel("Mask Opacity"), 0, 0)
        mask_settings_layout.addWidget(
            QtWidgets.QLabel("Mask Smoothness"), 1, 0)
        mask_settings_layout.addWidget(self.mask_opacity_sp, 0, 2)
        mask_settings_layout.addWidget(self.mask_smoothness_sp, 1, 2)

        mask_settings_layout.addWidget(self.create_new_separator(), 2, 0, 1, 3)

        self.mask_label_cbs = []
        brats_mask_labels = ['Whole tumor', 'Enhance tumor core', 'Tumor core']
        for i in range(0, 3):
            self.mask_label_cbs.append(
                QtWidgets.QCheckBox(brats_mask_labels[i]))
            mask_settings_layout.addWidget(
                self.mask_label_cbs[i], 3, i)
            # c_row = c_row + 1 if c_col == 1 else c_row
            # c_col = 0 if c_col == 1 else 1

        mask_settings_group_box.setLayout(mask_settings_layout)
        self.grid.addWidget(mask_settings_group_box, 2, 0, 2, 2)

        if self.mask_file:
            for i, cb in enumerate(self.mask_label_cbs):
                if i < len(self.mask.labels) and self.mask.labels[i].actor:
                    cb.setChecked(True)
                    cb.clicked.connect(self.mask_label_checked)
                else:
                    cb.setDisabled(True)

    def add_view_settings_widget(self):
        """ add option to choose view """
        axial_view = QtWidgets.QPushButton("Axial")
        coronal_view = QtWidgets.QPushButton("Coronal")
        sagittal_view = QtWidgets.QPushButton("Sagittal")
        views_box = QtWidgets.QGroupBox("Views")
        views_box_layout = QtWidgets.QVBoxLayout()
        views_box_layout.addWidget(axial_view)
        views_box_layout.addWidget(coronal_view)
        views_box_layout.addWidget(sagittal_view)
        views_box.setLayout(views_box_layout)
        self.grid.addWidget(views_box, 4, 0, 2, 2)
        axial_view.clicked.connect(self.set_axial_view)
        coronal_view.clicked.connect(self.set_coronal_view)
        sagittal_view.clicked.connect(self.set_sagittal_view)

    def add_functions_settings_widget(self):
        """ option to load file, segment and predict"""
        load_brain_button = QtWidgets.QPushButton("Open brain MRI")
        load_mask_button = QtWidgets.QPushButton("Open mask")
        segment_button = QtWidgets.QPushButton("Segment tumor")
        predict_button = QtWidgets.QPushButton("Predict tumor growth")
        function_box = QtWidgets.QGroupBox("Utils")
        function_box_layout = QtWidgets.QGridLayout()
        function_box_layout.addWidget(load_brain_button, 0, 0)
        function_box_layout.addWidget(load_mask_button, 0, 1)
        function_box_layout.addWidget(segment_button, 1, 0)
        function_box_layout.addWidget(predict_button, 1, 1)
        function_box.setLayout(function_box_layout)
        self.grid.addWidget(function_box, 0, 0, 2, 2)
        load_brain_button.clicked.connect(self.load_brain_file)
        load_mask_button.clicked.connect(self.load_mask_file)
        segment_button.clicked.connect(self.segment_mask)
        predict_button.clicked.connect(self.predict_growth)

    def add_vtk_window_widget(self):
        """ grid to view brain and mask"""

        object_group_box = QtWidgets.QGroupBox()
        object_layout = QtWidgets.QVBoxLayout()
        object_layout.addWidget(self.vtk_widget)
        object_group_box.setLayout(object_layout)
        self.grid.addWidget(object_group_box, 0, 2, 12, 8)
        self.grid.setColumnMinimumWidth(2, 700)

    # FUNCTION SETTINGS
    def load_brain_file(self):
        # if self.brain
        brain_file = self.file_dialog.get_nii_dir()
        if self.brain_loaded:
            self.renderer.RemoveAllViewProps()
            self.reset_slicers()

        file_reader.renderer = self.renderer
        self.brain = file_reader.read_brain(brain_file)
        self.brain_slicer_props = file_reader.setup_slice(self.brain)
        self.renderer = file_reader.renderer
        self.set_axial_view()
        if not self.brain_loaded:
            self.add_brain_settings_widget()
            self.brain_loaded = True

        self.renderer.Render()

    def load_mask_file(self):
        mask_file = self.file_dialog.get_nii_dir()
        file_reader.renderer = self.renderer
        self.mask = file_reader.read_mask(mask_file)
        # checkbox error here
        self.renderer = file_reader.renderer
        self.renderer.Render()

    def segment_mask(self):
        ...

    def predict_growth(self):
        ...

    # BRAIN SETTINGS

    def brain_axial_slice_changed(self):
        pos = self.slicer_widgets[0].value()
        self.brain_slicer_props[0].SetDisplayExtent(self.brain.extent[0], self.brain.extent[1], self.brain.extent[2],
                                                    self.brain.extent[3], pos, pos)
        self.render_window.Render()

    def brain_coronal_slice_changed(self):
        pos = self.slicer_widgets[1].value()
        self.brain_slicer_props[1].SetDisplayExtent(self.brain.extent[0], self.brain.extent[1], pos, pos,
                                                    self.brain.extent[4], self.brain.extent[5])
        self.render_window.Render()

    def brain_sagittal_slice_changed(self):
        pos = self.slicer_widgets[2].value()
        self.brain_slicer_props[2].SetDisplayExtent(pos, pos, self.brain.extent[2], self.brain.extent[3],
                                                    self.brain.extent[4], self.brain.extent[5])
        self.render_window.Render()

    # MASK SETTINGS
    def mask_opacity_vc(self):
        opacity = round(self.mask_opacity_sp.value(), 2)
        for i, label in enumerate(self.mask.labels):
            if label.property and self.mask_label_cbs[i].isChecked():
                label.property.SetOpacity(opacity)
        self.render_window.Render()

    def mask_smoothness_vc(self):
        self.process_changes()
        smoothness = self.mask_smoothness_sp.value()
        for label in self.mask.labels:
            if label.smoother:
                label.smoother.SetNumberOfIterations(smoothness)
        self.render_window.Render()

    def mask_label_checked(self):
        ...

    # VIEWS SETTINGS
    def set_axial_view(self):
        self.renderer.ResetCamera()
        fp = self.renderer.GetActiveCamera().GetFocalPoint()
        p = self.renderer.GetActiveCamera().GetPosition()
        dist = math.sqrt((p[0] - fp[0]) ** 2 + (p[1] - fp[1])
                         ** 2 + (p[2] - fp[2]) ** 2)
        self.renderer.GetActiveCamera().SetPosition(fp[0], fp[1], fp[2] + dist)
        self.renderer.GetActiveCamera().SetViewUp(0.0, 1.0, 0.0)
        self.renderer.GetActiveCamera().Zoom(1.8)
        self.render_window.Render()

    def set_coronal_view(self):
        self.renderer.ResetCamera()
        fp = self.renderer.GetActiveCamera().GetFocalPoint()
        p = self.renderer.GetActiveCamera().GetPosition()
        dist = math.sqrt((p[0] - fp[0]) ** 2 + (p[1] - fp[1])
                         ** 2 + (p[2] - fp[2]) ** 2)
        self.renderer.GetActiveCamera().SetPosition(fp[0], fp[2] - dist, fp[1])
        self.renderer.GetActiveCamera().SetViewUp(0.0, 0.5, 0.5)
        self.renderer.GetActiveCamera().Zoom(1.8)
        self.render_window.Render()

    def set_sagittal_view(self):
        self.renderer.ResetCamera()
        fp = self.renderer.GetActiveCamera().GetFocalPoint()
        p = self.renderer.GetActiveCamera().GetPosition()
        dist = math.sqrt((p[0] - fp[0]) ** 2 + (p[1] - fp[1])
                         ** 2 + (p[2] - fp[2]) ** 2)
        self.renderer.GetActiveCamera().SetPosition(fp[2] + dist, fp[0], fp[1])
        self.renderer.GetActiveCamera().SetViewUp(0.0, 0.0, 1.0)
        self.renderer.GetActiveCamera().Zoom(1.6)
        self.render_window.Render()

    @staticmethod
    def create_new_separator():
        horizontal_line = QtWidgets.QWidget()
        horizontal_line.setFixedHeight(1)
        horizontal_line.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        horizontal_line.setStyleSheet("background-color: #c8c8c8;")
        return horizontal_line

    @staticmethod
    def create_new_picker(max_value, min_value, step, picker_value, value_changed_func):
        if isinstance(max_value, int):
            picker = QtWidgets.QSpinBox()
        else:
            picker = QtWidgets.QDoubleSpinBox()

        picker.setMaximum(max_value)
        picker.setMinimum(min_value)
        picker.setSingleStep(step)
        picker.setValue(picker_value)
        picker.valueChanged.connect(value_changed_func)
        return picker

    def process_changes(self):
        for _ in range(10):
            self.app.processEvents()
            time.sleep(0.1)
