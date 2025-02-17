from PySide6.QtWidgets import (
    QMainWindow, QWidget, QSplitter,
    QVBoxLayout, QStackedLayout,
    QListWidget, QListWidgetItem,
    QFileDialog,
    QSizePolicy,
)

from data_manager.dataset_section.dataset_section import DatasetWidget
from data_manager.training_section.training_section import TrainingWidget
from data_manager.file_info_widget import FileInfoWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Data Manager")

        #### GUI ####
        #### Navigation ####
        self.navigation_widget = QListWidget()
        self.navigation_widget.addItem("Dataset")
        self.navigation_widget.addItem("Training")
        self.navigation_widget.addItem("Classification")
        self.navigation_widget.itemClicked.connect(self.on_navigation_item_clicked)

        self.database_info = FileInfoWidget("Select Database Directory", file_mode=QFileDialog.Directory)
        self.database_info.directory_selected.connect(self.on_database_directory_selected)
        self.database_info.directory_removed.connect(self.on_database_directory_removed)
        self.model_info = FileInfoWidget("Select Model", file_mode=QFileDialog.AnyFile)
        self.model_info.directory_selected.connect(self.on_model_selected)
        self.model_info.directory_removed.connect(self.on_model_removed)

        sidebar_layout = QVBoxLayout()
        sidebar_layout.addWidget(self.navigation_widget)
        sidebar_layout.addWidget(self.database_info)
        sidebar_layout.addWidget(self.model_info)

        sidebar = QWidget()
        sidebar.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        sidebar.setLayout(sidebar_layout)

        #### Tool Section ####        
        ## Track Management Section ##
        ## Table Section ##
        self.dataset_section = DatasetWidget(parent=self)
        self.dataset_section.data_changed.connect(self.on_data_changed)

        self.training_section = TrainingWidget(parent=self)

        self.tool_section_layout = QStackedLayout()
        self.tool_section_layout.addWidget(self.dataset_section) # Dataset Section
        self.tool_section_layout.addWidget(self.training_section) # Training Section
        self.tool_section_layout.addWidget(QWidget()) # Classification Section

        tool_section = QWidget()
        tool_section.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        tool_section.setLayout(self.tool_section_layout)

        centralWidget = QSplitter()
        centralWidget.addWidget(sidebar)
        centralWidget.addWidget(tool_section)
        centralWidget.setStretchFactor(0, 1)
        centralWidget.setStretchFactor(1, 3)
        
        self.setCentralWidget(centralWidget)

    def on_navigation_item_clicked(self, item: QListWidgetItem):
        self.tool_section_layout.setCurrentIndex(self.navigation_widget.row(item))

    def on_database_directory_selected(self, path):
        self.dataset_section.database_selected(path)

    def on_database_directory_removed(self):
        self.dataset_section.database_removed()

    def on_data_changed(self, path, data):
        self.database_info.info_label.setText(self.dataset_section.info())
        self.training_section.on_data_changed(path, data)

    def on_model_selected(self, path):
        self.training_section.model_selected(path)
        self.model_info.info_label.setText(self.training_section.info())

    def on_model_removed(self):
        self.training_section.model_removed()
        self.model_info.info_label.setText("")