from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QStackedLayout,
    QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel,
    QSizePolicy,
)

class FileInfoWidget(QWidget):
    _path: Path
    directory_selected = Signal(Path)
    directory_removed = Signal()

    def __init__(self, button_text, file_mode, parent=None):
        super().__init__(parent=parent)
        self._path = None

        self._layout = QStackedLayout()

        directory_file_dialog = QFileDialog(self, "Select Directory", fileMode=file_mode)
        directory_file_dialog.fileSelected.connect(self.on_directory_selected)

        open_file_dialog_button = QPushButton(button_text)
        open_file_dialog_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        open_file_dialog_button.clicked.connect(directory_file_dialog.exec)

        directory_select_widget = QWidget()
        directory_select_layout = QVBoxLayout()
        directory_select_layout.addWidget(open_file_dialog_button)
        directory_select_widget.setLayout(directory_select_layout)

        self._layout.addWidget(directory_select_widget)

        info_widget = QWidget()
        info_layout = QVBoxLayout()
        self.path_label = QLabel()
        self.info_label = QLabel()

        button_section = QWidget()
        self.remove_path_button = QPushButton("Remove Path")
        self.remove_path_button.clicked.connect(self.on_remove_path_button_clicked)
        self.change_path_button = QPushButton("Change Path")
        self.change_path_button.clicked.connect(directory_file_dialog.exec)
        button_section_layout = QHBoxLayout()
        button_section_layout.addWidget(self.change_path_button)
        button_section_layout.addWidget(self.remove_path_button)
        button_section.setLayout(button_section_layout)

        info_layout.addWidget(self.path_label)
        info_layout.addWidget(self.info_label)
        info_layout.addWidget(button_section)
        info_widget.setLayout(info_layout)

        self._layout.addWidget(info_widget)

        self.setLayout(self._layout)
        self._layout.setCurrentIndex(0)

    def on_directory_selected(self, path):
        if path:
            self._path = Path(path)
            self.path_label.setText(self._path.as_posix())
            self._layout.setCurrentIndex(1)
            self.directory_selected.emit(self._path)

    def on_remove_path_button_clicked(self):
        self._path = None
        self._layout.setCurrentIndex(0)
        self.directory_removed.emit()
