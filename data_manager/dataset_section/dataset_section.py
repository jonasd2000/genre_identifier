import os
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QSplitter,
    QVBoxLayout,
    QTableView, QPushButton,
    QMenu, QMessageBox,
    QAbstractItemView, QSizePolicy,
)

from data_manager.dataset_section.track_table_model import TrackTableModel, DATA_FILE_NAME
from data_manager.dataset_section.download_widget import DownloadWidget

import polars as pl


class DatasetWidget(QWidget):
    data_path: Path

    data_changed = Signal(Path, pl.DataFrame)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.data_path = None

        repair_database_button = QPushButton("Repair Database")
        repair_database_button.clicked.connect(self.repair_database)

        self.tracks_table = QTableView()
        self.tracks_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tracks_table.setSortingEnabled(True)
        self.tracks_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tracks_table.setHorizontalScrollMode(QTableView.ScrollPerPixel)
        self.tracks_table.setVerticalScrollMode(QTableView.ScrollPerItem)
        self.tracks_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tracks_table.customContextMenuRequested.connect(self.on_table_right_click)

        self.set_table_model(TrackTableModel())

        table_section_layout = QVBoxLayout()
        table_section_layout.addWidget(repair_database_button)
        table_section_layout.addWidget(self.tracks_table)

        table_section = QWidget()
        table_section.setLayout(table_section_layout)

        self.download_section = DownloadWidget()
        self.download_section.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.download_section.item_download_started.connect(self.on_track_registered)
        self.download_section.item_download_finished.connect(self.on_track_file_added)
        self.data_changed.connect(self.download_section.on_data_changed)

        track_management_section = QSplitter()
        track_management_section.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        track_management_section.addWidget(table_section)
        track_management_section.addWidget(self.download_section)
        track_management_section.setStretchFactor(0, 3)
        track_management_section.setStretchFactor(1, 1)

        layout = QVBoxLayout()
        layout.addWidget(track_management_section)

        self.setLayout(layout)
        
    def info(self):
        return f"Tracks: {self.tracks_table_model._data.shape[0]}\nGenres: {self.tracks_table_model._data['genre'].n_unique()}"

    def set_table_model(self, model: TrackTableModel):
        model.dataChanged.connect(self.on_database_modified)
        model.rowsInserted.connect(self.on_database_modified)
        model.rowsRemoved.connect(self.on_database_modified)

        self.tracks_table_model = model
        self.tracks_table.setModel(model)

    def database_selected(self, path):
        self.data_path = Path(path)

        self.set_table_model(TrackTableModel.from_path(self.data_path))

        data = self.tracks_table_model.get_dataframe()
        self.tracks_table_model.add_items_to_table(data)
        self.data_changed.emit(self.data_path, data)

    def database_removed(self):
        self.data_path = None
        self.set_table_model(TrackTableModel())

        self.data_changed.emit(self.data_path, self.tracks_table_model.get_dataframe())

    def on_track_registered(self, video_url: str, genre: str, is_training_data: bool):
        self.tracks_table_model.add_items_to_table(pl.DataFrame({
            "source": video_url, 
            "filepath": None, 
            "genre": genre, 
            "is_training_data": is_training_data
        }))

    def on_track_file_added(self, video_url: str, file_path: str):
        self.tracks_table_model.set_filepath(video_url, file_path)

    def on_database_modified(self, *args, **kwargs):
        self.tracks_table_model.save(self.data_path / DATA_FILE_NAME)
        self.data_changed.emit(self.data_path, self.tracks_table_model.get_dataframe())

    def on_table_right_click(self, pos):
        item_index = self.tracks_table.indexAt(pos)
        item_index.row()

        menu = QMenu()
        delete_action = menu.addAction("Delete")
        action = menu.exec(self.tracks_table.mapToGlobal(pos))

        match action:
            case action if action == delete_action:
                selected_rows = self.tracks_table.selectionModel().selectedRows()
                sorted_rows = sorted(selected_rows, key=lambda x: x.row(), reverse=True)
                for row in sorted_rows:
                    self.tracks_table_model.removeRow(row.row())
            case _:
                pass

    def repair_database(self):
        # Message Box that asks the user if they want to repair the database
        data = self.tracks_table_model.get_dataframe()

        missing_files = data.filter(pl.col("filepath").is_null())
        existing_files = pl.from_dicts({
            "filepath": [str(self.data_path / file) for file in os.listdir(self.data_path)]
        }).filter(~pl.col("filepath").str.ends_with(".json"))
        files_without_entry = existing_files.filter(~pl.col("filepath").is_in(data["filepath"]))
        
        message = f"""
        This will delete all files not registered in the database and download files for all entries with missing files.
        This would result in downloading {len(missing_files)} files and deleting {len(files_without_entry)} files.
        This may take a while. Are you sure you want to continue?
        """
        reply = QMessageBox.question(self, "Database Repair", message, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

        match reply:
            case QMessageBox.StandardButton.Yes:
                for filepath in files_without_entry["filepath"]:
                    print(f"Deleting {filepath}")
                    os.remove(filepath)

                for row in missing_files.to_dicts():
                    self.download_section.download(download_url=row["source"], genre=row["genre"], is_training_data=row["is_training_data"])

            case _:
                return
