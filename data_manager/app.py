import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QSplitter,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QAbstractItemView,
    QFileDialog, QMenu, QMessageBox,
    QHBoxLayout, QVBoxLayout,
    QSizePolicy,
)

import polars as pl

from data_manager.data_manager import DataManager, data_schema
from data_manager.download_widget import DownloadWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.data_manager = None

        self.setWindowTitle("Data Manager")

        #### GUI ####
        ## Directory Section ##
        directory_file_dialog = QFileDialog(self, "Select Directory", fileMode=QFileDialog.Directory)
        directory_file_dialog.fileSelected.connect(self.on_directory_selected)

        open_file_dialog_button = QPushButton("Select Directory")
        open_file_dialog_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        open_file_dialog_button.clicked.connect(directory_file_dialog.exec)

        self.directory_label = QLabel("Select a directory to start working with.")
        self.directory_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        
        directory_section_layout = QHBoxLayout()
        directory_section_layout.addWidget(open_file_dialog_button)
        directory_section_layout.addWidget(self.directory_label)

        directory_section = QWidget()
        directory_section.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        directory_section.setLayout(directory_section_layout)

        ## Track Management Section ##
        ## Table Section ##
        repair_database_button = QPushButton("Repair Database")
        repair_database_button.clicked.connect(self.repair_database)

        self.tracks_table = QTableWidget(0, 5)
        self.tracks_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tracks_table.setHorizontalHeaderLabels(data_schema.keys())
        self.tracks_table.setSortingEnabled(True)
        self.tracks_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tracks_table.setHorizontalScrollMode(QTableWidget.ScrollPerPixel)
        self.tracks_table.setVerticalScrollMode(QTableWidget.ScrollPerItem)
        self.tracks_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tracks_table.customContextMenuRequested.connect(self.on_table_right_click)

        table_section_layout = QVBoxLayout()
        table_section_layout.addWidget(repair_database_button)
        table_section_layout.addWidget(self.tracks_table)

        table_section = QWidget()
        table_section.setLayout(table_section_layout)

        self.download_section = DownloadWidget()
        self.download_section.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.download_section.item_download_started.connect(self.on_track_registered)
        self.download_section.item_download_finished.connect(self.on_track_file_added)

        track_management_section = QSplitter()
        track_management_section.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        track_management_section.addWidget(table_section)
        track_management_section.addWidget(self.download_section)

        ## Main Layout ##
        main_layout = QVBoxLayout()
        main_layout.addWidget(directory_section)
        main_layout.addWidget(track_management_section)

        centralWidget = QWidget()
        centralWidget.setLayout(main_layout)
        
        self.setCentralWidget(centralWidget)

    def get_row_index_from_track_id(self, track_id: int) -> int:
        for i in range(self.tracks_table.rowCount() - 1, -1, -1):
            if int(self.tracks_table.item(i, 0).text()) == track_id:
                return i
            
        return None

    def on_directory_selected(self, path):
        try:
            self.data_manager = DataManager(path)
        except Exception as e:
            print(e)
            return

        self.directory_label.setText(self.data_manager.info())
        self.tracks_table.setRowCount(0)
        self.add_tracks_to_table(self.data_manager.data)
        self.download_section.data_selected(self.data_manager.path, self.data_manager.data)

    def add_tracks_to_table(self, tracks: pl.DataFrame):
        # when sorting is enabled, rows are sorted the moment an item is inserted
        # leading to unexpected results
        sorting_enabled = self.tracks_table.isSortingEnabled()
        self.tracks_table.setSortingEnabled(False)
        
        for track_dict in tracks.to_dicts():
            row_index = self.get_row_index_from_track_id(track_dict["id"])
            if row_index is None: # if track is not in table
                row_index = 0
                self.tracks_table.insertRow(row_index)
            # add track info to table
            for i, (key, value) in enumerate(track_dict.items()):
                self.tracks_table.setItem(row_index, i, QTableWidgetItem(str(value)))
            self.tracks_table.resizeColumnsToContents()

        # restore sorting
        self.tracks_table.setSortingEnabled(sorting_enabled)

    def on_track_registered(self, video_url: str, genre: str, is_training_data: bool):
        track, inserted = self.data_manager.add_track(video_url, None, genre, is_training_data)
        self.add_tracks_to_table(track)

    def on_track_file_added(self, video_url: str, file_path: str):
        track = self.data_manager.data.row(by_predicate=pl.col("source") == video_url, named=True)
        track_id = track["id"]
        self.data_manager.data = self.data_manager.data.with_columns(
            filepath=pl.when(pl.col("id") == track_id).then(pl.lit(file_path)).otherwise(pl.col("filepath")),
        )

        table_row_index = self.get_row_index_from_track_id(track_id)
        self.tracks_table.setItem(table_row_index, 2, QTableWidgetItem(file_path))
        self.directory_label.setText(self.data_manager.info())

    def on_table_right_click(self, pos):
        item_index = self.tracks_table.indexAt(pos)
        item_index.row()

        menu = QMenu()
        delete_action = menu.addAction("Delete")
        action = menu.exec(self.tracks_table.mapToGlobal(pos))

        match action:
            case action if action == delete_action:
                selected_rows = self.tracks_table.selectionModel().selectedRows()
                row_indexes = [row.row() for row in selected_rows]
                track_ids = [int(self.tracks_table.item(row_index, 0).text()) for row_index in row_indexes]

                self.data_manager.delete_tracks(track_ids=track_ids)
                for row_index in sorted(row_indexes, reverse=True):
                    self.tracks_table.removeRow(row_index)
                self.directory_label.setText(self.data_manager.info())
            case _:
                pass

    def repair_database(self):
        # Message Box that asks the user if they want to repair the database
        missing_files = self.data_manager.data.filter(pl.col("filepath").is_null())
        existing_files = pl.from_dicts({"filepath": [str(self.data_manager.path / file) for file in os.listdir(self.data_manager.path)]}).filter(~pl.col("filepath").str.ends_with(".json"))
        files_without_entry = existing_files.filter(~pl.col("filepath").is_in(self.data_manager.data["filepath"]))
        
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

        