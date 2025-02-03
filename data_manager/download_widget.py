from typing import Dict, Any

from PySide6.QtCore import QObject, QRunnable, Signal, QThreadPool
from PySide6.QtWidgets import (
    QWidget, QComboBox, QCheckBox, QLineEdit, QLabel, QPushButton,
    QListWidget, QListWidgetItem,
    QVBoxLayout, QFormLayout,
)

import yt_dlp

class YTDLPThread(QRunnable, QObject):
    batch_id: int

    started: Signal = Signal(int)
    failed: Signal = Signal(int)
    finished: Signal = Signal(int)

    batch_download_started: Signal = Signal(int, int) # number of tracks in playlist
    individual_download_started: Signal = Signal(int, str, int) # url, genre, is_training_data, playlist_index
    individual_download_finished: Signal = Signal(str, str) # url, file_path

    def __init__(self, batch_id: int, batch_url: str, path: str):
        QObject.__init__(self)
        QRunnable.__init__(self)

        self.batch_id = batch_id
        self.url = batch_url
        self.path = path
        
        self.video_url = None
        self.last_video_url = None

    def run(self):
        self.started.emit(self.batch_id)
        ydl_opts = {
            "paths": {"home": self.path},
            "ignoreerrors": "only_download",
            "format": "bestaudio/best",
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "aac",
            }],
            "progress_hooks": [self.on_download_progress],
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.add_post_hook(self.on_individual_download_finish)
                ydl.download([self.url])
        except Exception as e:
            self.failed.emit(self.batch_id)
            return
        
        self.finished.emit(self.batch_id)

    def on_download_progress(self, info_dict: Dict[str, Any]):
        self.video_url = info_dict["info_dict"]["original_url"]
        match info_dict["status"]:
            case "downloading":
                if self.video_url != self.last_video_url:
                    # if this is the first download, emit the first_item_download_started signal
                    if self.last_video_url is None:
                        n_entries = info_dict["info_dict"]["n_entries"] if "n_entries" in info_dict["info_dict"] else 1
                        self.batch_download_started.emit(self.batch_id, n_entries)
                    
                    index = info_dict["info_dict"]["playlist_index"] or 1
                    self.individual_download_started.emit(self.batch_id, self.video_url, index)

                    self.last_video_url = self.video_url

    def on_individual_download_finish(self, file_path: str):
        self.individual_download_finished.emit(self.video_url, file_path)

class DownloadWidget(QWidget):
    item_download_started: Signal = Signal(str, str, bool)
    item_download_finished: Signal = Signal(str, str)

    def __init__(self):
        super().__init__()

        self.threadpool = QThreadPool()
        self.download_items = {}
        self.data_path = None

        self.genre_combobox = QComboBox(editable=True)
        self.genre_combobox.setInsertPolicy(QComboBox.InsertPolicy.InsertAlphabetically)
        self.genre_combobox.currentTextChanged.connect(self.enable_download_button)

        self.is_training_data_checkbox = QCheckBox()

        self.url_line_edit = QLineEdit(clearButtonEnabled=True)
        self.url_line_edit.textChanged.connect(self.enable_download_button)

        track_details_section_layout = QFormLayout()
        track_details_section_layout.addWidget(QLabel("Genre:"))
        track_details_section_layout.addWidget(self.genre_combobox)
        track_details_section_layout.addWidget(QLabel("Is Training Data:"))
        track_details_section_layout.addWidget(self.is_training_data_checkbox)
        track_details_section_layout.addWidget(QLabel("URL:"))
        track_details_section_layout.addWidget(self.url_line_edit)

        track_details_section = QWidget()
        track_details_section.setLayout(track_details_section_layout)

        ## Button Section ##
        self.download_button = QPushButton("Download")
        self.download_button.setEnabled(False)
        self.download_button.clicked.connect(self.on_download_button_clicked)

        ## Download Status Section ##
        self.download_status_list = QListWidget()
        self.download_status_list.itemDoubleClicked.connect(lambda item: self.download_status_list.takeItem(self.download_status_list.row(item)))

        download_section_layout = QVBoxLayout()
        download_section_layout.addWidget(track_details_section)
        download_section_layout.addWidget(self.download_button)
        download_section_layout.addWidget(self.download_status_list)
        
        self.setLayout(download_section_layout)
    
    def data_selected(self, data_path, data):
        self.data_path = data_path
        combobox_text = self.genre_combobox.currentText()
        self.genre_combobox.clear()
        self.genre_combobox.addItems([""] + list(data["genre"].unique()))
        self.genre_combobox.setCurrentText(combobox_text)
        self.enable_download_button()

    def enable_download_button(self):
        if (self.data_path is not None) and (self.genre_combobox.currentText() != "") and (self.url_line_edit.text() != ""):
            self.download_button.setEnabled(True)
        else:
            self.download_button.setEnabled(False)

    def on_batch_download_started(self, batch_id: int, n_entries: int):
        print("Calling on_batch_download_started")
        self.download_items[batch_id]["widget"].setText(f"Downloading {n_entries} tracks from {self.download_items[batch_id]['url']}...")
        self.download_items[batch_id]["n_entries"] = n_entries

    def on_download_progress(self, info_dict: Dict[str, Any]):
        ...

    def on_individual_download_started(self, batch_id: int, video_url: str, playlist_index: int):
        self.download_items[batch_id]["widget"].setText(f"Downloading track {playlist_index} of {self.download_items[batch_id]['n_entries']} from {self.download_items[batch_id]['url']}...")
        self.item_download_started.emit(video_url, self.download_items[batch_id]["genre"], self.download_items[batch_id]["is_training_data"])

    def on_individual_download_finish(self, video_url: str, file_path: str):
        self.item_download_finished.emit(video_url, file_path)

    def on_batch_download_finished(self, batch_id: int):
        widget_item = self.download_items[batch_id]["widget"]
        widget_item.setText(f"Downloaded {self.download_items[batch_id]['n_entries']} tracks from {self.download_items[batch_id]['url']}!")

    def on_download_button_clicked(self):
        self.download(self.url_line_edit.text(), self.genre_combobox.currentText(), self.is_training_data_checkbox.isChecked())

    def download(self, download_url: str, genre: str, is_training_data: bool):
        batch_id = max(self.download_items.keys() or {0}) + 1

        widget_item = QListWidgetItem(f"Starting download from {download_url}...")
        self.download_items[batch_id] = {}
        self.download_items[batch_id]["url"] = download_url
        self.download_items[batch_id]["widget"] = widget_item
        self.download_items[batch_id]["genre"] = genre
        self.download_items[batch_id]["is_training_data"] = is_training_data
        self.download_status_list.addItem(widget_item)

        ytdlp_thread = YTDLPThread(
            batch_id         = batch_id,
            batch_url        = download_url,
            path             = str(self.data_path),
        )
        ytdlp_thread.batch_download_started.connect(self.on_batch_download_started)
        ytdlp_thread.individual_download_started.connect(self.on_individual_download_started)
        ytdlp_thread.individual_download_finished.connect(self.on_individual_download_finish)
        ytdlp_thread.finished.connect(self.on_batch_download_finished)
        
        self.threadpool.start(ytdlp_thread)
