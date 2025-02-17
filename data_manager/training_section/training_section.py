from typing import Callable
from pathlib import Path

from PySide6.QtCore import Signal, QRunnable, QObject, QThreadPool
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout, QHBoxLayout, QStackedLayout, QFormLayout,
    QPushButton, QLabel, QDoubleSpinBox, QSpinBox, QCheckBox,
    QSizePolicy,
)
import pyqtgraph as pg

import torch
from torch.utils.data import DataLoader
import polars as pl

from neural_network.nn import GenreClassifier, train, test
from audio.dataset import TrackGenreDataset
from utility.torch import get_device

class Thread(QRunnable, QObject):
    thread_id: int
    func: Callable

    started: Signal = Signal(int)
    failed: Signal = Signal(int)
    finished: Signal = Signal(int)

    def __init__(self, thread_id: int, func: Callable, *args, **kwargs):
        QObject.__init__(self)
        QRunnable.__init__(self)

        self.thread_id = thread_id
        
        self.func = func
        self.args = args
        self.kwargs = kwargs
        
    def run(self):
        self.started.emit(self.thread_id)
        try:
            self.func(*self.args, **self.kwargs)
        except Exception as e:
            self.failed.emit(self.thread_id)
            return
        self.finished.emit(self.thread_id)

class TrainingWidget(QWidget):
    _model: GenreClassifier
    _model_path: Path

    data_path: Path
    _data: pl.DataFrame

    thread_pool: QThreadPool

    training_started = Signal()
    iteration_finished = Signal(float, float)
    training_stopped = Signal()

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self._model = None
        self._model_path = None

        self.data_path = None
        self._data = None

        self.thread_pool = QThreadPool()
        self.iterations = 0
        self.losses = []
        self.accuracies = []

        self._run_training = False
        self._trining_paused = False

        layout = QHBoxLayout()

        sidebar_layout = QVBoxLayout()
        sidebar = QWidget()
        sidebar.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        sidebar.setLayout(sidebar_layout)

        hyperparameters_widget = QWidget()
        hyperparameters_layout = QFormLayout()
        self.learning_rate_label = QLabel("Learning Rate (1e-4)")
        self.learning_rate_value = QDoubleSpinBox(minimum=1, maximum=1e6)
        self.batch_size_label = QLabel("Batch Size")
        self.batch_size_value = QSpinBox(minimum=2, maximum=128, value=32)
        self.read_to_memory_label = QLabel("Read to Memory")
        self.read_to_memory_checkbox = QCheckBox()
        hyperparameters_layout.addWidget(self.read_to_memory_label)
        hyperparameters_layout.addWidget(self.read_to_memory_checkbox)
        hyperparameters_layout.addWidget(self.batch_size_label)
        hyperparameters_layout.addWidget(self.batch_size_value)
        hyperparameters_layout.addWidget(self.learning_rate_label)
        hyperparameters_layout.addWidget(self.learning_rate_value)
        hyperparameters_widget.setLayout(hyperparameters_layout)

        self.train_button = QPushButton("Train")
        self.train_button.clicked.connect(self.on_train_button_clicked)
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.on_pause_button_clicked)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.on_stop_button_clicked)

        pause_stop_widget = QWidget()
        pause_stop_layout = QHBoxLayout()
        pause_stop_layout.addWidget(self.pause_button)
        pause_stop_layout.addWidget(self.stop_button)
        pause_stop_widget.setLayout(pause_stop_layout)

        self.buttons_widget = QWidget()
        buttons_widget_layout = QStackedLayout()
        buttons_widget_layout.addWidget(self.train_button)
        buttons_widget_layout.addWidget(pause_stop_widget)
        self.buttons_widget.setLayout(buttons_widget_layout)
        sidebar_layout.addWidget(hyperparameters_widget)
        sidebar_layout.addWidget(self.buttons_widget)

        self.plot_widget = pg.PlotWidget()
        
        layout.addWidget(self.plot_widget)
        layout.addWidget(sidebar)
        self.setLayout(layout)

        self.training_started.connect(self.on_training_started)
        self.iteration_finished.connect(self.on_iteration_finished)
        self.training_stopped.connect(self.on_training_interrupted)
    
    def info(self) -> str:
        return str(self._model)

    def set_model(self, model: GenreClassifier):
        self._model = model

    def set_model_path(self, path: Path):
        self._model_path = path

    def model_selected(self, path: Path):
        model = None
        if not path.exists():
            # create new model
            model = GenreClassifier()
        else:
            # load model
            model = torch.load(path, weights_only=False)

        self.set_model(model)
        self.add_data_genres_to_model()
        self.set_model_path(path)

    def model_removed(self):
        torch.save(self._model, self._model_path)
        self.set_model(None)
        self.set_model_path(None)

    def on_data_changed(self, path: Path, data: pl.DataFrame):
        self.data_path = path
        self._data = data
        self.add_data_genres_to_model()

    def database_removed(self):
        self.data_path = None

    def add_data_genres_to_model(self):
        if self._model is None or self._data is None:
            return
        for genre in self._data["genre"].unique():
            self._model.add_genre(genre)

    def train_model(self):
        training_dataloader = self.get_training_dataloader()
        testing_dataloader = self.get_testing_dataloader()
        device = get_device()
        learning_rate = self.learning_rate_value.value() * 1e-4

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self._model.parameters(), lr=learning_rate)

        self._model.to(device)
        thread = Thread(0, TrainingWidget.train_loop, self, training_dataloader, testing_dataloader, loss_fn, optimizer, device)
        thread.started.connect(self.on_training_started)
        thread.finished.connect(self.on_training_interrupted)
        self.thread_pool.start(thread)

    def train_loop(self, training_dataloader, testing_dataloader, loss_fn, optimizer, device):
        while self._run_training:
            if self._training_paused:
                continue
            train(training_dataloader, self._model, loss_fn, optimizer, device)
            accuracy, avg_loss = test(testing_dataloader, self._model, loss_fn, device)
            self.iteration_finished.emit(avg_loss, accuracy)

    def on_iteration_finished(self, avg_loss, accuracy):
        self.iterations += 1
        self.losses.append(avg_loss)
        self.accuracies.append(accuracy)
        self.plot_widget.clear()
        self.plot_widget.plot(list(range(self.iterations)), self.losses)
        self.plot_widget.plot(list(range(self.iterations)), self.accuracies)

    def on_train_button_clicked(self):
        self._run_training = True
        self._training_paused = False
        self.train_model()

    def on_pause_button_clicked(self):
        self._training_paused = not self._training_paused
        self.pause_button.setText("Resume" if self._training_paused else "Pause")

    def on_stop_button_clicked(self):
        self._training_paused = False
        self._run_training = False

    def on_training_started(self):
        self.iterations = 0
        self.losses = []
        self.accuracies = []
        self.buttons_widget.layout().setCurrentIndex(1)

    def on_training_interrupted(self):
        self.buttons_widget.layout().setCurrentIndex(0)
    
    def get_training_dataloader(self):
        df = self._data.filter(pl.col("is_training_data"))
        batch_size = self.batch_size_value.value()
        read_to_memory = self.read_to_memory_checkbox.isChecked()

        dataset = TrackGenreDataset(self.data_path, df, self._model.genre_map, 131_072, read_to_memory)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def get_testing_dataloader(self):
        df = self._data.filter(~pl.col("is_training_data"))
        batch_size = self.batch_size_value.value()
        read_to_memory = self.read_to_memory_checkbox.isChecked()

        dataset = TrackGenreDataset(self.data_path, df, self._model.genre_map, 131_072, read_to_memory)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    