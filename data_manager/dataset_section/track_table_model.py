import os
from pathlib import Path

from PySide6.QtCore import QAbstractTableModel, Qt, QModelIndex

import polars as pl


DATA_FILE_NAME = "data.json"
data_schema = pl.Schema({"source": str, "filepath": str, "genre": str, "is_training_data": bool})

class TrackTableModel(QAbstractTableModel):
    def __init__(self, data=None):
        super().__init__()
        self._data: pl.DataFrame = data if data is not None else pl.DataFrame(schema=data_schema)

    @staticmethod
    def from_path(path: Path) -> "TrackTableModel":
        database_filepath = path / DATA_FILE_NAME
        if not database_filepath.is_file():
            with open(database_filepath, "w") as f:
                f.write("[]")

        data = pl.read_json(database_filepath, schema=data_schema)
        validated_data = TrackTableModel.validate_data(data, path)

        return TrackTableModel(validated_data)

    @staticmethod
    def validate_data(data: pl.DataFrame, data_path: Path):
        existing_files = pl.from_dicts({"filepath": [str(data_path / file) for file in os.listdir(data_path)]}).filter(~pl.col("filepath").str.ends_with(".json"))
        files_without_entry = existing_files.filter(~pl.col("filepath").is_in(data["filepath"]))
        entries_without_file = data.filter(pl.col("filepath").is_null() | ~data["filepath"].is_in(existing_files["filepath"]))

        data = data.with_columns(
            filepath=pl.when(pl.col("filepath").is_in(entries_without_file["filepath"])).then(None).otherwise(pl.col("filepath")),
        )

        return data

    def get_dataframe(self):
        return self._data.clone()

    def save(self, path: Path):
        self._data.write_json(path)

    def headerData(self, section, orientation, role):
        match role:
            case Qt.DisplayRole:
                if orientation == Qt.Horizontal:
                    return list(self._data.schema.keys())[section]

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        match role:
            case Qt.DisplayRole:
                return str(self._data[index.row(), index.column()])

    def add_items_to_table(self, items: pl.DataFrame):
        self.beginInsertRows(QModelIndex(), self.rowCount(), self.rowCount() + items.shape[0] - 1)
        self._data = self._data.vstack(items).unique(keep="last", subset=["source", "is_training_data"])
        self.endInsertRows()

    def removeRow(self, row: int, parent=None):
        data_with_row_index = self._data.with_row_index()
        item = data_with_row_index.filter(pl.col("index") == row).to_dicts()[0]
        source = item["source"]
        filepath = item["filepath"]

        self.beginRemoveRows(QModelIndex(), row, row)
        if filepath is not None:
            os.remove(filepath)
        self._data = self._data.filter(~(pl.col("source") == source))
        self.endRemoveRows()

    def set_filepath(self, source: str, filepath: str):
        filepath_column_index = 1
        data_with_row_index = self._data.with_row_index()
        row_index = data_with_row_index.filter(pl.col("source") == source).to_dicts()[0]["index"]
        self._data[row_index, filepath_column_index] = filepath

        self.dataChanged.emit(self.index(row_index, filepath_column_index), self.index(row_index, filepath_column_index))
