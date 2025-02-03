import os
import sys

from PySide6.QtWidgets import QApplication

from data_manager.app import MainWindow


def main():
    os.environ["QT_SCALE_FACTOR"] = "1.25"
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    app.exec()

if __name__ == "__main__":
    main()