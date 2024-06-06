import sys
import os
import time
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QPixmap, QMovie
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QHBoxLayout, QComboBox, QSlider, QPushButton
from datetime import datetime, timedelta
import warnings
import imageio
from PIL import Image
import torch
from models.model import Model
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")


class SatelliteApp(QtWidgets.QWidget):
    def __init__(self, every=10, days=3):
        super().__init__()

        self.parameters = ['Height', 'Radius', 'SZA', 'SAz', 'ST', 'n']
        self.param_lims = {'Height': {'min': 455, 'max': 520},
                           'Radius': {'min': 6825, 'max': 6875},
                           'SZA': {'min': 0, 'max': 180},
                           'SAz': {'min': -180, 'max': 180},
                           'ST': {'min': 0, 'max': 25},
                           'n': {'min': 0, 'max': 4e6}}
        self.data = self.load_data(n=days, skip_existing=True)
        self.every = every
        self.days = days

        self.cache_dir = 'cache'
        self.pred_dir = 'predictions'
        self.gif_dir = 'gifs'

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        if not os.path.exists(self.gif_dir):
            os.makedirs(self.gif_dir)
        if not os.path.exists(self.pred_dir):
            os.makedirs(self.pred_dir)

        self.dir_up = True
        self.filename = None
        self.pred_mode = False

        self.models = {}
        self.means = (pd.read_csv('mean_values_0.csv'), pd.read_csv('mean_values_1.csv'))

        self.init_ui()
        self.init_models()
        self.cache_plots()
        self.cache_predictions()
        self.update_plot()

    def init_models(self):
        for parameter in self.parameters:
            # if os.path.exists(f'models/{parameter}.pth'):
            #     self.models[parameter] = [Models[parameter](), 'tf']
            #     self.models[parameter][0].load_state_dict(
            #         torch.load(f'models/{parameter}.pth', map_location=torch.device('cpu')))
            #     self.models[parameter][0].eval()
            # # elif os.path.exists(f'models/{parameter}.cmb'):
            # #     self.models[parameter] = [Models[parameter](), 'cb']
            # #     self.models[parameter][0].load_model(f'models/{parameter}.cbm')
            if os.path.exists(f'models/{parameter}.pth'):
                self.models[parameter] = Model(f'models/{parameter}.pth')
            elif os.path.exists(f'models/{parameter}.cbm'):
                self.models[parameter] = Model(f'models/{parameter}.cbm')
            else:
                raise ValueError

    def init_ui(self):
        self.setWindowTitle('Satellite Data Map')
        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumSize(800, 600)

        # Dropdown for parameter selection
        self.parameter_dropdown = QComboBox(self)
        self.parameter_dropdown.addItems(self.parameters)
        self.parameter_dropdown.currentIndexChanged.connect(lambda: self.update_plot())

        # Slider for date selection
        self.date_slider = QSlider(QtCore.Qt.Horizontal, self)
        self.date_slider.setMinimum(0)
        self.date_slider.setMaximum(self.days - 1)
        self.date_slider.setValue(0)
        self.date_slider.valueChanged.connect(lambda: self.update_plot())

        # Button to generate GIF
        self.gif_button = QPushButton("Generate GIF", self)
        self.gif_button.clicked.connect(self.generate_gif)
        self.gif_button.setObjectName("gif_button")

        # Button to show prediction
        self.mode_button = QPushButton("Show Predictions", self)
        self.mode_button.clicked.connect(self.toggle_mode)
        self.mode_button.setObjectName("mode_button")

        # Arrow button
        self.arrow_button = QPushButton("↑", self)
        self.arrow_button.clicked.connect(self.toggle_dir)
        self.arrow_button.setObjectName("arrow_button")

        self.arrow_button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.gif_button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.mode_button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        # Label for displaying the cached plots
        self.image_label = QLabel(self)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setScaledContents(True)
        self.image_label.setStyleSheet("background-color: #eee;")

        # Layout
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.parameter_dropdown)
        control_layout.addWidget(self.date_slider)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.gif_button)
        button_layout.addWidget(self.mode_button)
        button_layout.addWidget(self.arrow_button)

        main_layout = QVBoxLayout()
        main_layout.addLayout(control_layout)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.image_label)

        self.setLayout(main_layout)

        # Apply modern styling
        self.setStyleSheet("""
            QWidget {
                font-size: 14px;
                font-family: Arial, sans-serif;
                letter-spacing: 1px;
            }
            QComboBox {
                padding: 10px;
                margin: 5px;
                border-radius: 5px;
                background-color: #ddd;
            }
            QComboBox::drop-down {
                border: 0px;
            }
            QComboBox::down-arrow {
                image: none;
            }
            QSlider::groove:horizontal {
                border: 1px solid #bbb;
                background: #eee;
                height: 10px;
                border-radius: 5px;
            }
            QSlider::sub-page:horizontal {
                background: #3a87ad;
                border: 1px solid #3a87ad;
                height: 10px;
                border-radius: 5px;
            }
            QSlider::add-page:horizontal {
                background: #eee;
                border: 1px solid #777;
                height: 10px;
                border-radius: 5px;
            }
            QSlider::handle:horizontal {
                background: #3a87ad;
                border: 1px solid #3a87ad;
                width: 20px;
                height: 20px;
                margin-top: -6px;
                margin-bottom: -6px;
                border-radius: 10px;
            }
            QSlider::handle:horizontal:hover {
                background: #2e6e9e;
                border: 1px solid #2e6e9e;
            }
            QPushButton {
                padding: 10px;
                margin: 5px;
                border-radius: 5px;
                background-color: #ddd;
            }
            QPushButton#arrow_button {
                width: 40px;
                height: 40px;
                padding: 0px;
                font-size: 16px;
                background-color: #ddd;
            }
            QPushButton:hover {
                background-color: #3a87ad;
                color: white;
            }
            QPushButton#arrow_button:hover {
                background-color: #3a87ad;
                color: white;
            }
            QHBoxLayout {
                display: flex;
                justify-content: space-between;
            }
            QLabel {
                padding: 10px;
                margin: 5px;
                border: 1px solid #ccc;
                border-radius: 5px;
            }
        """)

        self.update_plot()

    def get_dataframe(self, filepath):
        return pd.read_csv(filepath, skiprows=3, delim_whitespace=True).drop(
            columns=['Te_hgn', 'Te_lgn', 'T_elec', 'Timestamp'])

    def get_date(self, day_of_year):
        base_date = datetime(2023, 1, 1)
        target_date = base_date + timedelta(days=day_of_year)
        return target_date.strftime('%Y_%m_%d')

    def load_data(self, n=365, skip_existing=True):
        all_files = glob.glob("./Swarm/2023*.txt")  # Assuming all files are in the 'data' folder
        start_time = time.time()
        all_data = {}
        for i in range(n):
            date = all_files[i][-14:-4]
            if skip_existing:
                skip = True
                for parameter in self.parameters:
                    img_filename = f"cache/{date}_{parameter}.png"
                    if not os.path.exists(img_filename):
                        skip = False
                        break
                if skip:
                    continue
            df = self.get_dataframe(f'./Swarm/{date}.txt')
            all_data[date] = df

            elapsed_time = time.time() - start_time
            average_time = elapsed_time / (i + 1)
            print(
                f'\rLoaded:    {i + 1}/{n} ({(i + 1) / n * 100:.2f}%)\tElapsed: {elapsed_time:.2f}s\tERT: {average_time * (n - i - 1):.2f}s',
                end='')
        print(f'\nAll files loaded', end='')
        return all_data

    def toggle_dir(self):
        self.dir_up = not self.dir_up
        self.arrow_button.setText("↑" if self.dir_up else "↓")
        if self.pred_mode:
            if self.filename and self.filename.endswith('.gif'):
                self.filename = self.filename.replace('down' if self.dir_up else 'up', 'up' if self.dir_up else 'down')
                self.play_gif(self.filename)
            else:
                self.update_plot()

    def toggle_mode(self):
        self.pred_mode = not self.pred_mode
        color = '#c03a63' if self.pred_mode else '#3a87ad'
        dark_color = '#ab3458' if self.pred_mode else '#2e6e9e'

        self.mode_button.setText("Show Data" if self.pred_mode else "Show Predictions")

        buttons = [self.mode_button, self.arrow_button, self.gif_button]

        for button in buttons:
            button.setStyleSheet(f"""
            QPushButton {{
                padding: 10px;
                margin: 5px;
                border-radius: 5px;
                background-color: #ddd;
                height: 18px;
                {'width: 18px;' if button == self.arrow_button else ''}
            }}
            QPushButton:hover {{
                background-color: {color};
                color: white;
            }}
            """)

        self.date_slider.setStyleSheet(f"""
        QSlider::groove:horizontal {{
                border: 1px solid #bbb;
                background: #eee;
                height: 10px;
                border-radius: 5px;
            }}
            QSlider::sub-page:horizontal {{
                background: {color};
                border: 1px solid {color};
                height: 10px;
                border-radius: 5px;
            }}
            QSlider::add-page:horizontal {{
                background: #eee;
                border: 1px solid #777;
                height: 10px;
                border-radius: 5px;
            }}
            QSlider::handle:horizontal {{
                background: {color};
                border: 1px solid {color};
                width: 20px;
                height: 20px;
                margin-top: -6px;
                margin-bottom: -6px;
                border-radius: 10px;
            }}
            QSlider::handle:horizontal:hover {{
                background: {dark_color};
                border: 1px solid {dark_color};
            }}
        """)
        self.update_plot()

    def play_gif(self, gif_filename):
        # self.update_plot()

        movie = QMovie(gif_filename)
        self.image_label.setMovie(movie)

        label_size = self.image_label.size()
        if not label_size.isEmpty():
            movie.setScaledSize(label_size.scaled(label_size, QtCore.Qt.KeepAspectRatio))

        self.filename = gif_filename
        movie.start()

    def generate_gif(self):
        parameter = self.parameter_dropdown.currentText()
        gif_filename = f"{self.gif_dir}/{parameter}_{'up' if self.dir_up else 'down'}.gif" if self.pred_mode else f"{self.gif_dir}/{parameter}.gif"

        if not os.path.exists(gif_filename):
            images = []
            for i in range(self.days):
                date = self.get_date(i)
                filename = f"{self.pred_dir}/{date}_{parameter}_{'up' if self.dir_up else 'down'}.png" if self.pred_mode else f"{self.cache_dir}/{date}_{parameter}.png"
                images.append(Image.open(filename))

            images[0].save(
                gif_filename,
                save_all=True,
                append_images=images[1:],
                duration=20,
                loop=0
            )
            print(f"GIF saved to {gif_filename}")

        self.play_gif(gif_filename)

    def string_to_interval(self, interval_str):
        # Remove the brackets and split by the semicolon
        left_bracket, right_bracket = interval_str[0], interval_str[-1]
        lower, upper = map(float, interval_str[1:-1].split(','))

        # Determine the closed/open nature of the interval
        closed = None
        if left_bracket == '[' and right_bracket == ')':
            closed = 'left'
        elif left_bracket == '(' and right_bracket == ']':
            closed = 'right'
        elif left_bracket == '[' and right_bracket == ']':
            closed = 'both'
        elif left_bracket == '(' and right_bracket == ')':
            closed = 'neither'

        return pd.Interval(left=lower, right=upper, closed=closed)

    def find_row_containing(self, df, intervals, x):
        for interval, (index, row) in zip(intervals, df.iterrows()):
            if x in interval:
                return row
        return None

    def calc_vals(self, latitude, intervals, df):
        vx = np.fromiter(map(lambda item: self.find_row_containing(df, intervals, item)['vx'], latitude),
                         dtype=float)
        vy = np.fromiter(map(lambda item: self.find_row_containing(df, intervals, item)['vy'], latitude),
                         dtype=float)
        v = np.fromiter(map(lambda item: self.find_row_containing(df, intervals, item)['v'], latitude),
                        dtype=float)
        phi = np.fromiter(map(lambda item: self.find_row_containing(df, intervals, item)['phi'], latitude),
                          dtype=float)
        return vx, vy, v, phi

    def generate_llv(self, latitudes, lats, lons, dir_up):
        latsf = lats.flatten()
        lonsf = lons.flatten()

        intervals = [self.string_to_interval(i) for i in self.means[int(dir_up)].bins]

        vx, vy, v, phi = self.calc_vals(latitudes, intervals, self.means[int(dir_up)])

        calc_vx = {latitudes[i]: vx[i] for i in range(len(latitudes))}
        calc_vy = {latitudes[i]: vy[i] for i in range(len(latitudes))}
        calc_v = {latitudes[i]: v[i] for i in range(len(latitudes))}
        calc_phi = {latitudes[i]: phi[i] for i in range(len(latitudes))}

        vx = np.array([calc_vx[item] for item in latsf])
        vy = np.array([calc_vy[item] for item in latsf])
        v = np.array([calc_v[item] for item in latsf])
        phi = np.array([calc_phi[item] for item in latsf])

        return vx, vy, v, phi

    def generate_predictions(self, model, day_number, lats, lons, vx, vy, v, phi, la, lo, direction):
        return model.predict(day_number, lats, lons, vx, vy, v, phi, la, lo, direction)

    def plot_every(self, df, parameter, n):
        fig, ax = plt.subplots(figsize=(12, 6.75))
        img = plt.imread("world_map.png")  # Make sure you have a 'world_map.png' file
        ax.imshow(img, extent=[-180, 180, -90, 90])

        # Plot every n-th record
        df = df.iloc[::n]
        scatter = ax.scatter(df['Longitude'], df['Latitude'], c=df[parameter], cmap='viridis', s=1, alpha=0.9,
                             vmin=self.param_lims[parameter]['min'], vmax=self.param_lims[parameter]['max'])

        # Add a color bar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(parameter)

        ax.set_xlim([-180, 180])
        ax.set_ylim([-90, 90])
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'Satellite Data ({parameter})')

        return fig

    def cache_plots(self):
        days = len(self.data)
        start_time = time.time()
        processed = 0

        for date in sorted(self.data.keys()):
            for parameter in self.parameters:
                filename = f"{self.cache_dir}/{date}_{parameter}.png"
                if not os.path.exists(filename):
                    fig = self.plot_every(self.data[date], parameter, self.every)
                    fig.savefig(filename, dpi=133)  # Increase the DPI for higher resolution
                    plt.close(fig)
            processed += 1
            elapsed_time = time.time() - start_time
            avg_time_per_day = elapsed_time / processed
            remaining_time = avg_time_per_day * (days - processed)
            print(
                f"\rProcessed: {processed}/{days} ({processed / days * 100:.2f}%)\tElapsed: {elapsed_time:.2f}s\tERT: {remaining_time:.2f}s",
                end='')

    def cache_predictions(self):
        start_time = time.time()
        processed = 0

        latitudes = np.linspace(-87, 87, 176)  # Grid of latitude points
        longitudes = np.linspace(-180, 180, 360)  # Grid of longitude points

        la, lo = len(latitudes), len(longitudes)

        lons, lats = np.meshgrid(longitudes, latitudes)
        lonsf, latsf = lons.flatten(), lats.flatten()

        vx0, vy0, v0, phi0 = self.generate_llv(latitudes, lats, lons, False)
        vx1, vy1, v1, phi1 = self.generate_llv(latitudes, lats, lons, True)

        for parameter, model in self.models.items():
            for day_number in range(self.days):
                date = self.get_date(day_number)
                for dir_up in [True, False]:
                    pred_filename = f"{self.pred_dir}/{date}_{parameter}_{'up' if dir_up else 'down'}.png"

                    if not os.path.exists(pred_filename):
                        # Generate predictions (replace 'model' with your actual model)
                        predictions = self.generate_predictions(
                            self.models[parameter],
                            day_number,
                            latsf,
                            lonsf,
                            vx1 if dir_up else vx0,
                            vy1 if dir_up else vy0,
                            v1 if dir_up else v0,
                            phi1 if dir_up else phi0,
                            la, lo, np.array([dir_up] * len(latsf)))

                        # Plot the predictions using pcolormesh
                        fig, ax = plt.subplots(figsize=(12, 6.75))
                        img = plt.imread("world_map.png")  # Make sure you have a 'world_map.png' file
                        ax.imshow(img, extent=[-180, 180, -90, 90], zorder=0)

                        c = ax.pcolormesh(lons, lats, predictions, vmin=self.param_lims[parameter]['min'],
                                          vmax=self.param_lims[parameter]['max'], cmap='viridis', shading='auto',
                                          alpha=0.9)

                        # Add a color bar
                        cbar = plt.colorbar(c, ax=ax)
                        cbar.set_label(parameter)

                        ax.set_xlim([-180, 180])
                        ax.set_ylim([-90, 90])
                        ax.set_xlabel('Longitude')
                        ax.set_ylabel('Latitude')
                        ax.set_title(f'Predicted {parameter} for {date}')

                        fig.savefig(pred_filename, dpi=133)
                        plt.close(fig)

                    processed += 1
                    elapsed_time = time.time() - start_time
                    avg_time_per_day = elapsed_time / processed
                    remaining_time = avg_time_per_day * (self.days * 2 * len(self.models) - processed)
                    print(
                        f"\rPredicted: {processed}/{self.days * 2 * len(self.models)} ({processed / (self.days * 2 * len(self.models)) * 100:.2f}%)\tElapsed: {elapsed_time:.2f}s\tERT: {remaining_time:.2f}s        ",
                        end='')

    def update_plot(self, filename=None):
        if filename is None:
            date_index = self.date_slider.value()
            parameter = self.parameter_dropdown.currentText()
            date = self.get_date(date_index)
            self.filename = None
            filename = f"{self.cache_dir}/{date}_{parameter}.png" if not self.pred_mode else f"{self.pred_dir}/{date}_{parameter}_{'up' if self.dir_up else 'down'}.png"

        pixmap = QPixmap(filename)
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(False)
        self.image_label.setPixmap(
            pixmap.scaled(self.image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

        self.image_label.repaint()

    def resizeEvent(self, event):
        if self.filename and self.filename.endswith('.gif'):
            self.play_gif(self.filename)
        else:
            self.update_plot()
        super().resizeEvent(event)


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    ex = SatelliteApp(every=2, days=365)
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
