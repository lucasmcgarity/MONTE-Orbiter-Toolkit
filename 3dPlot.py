import os
import sys
import base64
import numpy as np
import math
import io
from io import BytesIO
from datetime import datetime, timedelta
import time
from PIL import Image
import plotly.graph_objects as go
from plotly.io import to_image
from PyQt5.QtWidgets import( 
                        QApplication, QLabel, QMainWindow, QScrollArea, QFrame,
                        QVBoxLayout, QGridLayout, QWidget, QSizePolicy 
                        )
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
import json


# ----------------------------------------------------------------------
# IMPORT USER INPUT AND MONTE SIMULATION DATA
# ----------------------------------------------------------------------

def load_json_to_globals(path):

    with open(path, "r") as f:
    
        data = json.load(f)
    
    for k in data:
    
        globals()[k] = data[k]

load_json_to_globals(sys.argv[1])
load_json_to_globals(sys.argv[2])

def cprint(txt,color="92"):

    color_map = {
        "red":"91",
        "green":"92",
        "yellow":"93",
        "blue":"94",
        "purple":"35"
    }

    color_code = color_map.get(color.lower(),"0")

    print(f"\033[{color_code}m{txt}\033[0m")


# ----------------------------------------------------------------------
# 3-D PLOTTING FUNCTIONS
# ----------------------------------------------------------------------

def plot_orbit_3d(primary, equatorial_radius, polar_radius,
                  camera_pos=None,
                  spacecraft_x=None, spacecraft_y=None, spacecraft_z=None,
                  coverage=None,
                  rotation_offset_deg=0.0,
                  ground_stations=None):

    camera_scale = 1.6
    camera_pos_mag = math.sqrt(camera_pos[0]**2+camera_pos[1]**2+camera_pos[2]**2)
    camera_pos_new = [0,0,0]

    camera_pos_new[0] = camera_pos[0] / camera_pos_mag
    camera_pos_new[1] = camera_pos[1] / camera_pos_mag
    camera_pos_new[2] = camera_pos[2] / camera_pos_mag

    camera_pos_new = [c * camera_scale for c in camera_pos_new]

    a = float(equatorial_radius)
    b = float(polar_radius)
    rotation_offset_rad = np.radians(rotation_offset_deg)

    image_path = f"{primary}_mosaic.png"
    img = Image.open(image_path).convert("RGB")
    img = img.resize((600, 300)).transpose(Image.FLIP_TOP_BOTTOM)
    img_data = np.array(img)

    n_lat, n_lon, _ = img_data.shape
    lats = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
    lons = np.linspace(0, 2 * np.pi, n_lon)

    faces = []
    for i in range(n_lat - 1):
        for j in range(n_lon - 1):
            idx = i * n_lon + j
            faces.append([idx, idx + 1, idx + n_lon])
            faces.append([idx + 1, idx + n_lon + 1, idx + n_lon])
    i_faces, j_faces, k_faces = zip(*faces)

    face_colors = []
    for f in faces:
        c1 = img_data[f[0] // n_lon, f[0] % n_lon]
        c2 = img_data[f[1] // n_lon, f[1] % n_lon]
        c3 = img_data[f[2] // n_lon, f[2] % n_lon]
        avg_rgb = np.mean([c1, c2, c3], axis=0).astype(int)
        face_colors.append(f'rgb({avg_rgb[0]},{avg_rgb[1]},{avg_rgb[2]})')

    if spacecraft_x is not None:
        spacecraft_x = np.array(spacecraft_x, dtype=float)
        spacecraft_y = np.array(spacecraft_y, dtype=float)
        spacecraft_z = np.array(spacecraft_z, dtype=float)
    else:
        spacecraft_x = spacecraft_y = spacecraft_z = np.array([])

    precomputed_coverage_surface = None
    if coverage is not None and np.any(coverage):
        coverage = np.array(coverage, dtype=bool)
        epsilon = np.radians(0.1)

        cov_lat = np.linspace(np.pi / 2 - epsilon, -np.pi / 2 + epsilon, coverage.shape[0])
        cov_lon = (np.linspace(0, 2 * np.pi, coverage.shape[1]) + rotation_offset_rad) % (2 * np.pi)
        lat_m, lon_m = np.meshgrid(cov_lat, cov_lon, indexing='ij')

        swath_scale = 1.01
        x_cov = swath_scale * a * np.cos(lat_m) * np.cos(lon_m)
        y_cov = swath_scale * a * np.cos(lat_m) * np.sin(lon_m)
        z_cov = swath_scale * b * np.sin(lat_m)

        x_flat = x_cov.flatten()
        y_flat = y_cov.flatten()
        z_flat = z_cov.flatten()

        mask = coverage
        faces = []
        rows, cols = coverage.shape

        for i in range(rows - 1):
            for j in range(cols - 1):
                if mask[i, j] and mask[i+1, j] and mask[i, j+1] and mask[i+1, j+1]:
                    idx = i * cols + j
                    faces.append([idx, idx + 1, idx + cols])
                    faces.append([idx + 1, idx + cols + 1, idx + cols])

        if faces:
            i_cov, j_cov, k_cov = zip(*faces)
            precomputed_coverage_surface = go.Mesh3d(
                x=x_flat, y=y_flat, z=z_flat,
                i=i_cov, j=j_cov, k=k_cov,
                color='blue', opacity=0.5,
                flatshading=True,
                lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0),
                showscale=False,
                name="Coverage"
            )

    def generate_plot():
        lon_grid, lat_grid = np.meshgrid(
            (np.linspace(0, 2 * np.pi, n_lon) + rotation_offset_rad) % (2 * np.pi),
            lats
        )
        x = a * np.cos(lat_grid) * np.cos(lon_grid)
        y = a * np.cos(lat_grid) * np.sin(lon_grid)
        z = b * np.sin(lat_grid)

        # Shrink spheroid slightly to avoid z-fighting
        planet_scale = 0.9995
        x *= planet_scale
        y *= planet_scale
        z *= planet_scale

        surface = go.Mesh3d(
            x=x.flatten(), y=y.flatten(), z=z.flatten(),
            i=i_faces, j=j_faces, k=k_faces,
            facecolor=face_colors,
            lighting=dict(ambient=0.9, diffuse=0.1, specular=0.05),
            flatshading=True,
            showscale=False,
            name="Planet"
        )

        traces = []

        # Draw coverage mesh first
        if precomputed_coverage_surface is not None:
            traces.append(precomputed_coverage_surface)

        # Draw spacecraft
        if len(spacecraft_x) > 0:
            traces.append(go.Scatter3d(
                x=spacecraft_x, y=spacecraft_y, z=spacecraft_z,
                mode='lines',
                line=dict(color='red', width=4),
                name="Spacecraft"
            ))

            traces.append(go.Scatter3d(
                x=[spacecraft_x[:-1]], 
                y=[spacecraft_y[:-1]], 
                z=[spacecraft_z[:-1]],
                mode='markers',
                marker=dict(size=6, color='red', symbol='circle'),
                name = 'Final Position',
                showlegend=False,
            ))
        
        traces.append(surface)

        if ground_stations:

            lat_list = []
            lon_list = []
            name_list = []

            for name, (lat, lon) in ground_stations.items():
                try:
                    lat = float(lat)
                    lon = float(lon)
                except ValueError:
                    continue
                lat_rad = np.radians(lat)
                lon_rad = (np.radians(lon + 180))
                lon_rad = (lon_rad + rotation_offset_rad) % (2 * np.pi)
                lat_list.append(lat_rad)
                lon_list.append(lon_rad)
                name_list.append(name)

            r_scale = 1.02
            x_gs = r_scale * a * np.cos(lat_list) * np.cos(lon_list)
            y_gs = r_scale * a * np.cos(lat_list) * np.sin(lon_list)
            z_gs = r_scale * b * np.sin(lat_list)
            
            traces.append(go.Scatter3d(
                x=x_gs, y=y_gs, z=z_gs,
                mode='markers+text',
                marker=dict(size=3, color='purple'),
                text=name_list,
                textposition='top center',
                textfont=dict(size=8,color='white'),
                name='Ground Stations'
            ))

        arrow_len = 1.2 * max(a, b)
        label_offset = 0.12 * arrow_len
        cone_size = 0.05 * arrow_len

        arrow_color_fixed = 'purple'
        arrow_color_rotating = 'yellow'

        def create_arrow_with_label(start, end, color, label):
            # Vector and normalized direction
            vec = np.array(end) - np.array(start)
            norm = np.linalg.norm(vec)
            if norm == 0:
                return []
            direction = vec / norm

            # Slightly extend label position
            label_pos = np.array(end) + direction * label_offset

            return [
                # Line
                go.Scatter3d(
                    x=[start[0], end[0]],
                    y=[start[1], end[1]],
                    z=[start[2], end[2]],
                    mode='lines',
                    line=dict(color=color, width=5),
                    showlegend=False
                ),
                # Label
                go.Scatter3d(
                    x=[label_pos[0]],
                    y=[label_pos[1]],
                    z=[label_pos[2]],
                    mode='text',
                    text=[label],
                    textposition='middle center',
                    textfont=dict(size=20, color=color),
                    showlegend=False
                ),
                # Arrowhead
                go.Cone(
                    x=[end[0]],
                    y=[end[1]],
                    z=[end[2]],
                    u=[direction[0]],
                    v=[direction[1]],
                    w=[direction[2]],
                    colorscale=[[0, color], [1, color]],
                    showscale=False,
                    sizemode='absolute',
                    sizeref=cone_size,
                    anchor='tail',
                    name=label
                )
            ]

        # Fixed axes (i₁, i₂, i₃)
        origin = [0, 0, 0]
        i1 = [arrow_len, 0, 0]
        i2 = [0, arrow_len, 0]
        i3 = [0, 0, arrow_len]

        for vec, name in zip([i1, i2, i3], ['i₁', 'i₂', 'i₃']):
            traces.extend(create_arrow_with_label(origin, vec, arrow_color_fixed, name))

        # Rotating axes (f₁, f₂, f₃)
        theta = rotation_offset_rad
        f1 = i3  # same as +z
        f2 = [arrow_len * np.cos(theta), arrow_len * np.sin(theta), 0]
        f3 = np.cross(f1, f2)
        f3 = arrow_len * np.array(f3) / np.linalg.norm(f3)

        for vec, name in zip([f1, f2, f3], ['f₁', 'f₂', 'f₃']):
            traces.extend(create_arrow_with_label(origin, vec, arrow_color_rotating, name))

        fig = go.Figure(data=traces)
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode="data",
                camera=dict(
                    eye=dict(x=camera_pos_new[0], y=camera_pos_new[1], z=camera_pos_new[2]),
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1)
                )
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
            paper_bgcolor="black",
            plot_bgcolor="black"
        )

        img_bytes = to_image(fig, format="png", width=1250, height=825)
        return Image.open(io.BytesIO(img_bytes)).convert("RGBA")

    class PlotWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("3-D Plot")
            self.resize(1000, 660)

            self.label = QLabel()
            self.label.setAlignment(Qt.AlignCenter)
            self.label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
            self.label.setScaledContents(False)

            layout = QVBoxLayout()
            layout.addWidget(self.label)
            container = QWidget()
            container.setLayout(layout)
            self.setCentralWidget(container)

            pil_img = generate_plot()
            data = pil_img.tobytes("raw", "RGBA")
            qimg = QImage(data, pil_img.width, pil_img.height, QImage.Format_RGBA8888)
            self.original_pixmap = QPixmap.fromImage(qimg)
            self.update_pixmap()

        def resizeEvent(self, event):
            self.update_pixmap()
            super().resizeEvent(event)

        def update_pixmap(self):
            if hasattr(self, 'original_pixmap'):
                scaled = self.original_pixmap.scaled(
                    self.centralWidget().size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.label.setPixmap(scaled)

    window = PlotWindow()
    window.show()

    print("")
    cprint("3-D Plot Created", "green")

    return window


# ------------------------------------------------------------------------------------
# CREATE PLOT
# ------------------------------------------------------------------------------------

export_path=""
camera_pos = [float(Plotting_XPositionkm),float(Plotting_YPositionkm),float(Plotting_ZPositionkm)]
app = QApplication([])

if Plotting_3DVizualization:

    print()
    cprint("Creating 3-D Plot","blue")
    plot3D_5 = plot_orbit_3d(primary,primary_equitorial_radius,primary_polar_radius,
                           camera_pos,xPositions, yPositions, zPositions,total_coverage,-sunLongitude+101,stations)
    
app.exec_()