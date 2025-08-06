import sys
import numpy as np
import math
import io
from PIL import Image
import plotly.graph_objects as go
from plotly.io import to_image
from PyQt5.QtWidgets import (
    QApplication, QLabel, QMainWindow,
    QVBoxLayout, QWidget, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
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


def cprint(txt, color="92"):
    color_map = {
        "red": "91",
        "green": "92",
        "yellow": "93",
        "blue": "94",
        "purple": "35"
    }
    color_code = color_map.get(color.lower(), "0")
    print(f"\033[{color_code}m{txt}\033[0m")


# ----------------------------------------------------------------------
# 3-D PLOTTING FUNCTIONS
# ----------------------------------------------------------------------
def plot_orbit_3d(primary, equatorial_radius, polar_radius,
                  camera_pos=None,
                  spacecraft_x=None, spacecraft_y=None, spacecraft_z=None,
                  coverage=None,
                  final_longitude_deg=None,  # <-- new parameter
                  ground_stations=None):

    # ----------------------------------------------------------
    # Compute rotation offset directly from given final longitude
    # ----------------------------------------------------------
    if final_longitude_deg is not None:
        mean_lon = math.radians(final_longitude_deg)
        rotation_offset_rad = (-mean_lon + np.pi) % (2 * np.pi)
    else:
        rotation_offset_rad = 0.0

    # ----------------------------------------------------------
    # Camera positioning
    # ----------------------------------------------------------
    camera_pos_mag = math.sqrt(camera_pos[0]**2 + camera_pos[1]**2 + camera_pos[2]**2)
    maxAbsX = max(np.abs(np.array(spacecraft_x))) if spacecraft_x is not None else 1
    maxAbsY = max(np.abs(np.array(spacecraft_y))) if spacecraft_y is not None else 1
    maxAbsZ = max(np.abs(np.array(spacecraft_z))) if spacecraft_z is not None else 1
    camera_scale = camera_pos_mag / max([maxAbsX, maxAbsY, maxAbsZ]) / 2
    camera_pos_new = [camera_pos[0] / camera_pos_mag,
                      camera_pos[1] / camera_pos_mag,
                      camera_pos[2] / camera_pos_mag]
    camera_pos_new = [c * camera_scale for c in camera_pos_new]

    a = float(equatorial_radius)
    b = float(polar_radius)

    # ----------------------------------------------------------
    # Load planet texture
    # ----------------------------------------------------------
    image_path = f"{primary}_mosaic.png"
    img = Image.open(image_path).convert("RGB")
    img = img.resize((600, 300)).transpose(Image.FLIP_TOP_BOTTOM)
    img_data = np.array(img)

    n_lat, n_lon, _ = img_data.shape
    lats = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
    lons = np.linspace(0, 2 * np.pi, n_lon)

    # Planet surface mesh
    lon_grid, lat_grid = np.meshgrid((lons + rotation_offset_rad) % (2 * np.pi), lats)
    x_surf = a * np.cos(lat_grid) * np.cos(lon_grid)
    y_surf = a * np.cos(lat_grid) * np.sin(lon_grid)
    z_surf = b * np.sin(lat_grid)

    faces = []
    for i in range(n_lat - 1):
        for j in range(n_lon - 1):
            idx = i * n_lon + j
            faces.append([idx, idx + 1, idx + n_lon])
            faces.append([idx + 1, idx + n_lon + 1, idx + n_lon])
    i_faces, j_faces, k_faces = zip(*faces)

    # Face colors
    face_colors = []
    for f in faces:
        c1 = img_data[f[0] // n_lon, f[0] % n_lon]
        c2 = img_data[f[1] // n_lon, f[1] % n_lon]
        c3 = img_data[f[2] // n_lon, f[2] % n_lon]
        avg_rgb = np.mean([c1, c2, c3], axis=0).astype(int)
        face_colors.append(f'rgb({avg_rgb[0]},{avg_rgb[1]},{avg_rgb[2]})')

    # ----------------------------------------------------------
    # Spacecraft track
    # ----------------------------------------------------------
    if spacecraft_x is not None:
        spacecraft_x = np.array(spacecraft_x, dtype=float)
        spacecraft_y = np.array(spacecraft_y, dtype=float)
        spacecraft_z = np.array(spacecraft_z, dtype=float)
    else:
        spacecraft_x = spacecraft_y = spacecraft_z = np.array([])

    # ----------------------------------------------------------
    # Build coverage mesh
    # ----------------------------------------------------------
    precomputed_coverage_surface = None
    if coverage is not None and np.any(coverage):
        coverage = np.array(coverage, dtype=bool)
        rows, cols = coverage.shape

        max_quads = 1_600_000
        est_quads = (rows - 1) * (cols - 1)
        if est_quads > max_quads:
            factor_r = math.ceil(rows / math.sqrt(max_quads))
            factor_c = math.ceil(cols / math.sqrt(max_quads))
            coverage = coverage[::factor_r, ::factor_c]
            rows, cols = coverage.shape
            print()
            cprint(f"Coverage downsampled to {rows}x{cols} for performance", "yellow")


        epsilon = np.radians(0.1)
        cov_lat = np.linspace(np.pi / 2 - epsilon, -np.pi / 2 + epsilon, rows)
        cov_lon = (np.linspace(0, 2 * np.pi, cols) + rotation_offset_rad) % (2 * np.pi)
        lat_m, lon_m = np.meshgrid(cov_lat, cov_lon, indexing='ij')

        swath_scale = 1.01
        x_cov = swath_scale * a * np.cos(lat_m) * np.cos(lon_m)
        y_cov = swath_scale * a * np.cos(lat_m) * np.sin(lon_m)
        z_cov = swath_scale * b * np.sin(lat_m)

        x_flat = x_cov.flatten()
        y_flat = y_cov.flatten()
        z_flat = z_cov.flatten()

        quad_mask = coverage[:-1, :-1] & coverage[1:, :-1] & coverage[:-1, 1:] & coverage[1:, 1:]
        i_idx, j_idx = np.where(quad_mask)
        if len(i_idx) > 0:
            f1 = np.column_stack([i_idx * cols + j_idx,
                                  i_idx * cols + (j_idx + 1),
                                  (i_idx + 1) * cols + j_idx])
            f2 = np.column_stack([i_idx * cols + (j_idx + 1),
                                  (i_idx + 1) * cols + (j_idx + 1),
                                  (i_idx + 1) * cols + j_idx])
            faces_cov = np.vstack([f1, f2])
            i_cov, j_cov, k_cov = faces_cov.T

            precomputed_coverage_surface = go.Mesh3d(
                x=x_flat, y=y_flat, z=z_flat,
                i=i_cov, j=j_cov, k=k_cov,
                color='blue', opacity=0.5,
                flatshading=True,
                lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0),
                showscale=False,
                name="Coverage"
            )

    # ----------------------------------------------------------
    # Ground stations
    # ----------------------------------------------------------
    ground_station_trace = None
    if ground_stations:
        lat_list, lon_list, name_list = [], [], []
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

        ground_station_trace = go.Scatter3d(
            x=x_gs, y=y_gs, z=z_gs,
            mode='markers+text', marker=dict(size=3, color='purple'),
            text=name_list, textposition='top center',
            textfont=dict(size=16, color='white'),
            name='Ground Stations'
        )

    # ----------------------------------------------------------
    # Generate plot
    # ----------------------------------------------------------
    def generate_plot():
        traces = []

        if precomputed_coverage_surface is not None:
            traces.append(precomputed_coverage_surface)

        if len(spacecraft_x) > 0:
            traces.append(go.Scatter3d(
                x=spacecraft_x, y=spacecraft_y, z=spacecraft_z,
                mode='lines', line=dict(color='red', width=4), name="Spacecraft"
            ))
            traces.append(go.Scatter3d(
                x=[spacecraft_x[-1]], y=[spacecraft_y[-1]], z=[spacecraft_z[-1]],
                mode='markers', marker=dict(size=6, color='red', symbol='circle'),
                name='Final Position', showlegend=False
            ))

        traces.append(go.Mesh3d(
            x=x_surf.flatten(), y=y_surf.flatten(), z=z_surf.flatten(),
            i=i_faces, j=j_faces, k=k_faces,
            facecolor=face_colors,
            lighting=dict(ambient=0.9, diffuse=0.1, specular=0.05),
            flatshading=True,
            showscale=False,
            name="Planet"
        ))

        if ground_station_trace:
            traces.append(ground_station_trace)

        # Fixed axes
        arrow_len = 1.2 * max(a, b)
        label_offset = 0.12 * arrow_len
        cone_size = 0.05 * arrow_len

        def create_arrow_with_label(start, end, color, label):
            vec = np.array(end) - np.array(start)
            norm = np.linalg.norm(vec)
            if norm == 0:
                return []
            direction = vec / norm
            label_pos = np.array(end) + direction * label_offset
            return [
                go.Scatter3d(x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]],
                             mode='lines', line=dict(color=color, width=5), showlegend=False),
                go.Scatter3d(x=[label_pos[0]], y=[label_pos[1]], z=[label_pos[2]],
                             mode='text', text=[label], textposition='middle center',
                             textfont=dict(size=28, color=color), showlegend=False),
                go.Cone(x=[end[0]], y=[end[1]], z=[end[2]],
                        u=[direction[0]], v=[direction[1]], w=[direction[2]],
                        colorscale=[[0, color], [1, color]], showscale=False,
                        sizemode='absolute', sizeref=cone_size, anchor='tail', name=label)
            ]

        for vec, name in zip([[arrow_len, 0, 0], [0, arrow_len, 0], [0, 0, arrow_len*1.2]],
                             ['i₁', 'i₂', 'i₃']):
            traces.extend(create_arrow_with_label([0, 0, 0], vec, 'purple', name))

        theta = -rotation_offset_rad + np.pi/2

        f1 = [arrow_len * np.cos(theta), arrow_len * np.sin(theta), 0]
        f3 = [0, 0, arrow_len]
        f2 = np.cross(f3, f1)
        f2 = arrow_len * np.array(f2) / np.linalg.norm(f2)

        for vec, name in zip([f1, f2, f3], ['f₁', 'f₂', 'f₃']):
            traces.extend(create_arrow_with_label([0, 0, 0], vec, 'yellow', name))

        fig = go.Figure(data=traces)
        fig.update_layout(
            scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                       aspectmode="data",
                       camera=dict(eye=dict(x=camera_pos_new[0], y=camera_pos_new[1], z=camera_pos_new[2]),
                                   center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=1))),
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
            paper_bgcolor="black",
            plot_bgcolor="black"
        )
        img_bytes = to_image(fig, format="png", width=1250, height=825)
        return Image.open(io.BytesIO(img_bytes)).convert("RGBA")

    # ----------------------------------------------------------
    # PyQt5 Window
    # ----------------------------------------------------------
    class PlotWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("3-D Plot")
            self.resize(1000, 660)
            self.label = QLabel()
            self.label.setAlignment(Qt.AlignCenter)
            self.label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
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
export_path = ""
camera_pos = [float(Plotting_XPositionkm), float(Plotting_YPositionkm), float(Plotting_ZPositionkm)]
app = QApplication([])

if Plotting_3DVizualization:
    print()
    cprint("Creating 3-D Plot", "blue")
    plot3D_5 = plot_orbit_3d(primary, primary_equitorial_radius, primary_polar_radius,
                             camera_pos, xPositions, yPositions, zPositions, total_coverage,
                             longitudes[-1], stations)

app.exec_()
