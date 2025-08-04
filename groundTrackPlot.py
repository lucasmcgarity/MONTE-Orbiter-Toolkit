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
# GROUND TRACK PLOTTING FUNCTIONS
# ----------------------------------------------------------------------

def plot_groundtrack(latitudes, longitudes,
                     initial_time_str, time_step,
                     animated, primary,
                     coverage=None,
                     export_path="", opacity=0.5,
                     in_contact=None,
                     ground_stations=None,
                     qt_app=None):

    image_path = f"{primary}_equirectangular.png"
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Background image not found: {image_path}")
    bg_img = Image.open(image_path).convert("RGBA")
    width, height = bg_img.size

    border_ratio_x = 90 / 1920
    border_ratio_y = 90 / 1140
    border_x = border_ratio_x * width
    border_y = border_ratio_y * height
    plot_left, plot_right = border_x, width - border_x
    plot_top, plot_bottom = border_y, height - border_y

    buffer = BytesIO()
    bg_img.save(buffer, format="PNG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    image_uri = f"data:image/png;base64,{encoded_image}"

    time_zone = initial_time_str.split()[-1]
    base_time = datetime.strptime(initial_time_str[:-len(time_zone)].strip(), "%d-%b-%Y %H:%M:%S.%f")
    times = [base_time + timedelta(seconds=i * time_step) for i in range(len(latitudes))]

    def lon_to_x(lon):
        return plot_left + (lon + 180) / 360 * (plot_right - plot_left)

    def lat_to_y(lat):
        lat_clamped = np.clip(lat, -90.0, 90.0)
        norm_y = (90.0 - lat_clamped) / 180.0
        return plot_top + norm_y * (plot_bottom - plot_top)

    x_coords = [lon_to_x(lon) for lon in longitudes]
    y_coords = [lat_to_y(lat) for lat in latitudes]

    fig = go.Figure()

    fig.add_layout_image(dict(
        source=image_uri,
        x=0, y=0,
        sizex=width, sizey=height,
        xref="x", yref="y",
        sizing="stretch",
        layer="below"
    ))

    if coverage is not None:
        coverage = np.array(coverage, dtype=np.uint8)
        lat_res, lon_res = coverage.shape
        alpha_val = int(opacity * 255)
        rgba_array = np.zeros((lat_res, lon_res, 4), dtype=np.uint8)
        rgba_array[:, :, 2] = 255 * coverage
        rgba_array[:, :, 3] = alpha_val * coverage

        coverage_img = Image.fromarray(rgba_array)
        plot_width = int(plot_right - plot_left)
        plot_height = int(plot_bottom - plot_top)
        coverage_img = coverage_img.resize((plot_width, plot_height), resample=Image.NEAREST)

        buf = BytesIO()
        coverage_img.save(buf, format="PNG")
        coverage_encoded = base64.b64encode(buf.getvalue()).decode()

        fig.add_layout_image(dict(
            source=f"data:image/png;base64,{coverage_encoded}",
            x=plot_left, y=plot_top,
            sizex=plot_width, sizey=plot_height,
            xref="x", yref="y",
            sizing="stretch",
            layer="below"
        ))

    def segment_groundtrack(lons, lats, contacts):
        segments = []
        current_segment_x, current_segment_y, current_segment_contact = [], [], []

        for i in range(len(lons)):
            if i > 0 and abs(lons[i] - lons[i - 1]) > 180:
                segments.append((current_segment_x, current_segment_y, current_segment_contact))
                current_segment_x, current_segment_y, current_segment_contact = [], [], []

            current_segment_x.append(lon_to_x(lons[i]))
            current_segment_y.append(lat_to_y(lats[i]))
            if contacts is not None:
                current_segment_contact.append(contacts[i])
            else:
                current_segment_contact.append(False)

        segments.append((current_segment_x, current_segment_y, current_segment_contact))
        return segments

    if animated:
        for i in range(1, len(latitudes)):
            if abs(longitudes[i] - longitudes[i - 1]) > 180:
                continue
            color = "green" if in_contact and in_contact[i] else "red"
            fig.add_trace(go.Scatter(
                x=[x_coords[i - 1], x_coords[i]],
                y=[y_coords[i - 1], y_coords[i]],
                mode="lines+markers",
                line=dict(color=color),
                marker=dict(size=5, color=color),
                showlegend=False
            ))
    else:
        segments = segment_groundtrack(longitudes, latitudes, in_contact or [False]*len(latitudes))
        for x_seg, y_seg, c_seg in segments:
            for i in range(1, len(x_seg)):
                color = "green" if c_seg[i] else "red"
                fig.add_trace(go.Scatter(
                    x=[x_seg[i - 1], x_seg[i]],
                    y=[y_seg[i - 1], y_seg[i]],
                    mode="lines",
                    line=dict(color=color),
                    showlegend=False
                ))

    if animated:
        fig.add_annotation(
            x=width // 2,
            y=height - border_y // 2,
            text=times[-1].strftime("%d-%b-%Y %H:%M:%S.%f")[:-3] + f" {time_zone}",
            showarrow=False,
            font=dict(color="white")
        )

    def draw_ground_stations(stations, dot_color="purple", dot_size=10,font_size=18, box_pad=2):

        for name, (lat_raw, lon_raw) in stations.items():
            try:
                lat = float(lat_raw)
                lon = float(lon_raw)
            except ValueError:
                print(f"Invalid lat/lon for {name}: ({lat_raw}, {lon_raw})")
                continue

            x = lon_to_x(lon)
            y = lat_to_y(lat)

            label_offset = 30  # vertical pixel offset for label above dot

            # Add annotation (label) above the marker
            fig.add_annotation(
                x=x,
                y=y - label_offset,
                text=name,
                showarrow=False,
                font=dict(size=font_size, color="black"),
                align="center",
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                borderpad=2,
                opacity=0.9
            )

            # Draw dot last to ensure it's on top
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode="markers",
                marker=dict(color=dot_color, size=dot_size),
                showlegend=False,
                hoverinfo="skip"
            ))

    if ground_stations:
        draw_ground_stations(ground_stations)

    fig.update_layout(
        title="Ground Track Viewer",
        xaxis=dict(visible=False, range=[0, width], fixedrange=True, constrain="domain"),
        yaxis=dict(visible=False, range=[height, 0], fixedrange=True, constrain="domain"),
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False
    )

    html_path = export_path.replace(".png", ".html") if export_path else "groundtrack_plot.html"
    image_output = export_path if export_path else "groundtrack_plot.png"
    fig.write_html(html_path)
    fig.write_image(image_output, width=width, height=height, scale=1)

    app = qt_app or QApplication.instance() or QApplication(sys.argv)
    window = QMainWindow()
    window.setWindowTitle("Ground Track Viewer")

    scroll = QScrollArea()
    central_widget = QWidget()
    layout = QVBoxLayout(central_widget)

    label = QLabel()
    pixmap = QPixmap(image_output)
    label.setPixmap(pixmap)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
    label.setScaledContents(True)

    layout.addWidget(label)
    scroll.setWidget(central_widget)
    scroll.setWidgetResizable(True)
    window.setCentralWidget(scroll)
    window.resize(1280, 720)
    window.show()
    window.activateWindow()
    window.raise_()
    window.destroyed.connect(lambda: None)

    if qt_app is None and not QApplication.instance():
        app.exec_()

    print()
    cprint("Ground Track Plot Created", "green")

    return window


# ------------------------------------------------------------------------------------
# CREATE PLOT
# ------------------------------------------------------------------------------------

export_path=""
app = QApplication([])

if Plotting_GroundTrack:
    
    print()
    cprint("Creating Ground Track Plot","blue")
    groundTrackPlot = plot_groundtrack(latitudes, longitudes, inital_epoch_str, time_step_seconds,
                                       Plotting_AnimatePlots, primary, total_coverage, export_path, 
                                       0.5, contact_bool, stations, app)

app.exec_()