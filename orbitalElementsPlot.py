import sys
import numpy as np
from io import BytesIO
from datetime import datetime, timedelta
from PIL import Image
import plotly.graph_objects as go
from PyQt5.QtWidgets import( 
                        QApplication, QLabel, QMainWindow
                        )
from PyQt5.QtGui import QPixmap
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
# ORBITAL ELEMENTS PLOTTING FUNCTIONS
# ----------------------------------------------------------------------

def plot_orbital_elements(globalVars, element_names, initial_time_str, time_step,
                          animated, export_path="", qt_app=None):

    # Plot Settings

    title_font_size = 30
    title_font_color = "#ff0000"
    x_label_font_size = 14
    y_label_font_size = 24
    x_tick_label_font_size = 20
    y_tick_label_font_size = 20
    axis_color = "#FFFFFF"
    axis_width = 1
    line_color = "#FF0000"
    line_width = 1
    bg_color = "black"

    def clean_key(name):
        return name.split("[")[0].strip().replace("-", "").replace(" ", "")

    time_zone = initial_time_str.split()[-1]
    base_time = datetime.strptime(initial_time_str[:-len(time_zone)].strip(), "%d-%b-%Y %H:%M:%S.%f")
    n_points = len(globalVars["longitudes"])
    time_array = [base_time + timedelta(seconds=i * time_step) for i in range(n_points)]

    num_ticks = 8
    tick_indices = np.linspace(0, n_points - 1, num=num_ticks, dtype=int)
    tickvals = [time_array[i] for i in tick_indices]
    ticktexts = [f"{t.strftime('%d-%b-%Y')}<br>{t.strftime('%H:%M')}" for t in tickvals]

    plot_images = []
    for label in element_names:
        title = label.split("[")[0].strip()
        var_key = clean_key(label)
        data = globalVars[var_key]

        y_min = np.min(data)
        y_max = np.max(data)
        y_range = y_max - y_min
        margin = 0.15 * y_range if y_range > 0 else 1.0
        adjusted_ymin = y_min - margin
        adjusted_ymax = y_max + margin

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_array,
            y=data,
            mode="lines+markers" if animated else "lines",
            line=dict(color=line_color, width=line_width),
            marker=dict(size=5, color=line_color)
        ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=title_font_size,color=title_font_color)),
            xaxis=dict(
                title=dict(text=f"Epoch [{time_zone}]",font=dict(size=x_label_font_size)),
                color=axis_color,
                linewidth=axis_width,
                tickvals=tickvals,
                ticktext=ticktexts,
                ticks="outside",
                tickangle=0,
                tickfont=dict(size=x_tick_label_font_size),
                showline=True,
                mirror=True,
            ),
            yaxis=dict(
                title=dict(text=label,font=dict(size=y_label_font_size)),
                color=axis_color,
                linewidth=axis_width,
                range=[adjusted_ymin, adjusted_ymax],
                tickformat=".2e",
                ticks="outside",
                tickfont=dict(size=y_tick_label_font_size),
                showline=True,
                mirror=True,
            ),
            plot_bgcolor=bg_color,
            paper_bgcolor=bg_color,
            margin=dict(l=50, r=50, t=50, b=50),
            height=300,
            width=600
        )

        buffer = BytesIO()
        fig.write_image(buffer, format="png", width=1200, height=600, scale=2)
        buffer.seek(0)
        img = Image.open(buffer).convert("RGBA")
        plot_images.append(img)

    spacing = 20
    padding = 20
    plot_w, plot_h = plot_images[0].size
    rows = (len(plot_images) + 1) // 2
    composed_w = 2 * plot_w + spacing + 2 * padding
    composed_h = rows * plot_h + (rows - 1) * spacing + 2 * padding
    composed_image = Image.new("RGBA", (composed_w, composed_h), bg_color)

    for idx, img in enumerate(plot_images):
        row = idx // 2
        col = idx % 2
        x = col * (plot_w + spacing) + padding
        y = row * (plot_h + spacing) + padding
        composed_image.paste(img, (x, y))

    final_image = composed_image.resize((1920, 1080), Image.LANCZOS)
    img_buffer = BytesIO()
    final_image.save(img_buffer, format="PNG")
    img_buffer.seek(0)

    pixmap = QPixmap()
    pixmap.loadFromData(img_buffer.getvalue())

    window = QMainWindow()
    window.setWindowTitle("Orbital Element Plots")
    window.setMinimumSize(640, 480)

    label = QLabel()
    label.setAlignment(Qt.AlignCenter)
    label.setPixmap(pixmap)
    label.setScaledContents(True)

    window.setCentralWidget(label)
    window.show()

    print()
    cprint("Orbital Elements Plot Created","green")
    return window


# ------------------------------------------------------------------------------------
# CREATE PLOT
# ------------------------------------------------------------------------------------

export_path=""
app = QApplication([])

if Plotting_OrbitalElements:

    print()
    cprint("Creating Orbital Elements Plot","blue")
    orbitalElementsPlot = plot_orbital_elements(globals(), element_names, inital_epoch_str, time_step_seconds, 
                                                Plotting_AnimatePlots, export_path, app)
    
app.exec_()