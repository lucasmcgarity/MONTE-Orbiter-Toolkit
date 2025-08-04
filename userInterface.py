import sys
import os
import signal
import json
import subprocess
import Monte as M
import mpy.io.data as defaultData
from mpy.units import *
from mpy.io.stuf import *
from PyQt6.QtWidgets import (
                        QApplication, QWidget, QLabel, QPushButton, QLineEdit, QVBoxLayout, QScrollBar,
                        QHBoxLayout, QComboBox, QCheckBox, QGroupBox, QFileDialog, QScrollArea, QFrame,
                        QGridLayout, QListWidget, QAbstractItemView
                        )
from PyQt6.QtCore import (
                        Qt, QTimer, QPropertyAnimation, QRect, QEasingCurve, QEvent
                        )
from PyQt6.QtGui import (
                        QFont
                        )


# ----------------------------------------------------------------------
# FUNCTIONS TO CONTROL RUNNING OF SIMULATION AND PLOTTING
# ----------------------------------------------------------------------

process_handle = None
is_paused = False

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

def run_script(path_to_script,json_path):
    global process_handle
    if process_handle is None:
        process_handle = subprocess.Popen(
            ["python3", "-W", "ignore:A NumPy version >=", path_to_script, json_path],
            preexec_fn=os.setsid  # start in a new process group
        )
        os.system("clear")
        cprint(f"Simulation Started on Process: {process_handle.pid}","green")
    else:
        print()
        print()
        cprint("Simulation Already Running","yellow")

def pause_script():
    global process_handle
    if process_handle and process_handle.poll() is None:
        os.killpg(process_handle.pid, signal.SIGSTOP)
        print()
        print()
        cprint("Simulation Paused","yellow")

def resume_script():
    global process_handle
    if process_handle and process_handle.poll() is None:
        os.killpg(process_handle.pid, signal.SIGCONT)
        print()
        print()
        cprint("Simulation Resumed","green")

def terminate_script():
    global process_handle
    if process_handle and process_handle.poll() is None:
        os.killpg(process_handle.pid, signal.SIGTERM)
        os.system("clear")
        cprint("Simulation Stopped","red")
    else:
        print()
        print()
        cprint("No Simulation to Terminate","yellow")
    process_handle = None


# ---------------------------------------------------------------------- 
# USER INTERFACE
# ----------------------------------------------------------------------

boa = M.BoaLoad()

defaultData.loadInto(boa, [
    "frame", 
    "body", 
    "ephem/planet/de405",
    "station"
    ])

class OrbitSimUI(QWidget):

    def __init__(self):

        os.system("clear")
        cprint("Initializing GUI","green")
        super().__init__()
        self.setWindowTitle("MONTE Orbiter Toolkit")

        self.pause_state = False
        self.setFixedSize(1000, 700)
        font = QFont("Arial", 10)
        self.setFont(font)
        self.bg_color = "#724f76"
        self.setObjectName("MainWindow")
        self.setStyleSheet("""
            QWidget#MainWindow {
                background-color: #2e2e2e;  
            }
        """)

        self.dynamic_view_inputs = []
        self.dynamic_oe_inputs = []

        layout = QHBoxLayout()

        left_col = QVBoxLayout()
        left_col.addWidget(self.primary_perturbations_box())
        left_col.addWidget(self.spacecraft_properties_box())
        left_col.addWidget(self.ground_stations_box())
        left_col.addWidget(self.plotting_box())
        left_col.addWidget(self.export_directory_box())

        right_col = QVBoxLayout()
        right_col.addWidget(self.orbital_elements_box())
        right_col.addWidget(self.epoch_duration_box())
        right_col.addWidget(self.output_box())
        right_col.addWidget(self.start_simulation_box())

        layout.addLayout(left_col)
        layout.addLayout(right_col)

        self.setLayout(layout)

        self.check_script_timer = QTimer()
        self.check_script_timer.timeout.connect(self.check_if_script_finished)

    def apply_background(self, widget, color):
        widget.setStyleSheet(f"""
            QGroupBox {{
                background-color: {color};
                border: 1px solid black;
                margin-top: 20px;
                font-weight: bold;
                color: white;
            }}
            QGroupBox:title {{
                subcontrol-origin: margin;
                left: 10px;
                top: 0px;
                padding: 0 3px;
            }}
        """)

    def apply_epoch_background(self, widget, color):
        widget.setStyleSheet(f"""
            QGroupBox {{
                background-color: {color}; 
                border-radius: 4px; 
                padding: 6px;
            }}
        """)

    def collect_and_run(self):
        inputs = {}

        def collect_from_groupbox(group: QGroupBox):
            group_title = group.title().strip(":").replace(" ", "")
            layout = group.layout()
            if not layout:
                return

            label_text = None

            def collect_from_layout(layout):
                nonlocal label_text

                for i in range(layout.count()):
                    item = layout.itemAt(i)
                    sublayout = item.layout()
                    widget = item.widget()

                    if sublayout:
                        collect_from_layout(sublayout)

                    elif isinstance(widget, QLabel):
                        text = widget.text().strip(':')
                        if text:
                            label_text = text

                    elif isinstance(widget, QLineEdit) and label_text:
                        key = f"{group_title}_{label_text}"
                        inputs[key] = widget.text()
                        label_text = None

                    elif isinstance(widget, QComboBox) and label_text:
                        key = f"{group_title}_{label_text}"
                        inputs[key] = widget.currentText()
                        label_text = None

                    elif isinstance(widget, QCheckBox) and label_text:
                        key = f"{group_title}_{label_text}"
                        inputs[key] = widget.isChecked()
                        label_text = None

                    elif hasattr(widget, "get_state") and hasattr(widget, "name"):
                        key = f"{group_title}_{widget.name}"
                        inputs[key] = widget.get_state()
                        label_text = None

            collect_from_layout(layout)

        for group in self.findChildren(QGroupBox):
            collect_from_groupbox(group)

        for label, widget in self.dynamic_oe_inputs:
            key = f"InitialOrbitalElements_{label.strip(':')}"
            inputs[key] = widget.text()

        for label, widget in self.dynamic_view_inputs:
            key = f"Plotting_{label.strip(':')}"
            inputs[key] = widget.text()

        selected_predefs = [item.text() for item in self.predef_list.selectedItems()]
        inputs["GroundStations_Predefined"] = selected_predefs

        for i in range(self.gs_layout.count()):
            custom_group = self.gs_layout.itemAt(i).widget()
            if isinstance(custom_group, QGroupBox):
                line_edits = custom_group.findChildren(QLineEdit)
                labels = custom_group.findChildren(QLabel)[1:]
                grayed_out = all(not le.isEnabled() for le in line_edits)
                inputs[f"CustomGS{i + 1}_GrayedOut"] = grayed_out
                for label, le in zip(labels, line_edits):
                    key = f"CustomGS{i + 1}_{label.text().strip(':')}"
                    inputs[key] = le.text()

        if self.oe_mode_button.isChecked():
            inputs["InitialOrbitalElements_DefinitionMode"] = "Orbital Elements"
            inputs["InitialOrbitalElements_Type"] = self.oe_type_combo.currentText()
        else:
            inputs["InitialOrbitalElements_DefinitionMode"] = "Special Orbit"
            inputs["InitialOrbitalElements_Type"] = self.special_orbit_combo.currentText()

        if len(self.active_epoch_btns) >= 2:
            selected_mode = self.get_selected_duration_mode()

            def extract_epoch_row(layout):
                if layout.count() >= 3:
                    label_widget = layout.itemAt(1).widget()
                    input_widget = layout.itemAt(2).widget()
                    if isinstance(label_widget, QLabel) and isinstance(input_widget, QLineEdit):
                        label_text = label_widget.text().strip(":")
                        inputs[f"EpochDuration_{label_text}"] = input_widget.text()

                    if layout.count() >= 4:
                        unit_widget = layout.itemAt(3).widget()
                        if isinstance(unit_widget, QComboBox):
                            inputs[f"EpochDuration_{label_text}_Units"] = unit_widget.currentText()

            extract_epoch_row(self.input_row_1)
            extract_epoch_row(self.input_row_2)

        if self.row_widgets.get("timeStep"):
            inputs["EpochDuration_SimulationTimeStep"] = self.row_widgets["timeStep"].text()
        if self.row_widgets.get("timeUnit"):
            inputs["EpochDuration_SimulationTimeStep_Units"] = self.row_widgets["timeUnit"].currentText()

        if hasattr(self, "dir_display"):
            inputs["ExportDirectory_Path"] = self.dir_display.text()

        with open("input_data.json", "w") as f:
            json.dump(inputs, f, indent=2)

        run_script("monteSimulation.py", "input_data.json")
        self.pause_state = False
        self.pause_btn.setText("\u275A\u275A")
        self.pause_btn.setStyleSheet("""
            QPushButton {
                background-color: blue; 
                font-weight: bold;
                color: black;
                font-family: Arial;
                font-size: 8pt;
            }
            QPushButton:hover {
                background-color: lightblue; 
                font-weight: bold;
            }               
            """)

        self.check_script_timer.start(1000)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: gray; 
                font-weight: bold;
                color: gray;
                font-family: Arial;
                font-size: 10pt;
            }        
            """)

        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #bf1029; 
                font-weight: bold;
                color: black;
                font-family: Arial;
                font-size: 14pt;
            }
            QPushButton:hover {
                background-color: red; 
                font-weight: bold;
            }               
            """)

    def pause_or_resume_script(self):

        global is_paused
        if process_handle is None or process_handle.poll() is not None:
            print()
            print()
            cprint("Simulation Not Running.","yellow")
            return
        self.pause_btn.setText("\u275A\u275A")
        self.pause_btn.setStyleSheet("""
            QPushButton {
                background-color: gray; 
                font-weight: bold;
                color: gray;
                font-family: Arial;
                font-size: 8pt;
            }        
            """)

        if self.pause_state:

            resume_script()
            self.pause_btn.setText("\u275A\u275A")
            self.pause_btn.setStyleSheet("""
                QPushButton {
                    background-color: blue; 
                    font-weight: bold;
                    color: black;
                    font-family: Arial;
                    font-size: 8pt;
                }
                QPushButton:hover {
                    background-color: lightblue; 
                    font-weight: bold;
                }               
                """)

            self.start_btn.setStyleSheet("""
                QPushButton {
                    background-color: gray; 
                    font-weight: bold;
                    color: gray;
                    font-family: Arial;
                    font-size: 10pt;
                }        
                """)

            self.stop_btn.setStyleSheet("""
                QPushButton {
                    background-color: #bf1029; 
                    font-weight: bold;
                    color: black;
                    font-family: Arial;
                    font-size: 14pt;
                }
                QPushButton:hover {
                    background-color: red; 
                    font-weight: bold;
                }               
                """)

        else:

            pause_script()
            self.pause_btn.setText("\u25B6")
            self.pause_btn.setStyleSheet("""
                QPushButton {
                    background-color: blue; 
                    font-weight: bold;
                    color: black;
                    font-family: Arial;
                    font-size: 10pt;
                }
                QPushButton:hover {
                    background-color: lightblue; 
                    font-weight: bold;
                }               
                """)

            self.start_btn.setStyleSheet("""
                QPushButton {
                    background-color: gray; 
                    font-weight: bold;
                    color: gray;
                    font-family: Arial;
                    font-size: 10pt;
                }        
                """)

            self.stop_btn.setStyleSheet("""
                QPushButton {
                    background-color: gray; 
                    font-weight: bold;
                    color: gray;
                    font-family: Arial;
                    font-size: 14pt;
                }        
                """)

        self.pause_state = not self.pause_state

    def stop_script(self):

        terminate_script()
        self.pause_state = False
        self.pause_btn.setText("\u275A\u275A")
        self.pause_btn.setStyleSheet("""
            QPushButton {
                background-color: gray; 
                font-weight: bold;
                color: gray;
                font-family: Arial;
                font-size: 8pt;
            }        
            """)

        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: green; 
                font-weight: bold;
                color: black;
                font-family: Arial;
                font-size: 10pt;
            }
            QPushButton:hover {
                background-color: lightgreen; 
                font-weight: bold;
            }               
            """)
        
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: gray; 
                font-weight: bold;
                color: gray;
                font-family: Arial;
                font-size: 14pt;
            }        
            """)

    def check_if_script_finished(self):
        global process_handle
        if process_handle and process_handle.poll() is not None:
            os.system("clear")
            cprint("Simulation Complete","green")
            process_handle = None
            self.pause_state = False
            self.pause_btn.setText("\u275A\u275A")
            self.pause_btn.setStyleSheet("""
                QPushButton {
                    background-color: gray; 
                    font-weight: bold;
                    color: gray;
                    font-family: Arial;
                    font-size: 8pt;
                }        
                """)
            
            self.start_btn.setStyleSheet("""
                QPushButton {
                    background-color: green; 
                    font-weight: bold;
                    color: black;
                    font-family: Arial;
                    font-size: 10pt;
                }
                QPushButton:hover {
                    background-color: lightgreen; 
                    font-weight: bold;
                }               
                """)
            
            self.stop_btn.setStyleSheet("""
                QPushButton {
                    background-color: gray; 
                    font-weight: bold;
                    color: gray;
                    font-family: Arial;
                    font-size: 14pt;
                }        
                """)

            self.check_script_timer.stop()

    def primary_perturbations_box(self):
        layout = QVBoxLayout()
        box = QGroupBox("Primary and Perturbations:")
        self.apply_background(box, self.bg_color)

        row1 = QHBoxLayout()
        self.primary_label = QLabel("Primary Body:")
        self.primary_label.setFixedWidth(200)
        row1.addWidget(self.primary_label)
        self.primary_input = QComboBox()
        self.primary_input.addItems(["Mercury", "Venus", "Earth", "Mars"])
        self.primary_input.setCurrentIndex(2)
        self.primary_input.setFixedWidth(250)
        self.primary_input.setStyleSheet("""
            QComboBox:hover {
                background-color: #9fe4e5;
            }               
            """)
        row1.addWidget(self.primary_input)
        row1.addStretch()
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.atmosphere_label = QLabel("Atmosphere Effects:")
        self.atmosphere_label.setFixedWidth(200)
        row2.addWidget(self.atmosphere_label)
        self.atmosphere_toggle = self.named_toggle(name="AtmosphereEffects")
        row2.addWidget(self.atmosphere_toggle)
        row2.addStretch()
        layout.addLayout(row2)

        box.setLayout(layout)
        box.setFixedHeight(90)
        return box

    def spacecraft_properties_box(self):
        box = QGroupBox("Spacecraft Physical Properties:")
        self.apply_background(box, self.bg_color)
        layout = QVBoxLayout()
        for label in ["On Orbit Mass [kg]:", "Drag Area [m²]:", "Conical Sensor FOV [deg]:"]:
            row = QHBoxLayout()
            propIn = QLabel(label)
            propIn.setFixedHeight(20)
            propIn.setFixedWidth(200)
            row.addWidget(propIn)
            le = QLineEdit()
            le.setStyleSheet("""
                QLineEdit { 
                    padding: 4px;
                }
                QLineEdit:hover { 
                    background-color: #9fe4e5;
                }
                """)
            
            le.setFixedHeight(20)
            le.setFixedWidth(250)
            row.addWidget(le)
            row.addStretch()
            layout.addLayout(row)
            if label == "On Orbit Mass [kg]:":
                le.setText("100.0")
            if label == "Drag Area [m²]:":
                le.setText("5.00")
            if label == "Conical Sensor FOV [deg]:":
                le.setText("1.00")
        box.setLayout(layout)
        box.setFixedHeight(120)
        return box
    
    def ground_stations_box(self):
        box = QGroupBox("Ground Stations:")
        self.apply_background(box, self.bg_color)
        outer_layout = QHBoxLayout()
    
        left_col = QVBoxLayout()
        label = QLabel("   Predefined Stations:")
        label_font = QFont("Arial", 10)
        label_font.setBold(True)
        label.setFont(label_font)
        left_col.addSpacing(5)
        left_col.addWidget(label)
        left_col.addSpacing(4)
        self.predef_scroll_container, self.predef_list = self.styled_list()

        stations = M.HorizonMaskBoa.getAll(boa)

        for station in stations:
            if station[:3]=="DSS":
                self.predef_list.addItem(station)

        for station in stations:
            if not station[:3]=="DSS":
                self.predef_list.addItem(station[:-2])

        left_col.addWidget(self.predef_scroll_container)
        left_col.addStretch()
        outer_layout.addLayout(left_col)
    
        self.right_col = QVBoxLayout()
        custom_button_row = QHBoxLayout()
        self.add_gs_button = QPushButton("+ Add Custom GS")
        self.add_gs_button.setFixedWidth(125)
        self.add_gs_button.setStyleSheet("padding: 2px 6px;")
        self.add_gs_button.setStyleSheet("""
            QPushButton {
                background-color: lightgray;
            }
            QPushButton:hover {
                background-color: #9fe4e5;
            }               
            """)
        
        self.add_gs_button.clicked.connect(self.handle_add_custom_button)
    
        self.custom_stations_label = QLabel("Custom Stations:")
        label_font = QFont("Arial", 10)
        label_font.setBold(True)
        self.custom_stations_label.setFont(label_font)
        custom_button_row.addSpacing(10)
        custom_button_row.addWidget(self.custom_stations_label)
        custom_button_row.addSpacing(10)
        custom_button_row.addWidget(self.add_gs_button)
        custom_button_row.addSpacing(10)
    
        self.right_col.addLayout(custom_button_row)
    
        self.gs_scroll = FreezableScrollArea()
        self.gs_scroll.setWidgetResizable(True)
        self.gs_scroll.setStyleSheet(f"""
            QScrollArea {{
                background-color: {self.bg_color}; 
                border: none; 
            }}
            QScrollBar:vertical {{
                border: none;
                background: #f0f0f0;
                width: 10px;
                margin: 0px 0px 0px 0px;
            }}
            QScrollBar::handle:vertical {{
                background: #555;
                min-height: 20px;
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: #9fe4e5;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                background: none;
                height: 0px;
            }}
            """)
    
        self.gs_content = QWidget()
        self.gs_content.setStyleSheet(f"background-color: {self.bg_color};")
        self.gs_layout = QVBoxLayout()
        self.gs_content.setLayout(self.gs_layout)
        self.gs_scroll.setWidget(self.gs_content)
    
        self.right_col.addWidget(self.gs_scroll)
        outer_layout.addLayout(self.right_col)
        box.setLayout(outer_layout)
        box.setFixedHeight(220)
    
        self.gs_count = 0

        if self.gs_count == 0:
            self.custom_stations_label.setStyleSheet("color: #6e6e6e")
        else:
            self.custom_stations_label.setStyleSheet("color: black")
        self.first_station_inputs = []
        self.first_station_locked = True
        self.add_ground_station(grayed_out=True)
        self.freeze_scrollbar(True)

        QTimer.singleShot(0,lambda: self.freeze_scrollbar(True,scroll_to=0.8))
    
        return box
    
    def styled_list(self):
        container = QWidget()
        container.setFixedWidth(160)

        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        list_widget = QListWidget()
        list_widget.setFixedWidth(120)
        list_widget.setFixedHeight(147)
        list_widget.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        list_widget.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        list_widget.setUniformItemSizes(True)

        list_widget.setStyleSheet("""
            QListWidget {{
                border: 1px solid black;
                background-color: white;
            }}
            QListView::item {{
                border-bottom: 1px solid black;
                padding: 3px;
            }}
            QListView::item:hover {{
                background-color: #cceeff;
            }}
            QListView::item:selected {{
                background-color: #66ccff;
                color: black;
            }}
        """)

        scrollbar = QScrollBar(Qt.Orientation.Vertical)
        scrollbar.setFixedWidth(10)
        scrollbar.setFixedHeight(147)
        scrollbar.setStyleSheet("""
            QScrollBar:vertical {
                background: white;
            }
            QScrollBar::handle:vertical {
                background: #555;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background: #9fe4e5;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                background: none;
                height: 0px;
            }
        """)

        internal_scroll = list_widget.verticalScrollBar()

        def sync_range():
            scrollbar.setRange(internal_scroll.minimum(), internal_scroll.maximum())
            scrollbar.setPageStep(internal_scroll.pageStep())
            scrollbar.setSingleStep(internal_scroll.singleStep())

        # One-time range sync on load
        sync_range()

        # Two-way scroll sync
        internal_scroll.valueChanged.connect(scrollbar.setValue)
        scrollbar.valueChanged.connect(internal_scroll.setValue)
        internal_scroll.rangeChanged.connect(sync_range)

        layout.addWidget(list_widget)
        layout.addWidget(scrollbar)

        return container, list_widget

    def handle_add_custom_button(self):
        if self.first_station_locked:
            for gs_group_label, dyn_label, le in self.first_station_inputs:
                gs_group_label.setStyleSheet("color: black")
                dyn_label.setStyleSheet("color: black")
                le.setEnabled(True)
                le.setStyleSheet("""
                    QLineEdit { 
                        padding: 3px;
                        background-color: white;
                    }
                    QLineEdit:hover { 
                        background-color: #9fe4e5;
                    }
                    """)
            self.gs_group.setStyleSheet(
                f"QGroupBox {{ background-color: {self.bg_color}; border: 1px solid black; padding: 6px; }}"
            )  
            self.first_station_locked = False
            self.freeze_scrollbar(False, scroll_to=0.8)
            self.custom_stations_label.setStyleSheet("color: black")
        else:
            self.add_ground_station()
            self.gs_group.setStyleSheet(
                f"QGroupBox {{ background-color: {self.bg_color}; border: 1px solid black; padding: 6px; }}"
            )            
    
    def add_ground_station(self, grayed_out=False):
        self.gs_count += 1
        self.gs_group = QGroupBox()

        if grayed_out:
            self.gs_group.setStyleSheet(
                f"QGroupBox {{ background-color: {self.bg_color}; border: 1px solid #999; padding: 6px; }}"
            )
        else:
            self.gs_group.setStyleSheet(
                f"QGroupBox {{ background-color: {self.bg_color}; border: 1px solid black; padding: 6px; }}"
            )

        layout = QGridLayout()
        gs_group_label = QLabel(f"Custom Station {self.gs_count}:")
        gs_group_label_font = QFont("Arial", 10)
        gs_group_label_font.setBold(True)
        gs_group_label.setFont(gs_group_label_font)
        gs_group_label.setStyleSheet("color: #6e6e6e" if grayed_out else "color: black")
        layout.addWidget(gs_group_label, 0, 0)
    
        labels = ["Latitude [deg]", "Longitude [deg]", "Altitude [m]", "Min. Elevation [deg]"]
    
        for i, label in enumerate(labels):
            gsInputHeight = 19
    
            dyn_label = QLabel(f"{label}:")
            dyn_label.setFixedHeight(gsInputHeight)
            dyn_label.setFixedWidth(130)
            dyn_label.setStyleSheet("color: #6e6e6e" if grayed_out else "color: black")
            layout.addWidget(dyn_label, i + 1, 0)
    
            le = QLineEdit()
            le.setFixedHeight(gsInputHeight)
            le.setFixedWidth(90)
    
            if self.gs_count == 1:
                default_values = ["34.20", "118.17", "183.0", "20.0"]
                le.setText(default_values[i])
            else:
                le.setText("0.00")
    
            if grayed_out:
                le.setEnabled(False)
                le.setStyleSheet("padding: 3px; background-color: lightgray")
                
                self.first_station_inputs.append((gs_group_label, dyn_label, le))
            else:
                le.setStyleSheet("""
                    QLineEdit { 
                        padding: 3px;
                        background-color: white;
                    }
                    QLineEdit:hover { 
                        background-color: #9fe4e5;
                    }
                    """)
    
            layout.addWidget(le, i + 1, 1)
    
        self.gs_group.setLayout(layout)
        self.gs_layout.addWidget(self.gs_group)
    
    def freeze_scrollbar(self, freeze: bool, scroll_to: float = None):
        self.gs_scroll.freeze(freeze)
        if scroll_to is not None:
            self.set_scroll_percent(scroll_to)

    def set_scroll_percent(self, percent: float):
        bar = self.gs_scroll.verticalScrollBar()
        max_value = bar.maximum()
        bar.setValue(int(max_value*percent))

    def plotting_box(self):
        box = QGroupBox("Plotting:")
        self.apply_background(box, self.bg_color)
        layout = QHBoxLayout()

        toggle_layout = QVBoxLayout()
        for option in ["Ground Track", "3-D Vizualization", "Orbital Elements", "Animate Plots", "Export Plots"]:
            row = QHBoxLayout()
            if option in ["Ground Track", "3-D Vizualization", "Orbital Elements"]:
                initialState=True
            else:
                initialState=False
            toggle = self.named_toggle(option,initialState)
            plotInLab = QLabel(option+":")
            plotInLab.setFixedWidth(105)
            row.addWidget(plotInLab)
            row.addWidget(toggle)
            row.addStretch()
            toggle_layout.addLayout(row)
        layout.addLayout(toggle_layout)

        view_layout = QVBoxLayout()
        view_type_row = QHBoxLayout()
        view_type_label = QLabel("  3-D View:")
        view_type_label.setFixedWidth(70)
        self.view_type_combo = QComboBox()
        self.view_type_combo.addItems(["Primary Centered Inertial", "Primary Centered Fixed", "Orbit Plane", "Satellite View"])
        self.view_type_combo.currentTextChanged.connect(self.update_view_type_fields)
        self.view_type_combo.setFixedWidth(185)
        self.view_type_combo.setStyleSheet("""
            QComboBox:hover {
                background-color: #9fe4e5;
            }               
            """)
        view_type_row.addWidget(view_type_label)
        view_type_row.addWidget(self.view_type_combo)
        view_type_row.addStretch()
        view_layout.addLayout(view_type_row)

        self.view_inputs_container = QVBoxLayout()
        self.view_inputs_widget = QWidget()
        self.view_inputs_widget.setLayout(self.view_inputs_container)
        self.view_inputs_widget.setFixedHeight(100)
        self.view_inputs_widget.setMinimumWidth(270)
        self.view_inputs_widget.setMaximumWidth(270)
        view_layout.addWidget(self.view_inputs_widget)

        layout.addLayout(view_layout)
        box.setLayout(layout)

        self.update_view_type_fields("Primary Centered Inertial")
        box.setFixedHeight(160)
        return box

    def update_view_type_fields(self, view_type):
        self.dynamic_view_inputs.clear()
        view_labels = {
            "Primary Centered Inertial": ["X-Position [km]:", "Y-Position [km]:", "Z-Position [km]:"],
            "Primary Centered Fixed": ["Latitude [deg]:", "Longitude [deg]:", "Altitude [km]:"],
            "Orbit Plane": ["Elevation [deg]:", "Azimuth [deg]:", "Radius [km]:"],
            "Satellite View": []
        }

        while self.view_inputs_container.count():
            item = self.view_inputs_container.takeAt(0)
            if item.layout():
                while item.layout().count():
                    sub_item = item.layout().takeAt(0)
                    if sub_item.widget():
                        sub_item.widget().deleteLater()
            elif item.widget():
                item.widget().deleteLater()

        for label_text in view_labels.get(view_type, []):
            row = QHBoxLayout()
            dynLab = QLabel(label_text)
            dynLab.setFixedWidth(100)
            dynLab.setFixedHeight(20)
            row.addWidget(dynLab)
            le = QLineEdit()
            le.setStyleSheet("""
                QLineEdit { 
                    padding: 4px;
                    background-color: white;
                }
                QLineEdit:hover { 
                    background-color: #9fe4e5;
                }
                """)
            le.setFixedHeight(20)
            row.addWidget(le)
            self.view_inputs_container.addLayout(row)
            if label_text != "":
                self.dynamic_view_inputs.append((label_text.strip(), le))
            else:
                le.setVisible(False)
            if label_text == "X-Position [km]:":
                le.setText("7500")
            if label_text == "Y-Position [km]:":
                le.setText("7500")
            if label_text == "Z-Position [km]:":
                le.setText("7500")
            if label_text == "Latitude [deg]:":
                le.setText("0.00")
            if label_text == "Longitude [deg]:":
                le.setText("0.00")
            if label_text == "Altitude [km]:":
                le.setText("2000")
            if label_text == "Elevation [deg]:":
                le.setText("45.00")
            if label_text == "Azimuth [deg]:":
                le.setText("0.00")
            if label_text == "Radius [km]:":
                le.setText("10000")

    def orbital_elements_box(self):
        spacing = 6  # Controls vertical spacing between rows

        box = QGroupBox("Initial Orbital Elements:")
        self.apply_background(box, self.bg_color)
        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(spacing)

        # === Mode Selection Row ===
        mode_selector_row = QHBoxLayout()
        mode_selector_row.setContentsMargins(0,0,0,0)
        mode_selector_row.setSpacing(10)
        mode_label = QLabel("  Define Orbit Using:")
        mode_label.setFixedWidth(183)

        self.oe_mode_button = QPushButton("Orbital Elements")
        self.special_mode_button = QPushButton("Special Orbit")

        for b in [self.oe_mode_button, self.special_mode_button]:
            b.setFixedWidth(120)
            b.setFixedHeight(20)
            b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)

        self.oe_mode_button.clicked.connect(lambda: self.update_definition_mode("oe"))
        self.special_mode_button.clicked.connect(lambda: self.update_definition_mode("special"))

        mode_selector_row.addSpacing(14)
        mode_selector_row.addWidget(mode_label)
        mode_selector_row.addWidget(self.oe_mode_button)
        mode_selector_row.addWidget(self.special_mode_button)
        mode_selector_row.addStretch()

        # === Orbital Elements Type Row ===
        self.oe_type_row_widget = QWidget()
        self.oe_type_row_widget.setContentsMargins(0, 0, 0, 0)
        oe_type_row = QHBoxLayout()
        oe_type_row.setContentsMargins(0, 0, 0, 0)
        oe_type_row.setSpacing(0)
        self.oe_type_label = QLabel("  Orbital Elements Type:")
        self.oe_type_label.setFixedWidth(192)
        self.oe_type_combo = QComboBox()
        self.oe_type_combo.addItems(["Keplarian", "Cartesian", "Spherical", "Equinoctial"])
        self.oe_type_combo.setFixedWidth(252)
        self.oe_type_combo.currentTextChanged.connect(self.update_orbital_elements_inputs)
        self.oe_type_combo.setStyleSheet("""
            QComboBox:hover {
                background-color: #9fe4e5;
            }
            """)
        oe_type_row.addSpacing(14)
        oe_type_row.addWidget(self.oe_type_label)
        oe_type_row.addWidget(self.oe_type_combo)
        oe_type_row.addStretch()
        self.oe_type_row_widget.setLayout(oe_type_row)

        # === Special Orbit Row ===
        self.special_orbit_row_widget = QWidget()
        self.special_orbit_row_widget.setContentsMargins(0, 0, 0, 0)
        special_orbit_row = QHBoxLayout()
        special_orbit_row.setContentsMargins(0, 0, 0, 0)
        special_orbit_row.setSpacing(0)
        self.special_orbit_label = QLabel("  Special Orbit:")
        self.special_orbit_label.setFixedWidth(192)
        self.special_orbit_combo = QComboBox()
        self.special_orbit_combo.addItems(["Sun Synchronous Orbit", "Frozen Orbit", "Repeat Ground Track",
                                           "Geosynchronous", "Molniya", "Tundra"])
        self.special_orbit_combo.setFixedWidth(252)
        self.special_orbit_combo.currentTextChanged.connect(self.update_orbital_elements_inputs)
        self.special_orbit_combo.setStyleSheet("""
            QComboBox:hover {
                background-color: #9fe4e5;
            }
            """)
        special_orbit_row.addSpacing(14)
        special_orbit_row.addWidget(self.special_orbit_label)
        special_orbit_row.addWidget(self.special_orbit_combo)
        special_orbit_row.addStretch()
        self.special_orbit_row_widget.setLayout(special_orbit_row)

        # === Reference Frame Row ===
        rf_row = QHBoxLayout()
        rf_row.setContentsMargins(0,0,0,0)
        rf_row.setSpacing(spacing)
        rf_label = QLabel("  Reference Frame:")
        rf_label.setFixedWidth(186)
        self.rf_combo = QComboBox()
        self.rf_combo.setFixedWidth(251)
        
        frames = M.CoordFrameBoa.getAll(boa)

        self.rf_combo.addItems(frames)
        self.rf_combo.insertItem(20, "EME2000")
        self.rf_combo.setCurrentIndex(20)
        self.rf_combo.setStyleSheet("""
            QComboBox:hover {
                background-color: #9fe4e5;
            }
            """)
        rf_row.addSpacing(14)
        rf_row.addWidget(rf_label)
        rf_row.addWidget(self.rf_combo)
        rf_row.addStretch()

        # === OE Input Rows ===
        self.oe_inputs_container = QVBoxLayout()
        self.oe_inputs_widget = QWidget()
        self.oe_inputs_widget.setLayout(self.oe_inputs_container)
        self.oe_inputs_container.setContentsMargins(0,0,0,0)
        self.oe_inputs_widget.setFixedHeight(225)

        # === Layout Assembly ===
        layout.addSpacing(10)
        layout.addLayout(mode_selector_row)
        layout.addWidget(self.oe_type_row_widget)
        layout.addWidget(self.special_orbit_row_widget)
        layout.addLayout(rf_row)
        layout.addWidget(self.oe_inputs_widget)
        layout.addStretch()

        box.setLayout(layout)
        box.setFixedHeight(345)

        self.update_definition_mode("oe")
        return box

    def update_definition_mode(self, mode):
        if mode == "oe":
            self.oe_mode_button.setChecked(True)
            self.oe_mode_button.setStyleSheet("""
                QPushButton {
                    background-color: #7bb1b2;
                }
                QPushButton:hover {
                    background-color: #9fe4e5;
                }
            """)
            self.special_mode_button.setChecked(False)
            self.special_mode_button.setStyleSheet("""
                QPushButton {
                    background-color: lightgray;
                }
                QPushButton:hover {
                    background-color: #9fe4e5;
                }
            """)

            self.oe_type_row_widget.show()
            self.special_orbit_row_widget.hide()
            self.update_orbital_elements_inputs(self.oe_type_combo.currentText())

        elif mode == "special":
            self.oe_mode_button.setChecked(False)
            self.oe_mode_button.setStyleSheet("""
                QPushButton {
                    background-color: lightgray;
                }
                QPushButton:hover {
                    background-color: #9fe4e5;
                }
            """)
            self.special_mode_button.setChecked(True)
            self.special_mode_button.setStyleSheet("""
                QPushButton {
                    background-color: #7bb1b2;
                }
                QPushButton:hover {
                    background-color: #9fe4e5;
                }
            """)

            self.oe_type_row_widget.hide()
            self.special_orbit_row_widget.show()
            self.update_orbital_elements_inputs(self.special_orbit_combo.currentText())

    def update_orbital_elements_inputs(self, oe_type):
        self.dynamic_oe_inputs.clear()
        field_map = {
            "Keplarian": ["Semi-Major Axis [km]:", "Eccentricity:", "Inclination [deg]:",
                          "RAAN [deg]:", "Argument of Periapsis [deg]:", "True Anomaly [deg]:", "", "",""],
            "Cartesian": ["X Position [km]:", "Y Position [km]:", "Z Position [km]:",
                          "X Velocity [km/s]:", "Y Velocity [km/s]:", "Z Velocity [km/s]:", "", "",""],
            "Spherical": ["Radius [km]:", "Latitude [deg]:", "Longitude [deg]:", "Radial Velocity [km/s]:", 
                          "Latitudinal Velocity [deg/s]:","Longitudinal Velocity [deg/s]:", "", "",""],
            "Equinoctial": ["SMA [km]:","H:","K:","Lambda [deg]:","P:","Q:","Alpha [deg/s]:","dLambda [deg/s]:","Gamma [deg/s]:"],
            "Geosynchronous": ["Target Longitude [deg]:","","","","","", "", "",""],
            "Molniya": ["Target Longitude [deg]:", "", "","","","", "", "",""],
            "Tundra": ["Target Longitude [deg]:", "", "","","","", "", "",""],
            "Sun Synchronous Orbit": ["Altitude [km]:", "MLTAN [hours]:", "True Anomaly [deg]:", "", "","", "", "",""],
            "Frozen Orbit": ["Semi-Major Axis [km]:", "Inclination [deg]:", "RAAN [deg]:", "Argument of Periapsis [deg]:", 
                             "True Anomaly [deg]:", "", "", "",""],
            "Repeat Ground Track": ["Repeat Time [days]:", "Repeat Cycles:", "Inclination [deg]:", "", "", "", "", "",""]
        }

        fields = field_map.get(oe_type, [])
        while self.oe_inputs_container.count():
            item = self.oe_inputs_container.takeAt(0)
            if item.layout():
                while item.layout().count():
                    sub_item = item.layout().takeAt(0)
                    if sub_item.widget():
                        sub_item.widget().deleteLater()
            elif item.widget():
                item.widget().deleteLater()

        for label_text in fields:
            row = QHBoxLayout()
            label = QLabel(label_text)
            label.setFixedWidth(180)
            label.setFixedHeight(20)
            le = QLineEdit()
            le.setStyleSheet("""
                QLineEdit { 
                    padding: 4px;
                    background-color: white;
                }
                QLineEdit:hover { 
                    background-color: #9fe4e5;
                }
                """)
            le.setFixedHeight(20)
            le.setFixedWidth(252)
            row.addSpacing(20)
            row.addWidget(label)
            row.addWidget(le)
            row.addStretch()
            self.oe_inputs_container.addLayout(row)
            if label_text != "":
                self.dynamic_oe_inputs.append((label_text.strip(), le))
            else:
                le.setVisible(False)

            if oe_type == "Keplarian":
                if label_text == "Semi-Major Axis [km]:":
                    le.setText("6778.0")
                if label_text == "Eccentricity:":
                    le.setText("0.000")
                if label_text == "Inclination [deg]:":
                    le.setText("51.600")
                if label_text == "RAAN [deg]:":
                    le.setText("0.000")
                if label_text == "Argument of Periapsis [deg]:":
                    le.setText("0.000")
                if label_text == "True Anomaly [deg]:":
                    le.setText("0.000")

            if oe_type == "Cartesian":
                if label_text == "X Position [km]:":
                    le.setText("0.0")
                if label_text == "Y Position [km]:":
                    le.setText("0.0")
                if label_text == "Z Position [km]:":
                    le.setText("6778.0")
                if label_text == "X Velocity [km/s]:":
                    le.setText("7.673")
                if label_text == "Y Velocity [km/s]:":
                    le.setText("0.000")
                if label_text == "Z Velocity [km/s]:":
                    le.setText("0.000")

            if oe_type == "Spherical":
                if label_text == "Radius [km]:":
                    le.setText("6778.0")
                if label_text == "Latitude [deg]:":
                    le.setText("0.000")
                if label_text == "Longitude [deg]:":
                    le.setText("0.000")
                if label_text == "Radial Velocity [km/s]:":
                    le.setText("0.000")
                if label_text == "Latitudinal Velocity [deg/s]:":
                    le.setText("0.000")
                if label_text == "Longitudinal Velocity [deg/s]:":
                    le.setText("0.065")

            if oe_type == "Equinoctial":
                if label_text == "SMA [km]:":
                    le.setText("6778.0")
                if label_text == "H:":
                    le.setText("0.000")
                if label_text == "K:":
                    le.setText("0.000")
                if label_text == "Lambda [deg]:":
                    le.setText("0.000")
                if label_text == "P:":
                    le.setText("0.466")
                if label_text == "Q:":
                    le.setText("0.000")
                if label_text == "Alpha [deg/s]:":
                    le.setText("0.000")
                if label_text == "dLambda [deg/s]:":
                    le.setText("0.000")
                if label_text == "Gamma [deg/s]:":
                    le.setText("0.000")

            if oe_type == "Sun Synchronous Orbit":
                if label_text == "Altitude [km]:":
                    le.setText("500.0")
                if label_text == "MLTAN [hours]:":
                    le.setText("6.0")
                if label_text == "True Anomaly [deg]:":
                    le.setText("0.000")

            if oe_type == "Frozen Orbit":
                if label_text == "Semi-Major Axis [km]:":
                    le.setText("6778.0")
                if label_text == "Inclination [deg]:":
                    le.setText("51.600")
                if label_text == "RAAN [deg]:":
                    le.setText("0.000")
                if label_text == "Argument of Periapsis [deg]:":
                    le.setText("90.000")
                if label_text == "True Anomaly [deg]:":
                    le.setText("0.000")

            if oe_type == "Repeat Ground Track":
                if label_text == "Repeat Time [days]:":
                    le.setText("2")
                if label_text == "Repeat Cycles:":
                    le.setText("17")
                if label_text == "Repeat Cycles:":
                    le.setText("60.0")

            if oe_type == "Geosynchronous":
                le.setText("-135.000")
            
            if oe_type == "Molniya":
                le.setText("110.000")
            
            if oe_type == "Tundra":
                le.setText("75.000")

        self.oe_inputs_container.addStretch()

    def epoch_duration_box(self):
        box = QGroupBox("Epoch and Duration:")
        self.apply_background(box, self.bg_color)

        self.ed_main_layout = QVBoxLayout()
        self.ed_main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.row_widgets = {"input1": None, "input2": None, "timeStep": None}

        self.epoch_btns = {}
        self.active_epoch_btns = []

        top_btn_row = QHBoxLayout()
        self.ed_main_container = QWidget()

        top_btn_label = QLabel("Event Types:")
        top_btn_label.setFixedWidth(180)
        top_btn_row.addWidget(top_btn_label)

        for label in ["Initial Epoch", "Duration", "Final Epoch"]:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setFixedHeight(20)
            btn.setFixedWidth(86 if label != "Duration" else 71)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: lightgray;
                }
                QPushButton:hover {
                    background-color: #9fe4e5;
                }              
            """)
            btn.clicked.connect(self.handle_epoch_btn)
            self.epoch_btns[label] = btn
            top_btn_row.addWidget(btn)

        top_btn_row.addStretch()
        self.ed_main_container.setLayout(top_btn_row)
        self.ed_main_container.setFixedHeight(29)
        self.ed_main_layout.addWidget(self.ed_main_container)

        self.duration_mode_buttons = {}
        self.duration_mode_row_widget = QWidget()
        self.duration_mode_layout = QHBoxLayout()
        self.duration_mode_layout.setContentsMargins(0, 0, 0, 0)
        self.duration_mode_row_widget.setLayout(self.duration_mode_layout)
        self.duration_mode_row_widget.setVisible(False)
        self.duration_mode_row_widget.setFixedHeight(29)
        self.ed_main_layout.addWidget(self.duration_mode_row_widget)

        self.duration_mode_label = QLabel("Duration Type:")
        self.duration_mode_label.setFixedWidth(180)
        self.duration_mode_layout.addSpacing(10)
        self.duration_mode_layout.addWidget(self.duration_mode_label)

        for label in ["Number of Periods", "Timespan"]:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setFixedWidth(124)
            btn.setFixedHeight(20)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: lightgray;
                }
                QPushButton:hover {
                    background-color: #9fe4e5;
                }              
            """)
            btn.clicked.connect(self.handle_duration_mode)
            self.duration_mode_buttons[label] = btn
            self.duration_mode_layout.addWidget(btn)

        self.duration_mode_layout.addStretch()

        self.input_row_1 = QHBoxLayout()
        self.input_row_2 = QHBoxLayout()
        self.input_row_1.setContentsMargins(0, 0, 0, 0)
        self.input_row_2.setContentsMargins(0, 0, 0, 0)

        self.input_row_1_container = QWidget()
        self.input_row_2_container = QWidget()
        self.input_row_1_container.setFixedHeight(29)
        self.input_row_2_container.setFixedHeight(29)
        self.input_row_1_container.setLayout(self.input_row_1)
        self.input_row_2_container.setLayout(self.input_row_2)

        self.ed_main_layout.addWidget(self.input_row_1_container)
        self.ed_main_layout.addWidget(self.input_row_2_container)

        self.time_step_row = QHBoxLayout()
        self.time_step_row.setContentsMargins(0, 0, 0, 0)
        self.time_step_row.setSpacing(8)

        self.time_step_row_container = QWidget()
        self.time_step_row_container.setLayout(self.time_step_row)
        self.time_step_row_container.setFixedHeight(29)

        time_step_label = QLabel("Simulation Time Step:")
        time_step_label.setFixedWidth(178)
        time_step_label.setFixedHeight(20)

        time_step_input = QLineEdit()
        time_step_input.setText("60")
        time_step_input.setFixedWidth(160)
        time_step_input.setFixedHeight(20)
        time_step_input.setStyleSheet("""
            QLineEdit { 
                padding: 2px;
                background-color: white;
            }
            QLineEdit:hover { 
                background-color: #9fe4e5;
            }
            """)
        time_step_combo = QComboBox()
        time_step_combo.addItems(["Seconds", "Minutes", "Hours", "Days"])
        time_step_combo.setFixedWidth(87)
        time_step_combo.setFixedHeight(20)
        time_step_combo.setStyleSheet("""
            QComboBox:hover {
                background-color: #9fe4e5;
            }              
        """)

        self.row_widgets["timeStep"] = time_step_input
        self.row_widgets["timeUnit"] = time_step_combo

        self.time_step_row.addSpacing(10)
        self.time_step_row.addWidget(time_step_label)
        self.time_step_row.addWidget(time_step_input)
        self.time_step_row.addWidget(time_step_combo)
        self.time_step_row.addStretch()

        self.ed_main_layout.addWidget(self.time_step_row_container)

        self.activate_epoch_button("Initial Epoch")
        self.activate_epoch_button("Duration")
        self.activate_duration_mode("Number of Periods")

        box.setLayout(self.ed_main_layout)
        box.setFixedHeight(185)

        return box

    def handle_epoch_btn(self):
        sender = self.sender()
        label = sender.text()
        if label in self.active_epoch_btns:
            return
        self.activate_epoch_button(label)

    def activate_epoch_button(self, label):
        if label in self.active_epoch_btns:
            return

        if len(self.active_epoch_btns) == 2:
            removed_label = self.active_epoch_btns.pop(0)
            self.epoch_btns[removed_label].setChecked(False)
            self.epoch_btns[removed_label].setStyleSheet("""
                QPushButton {
                    background-color: lightgray;
                }
                QPushButton:hover {
                    background-color: #9fe4e5;
                }              
            """)

        self.active_epoch_btns.append(label)
        self.epoch_btns[label].setChecked(True)
        self.epoch_btns[label].setStyleSheet("""
            QPushButton {
                background-color: #7bb1b2;
            }
            QPushButton:hover {
                background-color: #9fe4e5;
            }              
        """)

        self.duration_mode_row_widget.setVisible("Duration" in self.active_epoch_btns)
        self.update_input_rows()

    def handle_duration_mode(self):
        sender = self.sender()
        label = sender.text()
        self.activate_duration_mode(label)

    def activate_duration_mode(self, label):
        for name, btn in self.duration_mode_buttons.items():
            if name == label:
                btn.setChecked(True)
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #7bb1b2;
                    }
                    QPushButton:hover {
                        background-color: #9fe4e5;
                    }              
                """)
            else:
                btn.setChecked(False)
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: lightgray;
                    }
                    QPushButton:hover {
                        background-color: #9fe4e5;
                    }              
                """)
        self.update_input_rows()

    def get_selected_duration_mode(self):
        for name, btn in self.duration_mode_buttons.items():
            if btn.isChecked():
                return name
        return "Number of Periods"

    def update_input_rows(self):
        def clear_layout(layout):
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()

        clear_layout(self.input_row_1)
        clear_layout(self.input_row_2)

        if len(self.active_epoch_btns) < 2:
            return

        label_map = {
            "Initial Epoch": "Initial Epoch:",
            "Final Epoch": "Final Epoch:",
            "Duration": {
                "Number of Periods": "Number of Orbital Periods:",
                "Timespan": "Timespan:"
            }
        }

        inputs = self.active_epoch_btns.copy()
        if "Initial Epoch" in inputs and "Final Epoch" in inputs:
            inputs.sort(key=lambda x: 0 if x == "Initial Epoch" else 1)

        label1 = inputs[0]
        label2 = inputs[1]
        mode = self.get_selected_duration_mode()

        def build_input_row(layout, label_key):
            layout.addSpacing(10)
            if label_key == "Duration":
                label_text = label_map["Duration"][mode]
            else:
                label_text = label_map[label_key]

            label = QLabel(label_text)
            label.setFixedWidth(180)
            label.setFixedHeight(20)
            layout.addWidget(label)

            if label_key == "Duration" and mode == "Timespan":
                input_width = 160
            else:
                input_width = 254

            line_edit = QLineEdit()
            line_edit.setFixedWidth(input_width)
            line_edit.setFixedHeight(20)
            line_edit.setStyleSheet("""
                QLineEdit:hover { 
                    background-color: #9fe4e5;
                }
                """)
            
            if label_key == "Duration" and mode == "Timespan":
                line_edit.setText("100000")
            if label_key == "Duration" and mode == "Number of Periods":
                line_edit.setText("150")
            if label_key == "Initial Epoch":
                line_edit.setText("01-JAN-2000 00:00:00 ET")
            if label_key == "Final Epoch":
                line_edit.setText("10-JAN-2000 00:00:00 ET")

            layout.addWidget(line_edit)

            if label_key == "Duration" and mode == "Timespan":
                combo = QComboBox()
                combo.addItems(["Seconds", "Minutes", "Hours", "Days"])
                combo.setFixedWidth(88)
                combo.setFixedHeight(20)
                combo.setStyleSheet("""
                    QComboBox:hover {
                        background-color: #9fe4e5;
                    }              
                """)
                layout.addWidget(combo)

            layout.addStretch()

        build_input_row(self.input_row_1, label1)
        build_input_row(self.input_row_2, label2)
    
    def output_box(self):
        box = QGroupBox("Data Output:")
        self.apply_background(box, self.bg_color)
        layout = QGridLayout()
        layout.addWidget(QLabel(""),0,0)

        contactEventsLabel = QLabel("Contact Events:")
        contactEventsLabel.setFixedWidth(110)
        contactEventsToggle = self.named_toggle("Contact Events",True)
        layout.addWidget(contactEventsLabel, 0, 1)
        layout.addWidget(contactEventsToggle, 0, 2)

        occultationEventsLabel = QLabel("Shadow Events:")
        occultationEventsLabel.setFixedWidth(130)
        occultationEventsToggle = self.named_toggle("Shadow Events",True)
        layout.addWidget(occultationEventsLabel, 0, 4)
        layout.addWidget(occultationEventsToggle, 0, 5)
        
        stretch = QHBoxLayout()
        stretch.addStretch()
        spacing = QHBoxLayout()
        spacing.addSpacing(50)
        layout.addLayout(spacing,0,3)
        layout.addLayout(stretch,0,6)
        box.setLayout(layout)
        box.setFixedHeight(65)
        return box

    def export_directory_box(self):
        box = QGroupBox("Export Directory:")
        self.apply_background(box, self.bg_color)
        layout = QGridLayout()
        self.dir_display = QLineEdit("/home/")
        self.dir_display.setStyleSheet("""
            QLineEdit { 
                padding: 2px;
                background-color: white;
            }
            QLineEdit:hover { 
                background-color: #9fe4e5;
            }
            """)
        self.dir_display.setFixedHeight(20)
        self.dir_display.setFixedWidth(305)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.select_directory)
        browse_btn.setFixedWidth(146)
        browse_btn.setStyleSheet("""
            QPushButton {
                background-color: lightgray;
            }
            QPushButton:hover {
                background-color: #9fe4e5;
            }               
            """)
        layout.addWidget(self.dir_display,0,0)
        layout.addWidget(browse_btn,0,1)
        stretch = QHBoxLayout()
        stretch.addStretch()
        layout.addLayout(stretch,0,2)
        box.setLayout(layout)
        box.setFixedHeight(67)
        return box

    def select_directory(self):
        dir_name = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if dir_name:
            self.dir_display.setText(dir_name)

    def start_simulation_box(self):
        box = QGroupBox("Run Orbit Simulation:")
        self.apply_background(box, self.bg_color)
        layout = QGridLayout()
        btn_width = 120

        self.start_btn = QPushButton("\u25B6")
        font = QFont("Arial",10)
        self.start_btn.setFont(font)
        self.start_btn.setFixedHeight(25)
        self.start_btn.setFixedWidth(btn_width)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: green; 
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: lightgreen; 
                font-weight: bold;
            }               
        """)
        self.start_btn.clicked.connect(self.collect_and_run)

        if process_handle is None or process_handle.poll() is not None:
            self.pause_btn = QPushButton("\u275A\u275A")
            self.pause_btn.setFixedHeight(25)
            self.pause_btn.setFixedWidth(btn_width)
            self.pause_btn.setStyleSheet("""
                QPushButton {
                    background-color: gray; 
                    font-weight: bold;
                    color: gray;
                    font-family: Arial;
                    font-size: 8pt;
                }        
            """)
        
        else:

            self.pause_btn = QPushButton("\u275A\u275A")
            self.pause_btn.setFixedHeight(25)
            self.pause_btn.setFixedWidth(btn_width)
            self.pause_btn.setStyleSheet("""
                QPushButton {
                    background-color: blue; 
                    font-weight: bold;
                    color: black;
                    font-family: Arial;
                    font-size: 8pt;
                }
                QPushButton:hover {
                    background-color: lightblue; 
                    font-weight: bold;
                }               
            """)

        self.pause_btn.clicked.connect(self.pause_or_resume_script)

        self.stop_btn = QPushButton("\u25A0")
        self.stop_btn.setFixedHeight(25)
        self.stop_btn.setFixedWidth(btn_width)
        if process_handle is None or process_handle.poll() is not None:
            self.stop_btn.setStyleSheet("""
                QPushButton {
                    background-color: gray; 
                    font-weight: bold;
                    color: gray;
                    font-family: Arial;
                    font-size: 14pt;
                }        
            """)

        else:

            self.stop_btn.setStyleSheet("""
                QPushButton {
                    background-color: #bf1029; 
                    font-weight: bold;
                    color: black;
                    font-family: Arial;
                    font-size: 14pt;
                }
                QPushButton:hover {
                    background-color: red; 
                    font-weight: bold;
                }               
            """)

        self.stop_btn.clicked.connect(self.stop_script)

        layout.addWidget(QLabel(""),0,0)
        layout.addWidget(self.start_btn,0,1)
        layout.addWidget(QLabel(""),0,2)
        layout.addWidget(self.pause_btn,0,3)
        layout.addWidget(QLabel(""),0,4)
        layout.addWidget(self.stop_btn,0,5)
        layout.addWidget(QLabel(""),0,6)
        box.setLayout(layout)
        box.setFixedHeight(67)
        return box
    
    def named_toggle(self, name: str, initialState=False, width=65, height=19):

        padding = 1
        full_width = width + padding * 2
        full_height = height + padding * 2
        half_width = width // 2
        radius = 3
        bg_radius = radius + 1

        toggle_widget = QWidget()
        toggle_widget.setFixedSize(full_width, full_height)
        toggle_widget.toggle_state = initialState
        toggle_widget.name = name

        # Background frame
        background = QFrame(toggle_widget)
        background.setGeometry(0, 0, full_width, full_height)
        background.setStyleSheet(f"""
            QFrame {{
                background-color: #2e2e2e;
                border: 1px solid #2e2e2e;
                border-radius: {bg_radius}px;
            }}
        """)

        # RIGHT = OFF (Red)
        bg_off = QFrame(toggle_widget)
        bg_off.setGeometry(padding + half_width, padding, half_width, height)
        bg_off.setStyleSheet(f"""
            QFrame {{
                background-color: #bf1029;
                border-top-right-radius: {radius}px;
                border-bottom-right-radius: {radius}px;
            }}
        """)

        # LEFT = ON (Green)
        bg_on = QFrame(toggle_widget)
        bg_on.setGeometry(padding, padding, half_width, height)
        bg_on.setStyleSheet(f"""
            QFrame {{
                background-color: #3f8f29;
                border-top-left-radius: {radius}px;
                border-bottom-left-radius: {radius}px;
            }}
        """)

        # Label buttons
        btn_on = QPushButton("On", toggle_widget)
        btn_on.setGeometry(padding, padding, half_width, height)
        btn_on.setEnabled(False)
        btn_on.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: black;
                border: none;
                font-weight: bold;
            }
        """)

        btn_off = QPushButton("Off", toggle_widget)
        btn_off.setGeometry(padding + half_width, padding, half_width, height)
        btn_off.setEnabled(False)
        btn_off.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: black;
                border: none;
                font-weight: bold;
            }
        """)

        # Slider
        slider = QFrame(toggle_widget)
        slider.setObjectName("slider")
        slider_x = padding + (half_width if initialState else 0)
        slider.setGeometry(slider_x, padding, half_width, height)

        slider_button = QPushButton("", toggle_widget)
        slider_button.setGeometry(slider.geometry())
        slider_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
            }
        """)

        # Hover behavior
        def on_slider_hover(event):
            slider.setProperty("hover", True)
            slider.setStyle(slider.style())

        def on_slider_leave(event):
            slider.setProperty("hover", False)
            update_slider_style(toggle_widget.toggle_state)

        def eventFilter(obj, event):
            if obj == slider_button:
                if event.type() == QEvent.Type.Enter:
                    on_slider_hover(event)
                elif event.type() == QEvent.Type.Leave:
                    on_slider_leave(event)
            return False

        slider_button.installEventFilter(toggle_widget)
        toggle_widget.eventFilter = eventFilter

        # Z-order
        background.lower()
        bg_off.raise_()
        bg_on.raise_()
        btn_off.raise_()
        btn_on.raise_()
        slider.raise_()
        slider_button.raise_()

        # Animations
        animation = QPropertyAnimation(slider, b"geometry")
        animation.setDuration(200)
        animation.setEasingCurve(QEasingCurve.Type.InOutQuad)

        animation_btn = QPropertyAnimation(slider_button, b"geometry")
        animation_btn.setDuration(200)
        animation_btn.setEasingCurve(QEasingCurve.Type.InOutQuad)

        # Style logic
        def update_slider_style(on):
            if on:
                # Right (ON)
                slider.setStyleSheet(f"""
                    QFrame#slider[hover="false"] {{
                        background-color: #555;
                        border: none;
                        border-top-right-radius: {radius}px;
                        border-bottom-right-radius: {radius}px;
                    }}
                    QFrame#slider[hover="true"] {{
                        background-color: #9fe4e5;
                        border: none;
                        border-top-right-radius: {radius}px;
                        border-bottom-right-radius: {radius}px;
                    }}
                """)
            else:
                # Left (OFF)
                slider.setStyleSheet(f"""
                    QFrame#slider[hover="false"] {{
                        background-color: #555;
                        border: none;
                        border-top-left-radius: {radius}px;
                        border-bottom-left-radius: {radius}px;
                    }}
                    QFrame#slider[hover="true"] {{
                        background-color: #9fe4e5;
                        border: none;
                        border-top-left-radius: {radius}px;
                        border-bottom-left-radius: {radius}px;
                    }}
                """)

        def set_toggle_state(on):
            toggle_widget.toggle_state = on
            update_slider_style(on)

            end_x = padding + (half_width if on else 0)
            end_rect = QRect(end_x, padding, half_width, height)

            animation.setStartValue(slider.geometry())
            animation.setEndValue(end_rect)

            animation_btn.setStartValue(slider_button.geometry())
            animation_btn.setEndValue(end_rect)

            animation.start()
            animation_btn.start()

        def toggle_state():
            slider.setProperty("hover", False)
            slider.setStyle(slider.style())
            set_toggle_state(not toggle_widget.toggle_state)

        # Connect toggle
        slider_button.clicked.connect(toggle_state)

        # Initialize appearance
        update_slider_style(initialState)
        slider.setProperty("hover", False)
        slider.setStyle(slider.style())
        set_toggle_state(initialState)

        # External access
        toggle_widget.get_state = lambda: toggle_widget.toggle_state
        toggle_widget.set_state = set_toggle_state

        return toggle_widget

class FreezableScrollArea(QScrollArea):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scroll_frozen = False

    def freeze(self, value: bool):
        self.scroll_frozen = value
        if value:
            self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        else:
            self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

    def wheelEvent(self, event):
        if not self.scroll_frozen:
            super().wheelEvent(event)
        else:
            event.ignore()

    def eventFilter(self, obj, event):
        # Optional: prevent keyboard/trackpad scroll events if needed
        if self.scroll_frozen and event.type() in (event.Type.Wheel, event.Type.KeyPress):
            return True
        return super().eventFilter(obj, event)
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OrbitSimUI()
    window.show()
    os.system("clear")
    cprint("GUI Initialized","green")
    sys.exit(app.exec())