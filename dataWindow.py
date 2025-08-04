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


# ------------------------------------------------------------------------------------
# DATA OUTPUT WINDOW FUNCTIONS
# ------------------------------------------------------------------------------------

def create_output_window(tile1_lines, tile2_lines, tile3_lines, app,
                         contact_events_toggle=True, shadow_events_toggle=True):
    def build_html_from_lines(lines):
        html = ""
        for text, style in lines:
            if "blank" in style:
                html += "<br>"
                continue
            styles = []
            if "font-size" in style:
                styles.append(f"font-size:{style['font-size']}pt;")
            if "color" in style:
                styles.append(f"color:{style['color']};")
            if style.get("bold"):
                text = f"<b>{text}</b>"
            if style.get("underline"):
                text = f"<u>{text}</u>"
            html += f'<span style="{" ".join(styles)}">{text}</span><br>'
        return html

    class OutputWindow(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Simulation Results")
            self.setStyleSheet("background-color: #2e2e2e;")
            grid = QGridLayout(self)
            grid.setContentsMargins(10, 10, 10, 10)
            grid.setSpacing(10)

            if contact_events_toggle and shadow_events_toggle:
                contact_html = build_html_from_lines(tile1_lines)
                shadow_html = build_html_from_lines(tile2_lines)
                right_html = build_html_from_lines(tile3_lines)

                left_top = self._create_scroll_tile(contact_html)
                left_bottom = self._create_scroll_tile(shadow_html)
                right_full = self._create_scroll_tile(right_html)

                grid.addWidget(left_top, 0, 0)
                grid.addWidget(left_bottom, 1, 0)
                grid.addWidget(right_full, 0, 1, 2, 1)

                grid.setColumnStretch(0, 1)
                grid.setColumnStretch(1, 1)
                grid.setRowStretch(0, 1)
                grid.setRowStretch(1, 1)

            elif contact_events_toggle:
                contact_html = build_html_from_lines(tile1_lines)
                right_html = build_html_from_lines(tile3_lines)

                left_full = self._create_scroll_tile(contact_html)
                right_full = self._create_scroll_tile(right_html)

                grid.addWidget(left_full, 0, 0, 2, 1)
                grid.addWidget(right_full, 0, 1, 2, 1)

                grid.setColumnStretch(0, 1)
                grid.setColumnStretch(1, 1)

            elif shadow_events_toggle:
                shadow_html = build_html_from_lines(tile2_lines)
                right_html = build_html_from_lines(tile3_lines)

                left_full = self._create_scroll_tile(shadow_html)
                right_full = self._create_scroll_tile(right_html)

                grid.addWidget(left_full, 0, 0, 2, 1)
                grid.addWidget(right_full, 0, 1, 2, 1)

                grid.setColumnStretch(0, 1)
                grid.setColumnStretch(1, 1)

            else:
                right_html = build_html_from_lines(tile3_lines)
                right_only = self._create_scroll_tile(right_html)

                grid.addWidget(right_only, 0, 0, 2, 2)

                grid.setColumnStretch(0, 1)

            self.setLayout(grid)
            
            if not contact_events_toggle and not shadow_events_toggle:
                self.resize(500,600)
            else:
                self.resize(1000,600)

        def _create_scroll_tile(self, html_text):
            frame = QFrame()
            frame.setStyleSheet("""
                QFrame {
                    background: white;
                    border-radius: 0px;
                }
                QLabel {
                    color: #222222;
                }
            """)
            frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

            vbox = QVBoxLayout(frame)
            vbox.setContentsMargins(10, 10, 10, 10)

            label = QLabel(html_text)
            label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
            label.setWordWrap(True)
            label.setTextFormat(Qt.RichText)
            vbox.addWidget(label)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(frame)
            scroll.setFrameShape(QFrame.NoFrame)

            scroll.setStyleSheet("""
                QScrollBar:vertical {
                    background: white;
                    width: 10px;
                    margin: 0px;
                    border-radius: 0px;
                }
                QScrollBar::handle:vertical {
                    background: #bbb;
                    border-radius: 5px;
                    min-height: 20px;
                }
                QScrollBar::handle:vertical:hover {
                    background: #888;
                }
                QScrollBar::handle:vertical:pressed {
                    background: #555;
                }
                QScrollBar::add-line:vertical,
                QScrollBar::sub-line:vertical {
                    height: 0px;  /* removes the up/down arrows */
                    subcontrol-origin: margin;
                }

                QScrollBar:horizontal {
                    background: transparent;
                    height: 10px;
                    margin: 0px;
                    border-radius: 5px;
                }
                QScrollBar::handle:horizontal {
                    background: #bbb;
                    border-radius: 5px;
                    min-width: 20px;
                }
                QScrollBar::handle:horizontal:hover {
                    background: #9fe4e5;
                }
                QScrollBar::handle:horizontal:pressed {
                    background: #555;
                }
                QScrollBar::add-line:horizontal,
                QScrollBar::sub-line:horizontal {
                    width: 0px;  /* removes the left/right arrows */
                    subcontrol-origin: margin;
                }
            """)
            return scroll

    window = OutputWindow()
    window.show()
    print()
    cprint("Output Window Created", "green")
    return window


# ----------------------------------------------------------------------
# FORMAT CONACT AND SHADOW EVENTS FUNCTIONS
# ----------------------------------------------------------------------

def build_contact_event_dict(events_string):

    if not events_string:

        return {}

    result = {}
    
    for ds, text in events_string.items():
        if "Spec :" not in text:
            continue
        for part in text.split("Spec :")[1:]:
            lines = [l.strip() for l in part.splitlines() if l.strip()]
            spec_line = lines[0]
            type_line = lines[1]
            epoch_line = lines[2]
    
            key = (spec_line
                   .replace("Rcvr horizon mask from", "")
                   .replace("to spacecraft1", "")
                   .strip())
            type_val = type_line.split(":", 1)[1].strip()
            epoch_val = epoch_line.split(":", 1)[1].strip()
            result.setdefault(key, []).append(f"{type_val} {epoch_val}")
    
    return result

def format_contact_dict(data):

    if not data:

        return "          "
    
    epoch = datetime.strptime("01-JAN-2000 00:00:00.0000", "%d-%b-%Y %H:%M:%S.%f")

    time_dict = {}

    for key, events in data.items():
        for event in events:
            try:

                dt_str = event.split(' ', 1)[1].rsplit(' ', 1)[0]
                dt = datetime.strptime(dt_str, "%d-%b-%Y %H:%M:%S.%f")
                seconds = (dt - epoch).total_seconds()

                time_dict[seconds] = f"{key}: {event}"
            except Exception as e:
                print(f"Error parsing {event} from {key}: {e}")

    formatted_output = []

    for seconds in sorted(time_dict):
        line = time_dict[seconds]
        station, event = line.split(': ', 1)
        if event.startswith("Set "):
            event = "Set  " + event[4:]
        formatted_output.append(f"{station}: {event}")

    return formatted_output

def build_shadow_event_dict(events_string):

    if not events_string:

        return {}

    result = {}
    
    for ds, text in events_string.items():
        if "Spec :" not in text:
            continue
        for part in text.split("Spec :")[1:]:
            lines = [l.strip() for l in part.splitlines() if l.strip()]
            spec_line = lines[0]
            type_line = lines[1]
            begin_line = lines[2]
            end_line = lines[3]

            if len(spec_line) <= 43:
                
                key = (spec_line
                       .replace("spacecraft1 passing through", "")
                       .replace("'s umbra", "")
                       .strip())
                
            elif len(spec_line) > 43:

                key = (spec_line
                       .replace("spacecraft1 passing through", "")
                       .replace("'s penumbra", "")
                       .strip())
                
            type_val = type_line.split(":", 1)[1].strip()
            begin_val = begin_line.split(":", 1)[1].strip()
            end_val = end_line.split(":", 1)[1].strip()
            result.setdefault(key, []).append(f"{type_val} {begin_val} {end_val}")
    
    return result

def format_shadow_dict(data):

    if not data:

        return "          "
    
    epoch = datetime.strptime("01-JAN-2000 00:00:00.0000", "%d-%b-%Y %H:%M:%S.%f")
    time_dict = {}

    for body, events in data.items():
        for line in events:
            parts = line.split()

            # Validate format
            if len(parts) < 8 or parts[0].lower() != "in":
                continue  # skip malformed lines

            region = parts[1].capitalize()

            # Correctly extract entry and exit timestamps
            entry_str = f"{parts[2]} {parts[3]}"
            exit_str  = f"{parts[5]} {parts[6]}"  # skip "ET" at index 4

            # Convert to datetime
            entry_dt = datetime.strptime(entry_str, "%d-%b-%Y %H:%M:%S.%f")
            exit_dt  = datetime.strptime(exit_str,  "%d-%b-%Y %H:%M:%S.%f")

            # Convert to seconds (with slight offset)
            entry_sec = (entry_dt - epoch).total_seconds() + 0.00001
            exit_sec  = (exit_dt - epoch).total_seconds() - 0.00001

            # Label each event
            entry_label = f"Entering {body}'s {region}:"
            exit_label  = f"Exiting {body}'s {region}:"

            # Store in dict for sorting
            time_dict[entry_sec] = (entry_label, entry_str)
            time_dict[exit_sec]  = (exit_label,  exit_str)

    # Determine padding width for alignment
    max_label_len = max(len(label) for label, _ in time_dict.values())

    # Build formatted output
    formatted_output = []
    for seconds in sorted(time_dict):
        label, timestamp = time_dict[seconds]
        padded_label = label.ljust(max_label_len)
        formatted_output.append(f"{padded_label} {timestamp} {parts[4]}")

    return formatted_output


# ------------------------------------------------------------------------------------
# PUPULATE AND CREATE DATA OUTPUT WINDOW 
# ------------------------------------------------------------------------------------

contact_events = format_contact_dict(build_contact_event_dict(contact_events_dict))
shadow_events = format_shadow_dict(build_shadow_event_dict(shadow_events_dict))

total_coverage_percentage = str(total_percent)[:5]
sunlit_coverage_percentage = str(lit_percent)[:5]

primary_shadow_ratio = np.mean(primary_shadow_array)
primary_shadow_percent_str = str(primary_shadow_ratio*100)[:6]

primary_shadow_seconds = len(shadow_array)*time_step_seconds*primary_shadow_ratio
primary_shadow_sec_str = str(primary_shadow_seconds*100)[:6]

if primary == "Earth":

    moon_shadow_ratio = np.mean(moon_shadow_array)
    moon_shadow_percent_str = str(moon_shadow_ratio*100)[:6]

    moon_shadow_seconds = len(shadow_array)*time_step_seconds*moon_shadow_ratio
    moon_shadow_sec_str = str(moon_shadow_seconds*100)[:6]

elif primary == "Mars":

    phobos_shadow_ratio = np.mean(phobos_shadow_array)
    phobos_shadow_percent_str = str(phobos_shadow_ratio*100)[:6]

    phobos_shadow_seconds = len(shadow_array)*time_step_seconds*phobos_shadow_ratio
    phobos_shadow_sec_str = str(phobos_shadow_seconds*100)[:6]

    deimos_shadow_ratio = np.mean(deimos_shadow_array)
    deimos_shadow_percent_str = str(deimos_shadow_ratio*100)[:6]

    deimos_shadow_seconds = len(shadow_array)*time_step_seconds*deimos_shadow_ratio
    deimos_shadow_sec_str = str(deimos_shadow_seconds*100)[:6]

sunlit_ratio = 1-np.mean(shadow_array)
sunlit_percent_str = str(sunlit_ratio*100)[:6]

sunlit_seconds = len(shadow_array)*time_step_seconds*sunlit_ratio
sunlit_sec_str = str(sunlit_seconds*100)[:6]


# Populate Output Window Tiles

contactEventsTile1 = [

                        ("Ground Station Contact Events:", {"font-size": 14, "bold": True,"underline": True}),
                        ("",{"blank": True}),
                        ("----------------------------------------------------------------------------------------------------", {"font-size": 9}),

                    ] 

contactEventsTile2 = []

if not all(s.strip() == "" and s != "" for s in contact_events):

    for i in range(0,len(contact_events)):

        contact_event = contact_events[i]
        event_parts = contact_event.split(":",1)

        rise_set_part = event_parts[1]
        rise_set_parts = rise_set_part.split(" ",2)

        event_label = (f"{event_parts[0]}: {rise_set_parts[1]}").ljust(30)

        if rise_set_parts[2].startswith(" "):

            event = event_label + rise_set_parts[2][1:]

        else:

            event = event_label + rise_set_parts[2]

        contactEventsTile2 += [
        
        ("",{"blank": True}),] + [
        (f'<div style="white-space: pre; font-family: monospace">{event}</div>', {"font-size": 9}),] + [
        ("",{"blank": True}),] + [
        ("----------------------------------------------------------------------------------------------------", {"font-size": 9}),]

shadowEventsTile1 =  [

                        ("Primary and Moons Shadow Events:", {"font-size": 14, "bold": True,"underline": True}),
                        ("",{"blank": True}),
                        ("----------------------------------------------------------------------------------------------------", {"font-size": 9}),

                    ]

shadowEventsTile2 = []

if not all(s.strip() == "" and s != "" for s in shadow_events):

    for i in range(0,len(shadow_events)):

        shadow_event = shadow_events[i]
        event_parts = shadow_event.split()

        event_label = (f"{event_parts[0]} {event_parts[1]} {event_parts[2]}").ljust(30)
        epoch_label = f"{event_parts[3]} {event_parts[4]} {event_parts[5]}"

        event = event_label + epoch_label

        shadowEventsTile2 += [

        ("",{"blank": True}),] + [
        (f'<div style="white-space: pre; font-family: monospace">{event}</div>', {"font-size": 9}),] + [
        ("",{"blank": True}),] + [
        ("----------------------------------------------------------------------------------------------------", {"font-size": 9}),]

label_width = 30

total_coverage_percent_label = (f"Total Coverage:").ljust(label_width) + (f"{total_coverage_percentage}").ljust(12) + "[%]"
total_coverage_label = (f"Sunlit Coverage:").ljust(label_width) + (f"{sunlit_coverage_percentage}").ljust(12) + "[%]"

primary_shadow_percent_label = (f"Time in {primary}\'s Shadow:").ljust(label_width) + (f"{primary_shadow_percent_str}").ljust(12) + "[%]"
primary_shadow_label = (f"Time in {primary}\'s Shadow:").ljust(label_width) + (f"{primary_shadow_sec_str}").ljust(12) + "[seconds]"

otherDataTile1 =     [

                        ("Ground Coverage:", {"font-size": 14, "bold": True,"underline": True}),
                        ("",{"blank": True}),
                        ("----------------------------------------------------------------------------------------------------", {"font-size": 9}),
                        (f'<div style="white-space: pre; font-family: monospace">{total_coverage_percent_label}</div>', {"font-size": 9}),
                        (f'<div style="white-space: pre; font-family: monospace">{total_coverage_label}</div>', {"font-size": 9}),
                        ("----------------------------------------------------------------------------------------------------", {"font-size": 9}),
                        ("",{"blank": True}),
                        ("",{"blank": True}),
                        ("Shadow Time:", {"font-size": 14, "bold": True,"underline": True}),
                        ("",{"blank": True}),
                        ("----------------------------------------------------------------------------------------------------", {"font-size": 9}),
                        (f'<div style="white-space: pre; font-family: monospace">{primary_shadow_percent_label}</div>', {"font-size": 9}),
                        (f'<div style="white-space: pre; font-family: monospace">{primary_shadow_label}</div>', {"font-size": 9}),
                        ("----------------------------------------------------------------------------------------------------", {"font-size": 9}),

                ]

if primary == "Earth":

    moon_shadow_percent_label = (f"Time in Moon's Shadow:").ljust(label_width) + (f"{moon_shadow_percent_str}").ljust(12) + "[%]"
    moon_shadow_label = (f"Time in Moon's Shadow:").ljust(label_width) + (f"{moon_shadow_sec_str}").ljust(12) + "[seconds]"

    otherDataTile2 =    [

                            (f'<div style="white-space: pre; font-family: monospace">{moon_shadow_percent_label}</div>', {"font-size": 9}),
                            (f'<div style="white-space: pre; font-family: monospace">{moon_shadow_label}</div>', {"font-size": 9}),
                            ("----------------------------------------------------------------------------------------------------", {"font-size": 9}),

                        ]

elif primary == "Mars":

    phobos_shadow_percent_label = (f"Time in Phobos' Shadow:").ljust(label_width) + (f"{phobos_shadow_percent_str}").ljust(12) + "[%]"
    phobos_shadow_label = (f"Time in Phobos' Shadow:").ljust(label_width) + (f"{phobos_shadow_sec_str}").ljust(12) + "[seconds]"

    deimos_shadow_percent_label = (f"Time in Deimos' Shadow:").ljust(label_width) + (f"{deimos_shadow_percent_str}").ljust(12) + "[%]"
    deimos_shadow_label = (f"Time in Deimos' Shadow:").ljust(label_width) + (f"{deimos_shadow_sec_str}").ljust(12) + "[seconds]"

    otherDataTile2 =    [

                            (f'<div style="white-space: pre; font-family: monospace">{phobos_shadow_percent_label}</div>', {"font-size": 9}),
                            (f'<div style="white-space: pre; font-family: monospace">{phobos_shadow_label}</div>', {"font-size": 9}),
                            ("----------------------------------------------------------------------------------------------------", {"font-size": 9}),
                            (f'<div style="white-space: pre; font-family: monospace">{deimos_shadow_percent_label}</div>', {"font-size": 9}),
                            (f'<div style="white-space: pre; font-family: monospace">{deimos_shadow_label}</div>', {"font-size": 9}),
                            ("----------------------------------------------------------------------------------------------------", {"font-size": 9}),

                        ]

sunlit_percent_label = (f"Time in Sunlight:").ljust(label_width) + (f"{sunlit_percent_str}").ljust(12) + "[%]"
sunit_label = (f"Time in Sunlight:").ljust(label_width) + (f"{sunlit_sec_str}").ljust(12) +  "[seconds]"

otherDataTile3 =    [

                        (f'<div style="white-space: pre; font-family: monospace">{sunlit_percent_label}</div>', {"font-size": 9}),
                        (f'<div style="white-space: pre; font-family: monospace">{sunit_label}</div>', {"font-size": 9}),
                        ("----------------------------------------------------------------------------------------------------", {"font-size": 9}),
                        ("",{"blank": True}),
                        ("",{"blank": True}), ]

if len(contact_events)!=10:
                        
    otherDataTile3 += [
                            ("Contact Time:", {"font-size": 14, "bold": True,"underline": True}),
                            ("",{"blank": True}),
                            ("Average Contact Time Per Event:", {"font-size": 10, "bold": True}),
    
                        ] 
    
    for key,value in contact_durations_avg.items():
    
        contact_event_label = (f"{key}:").ljust(label_width) + (f"{round(value,4)}").ljust(12) + "[seconds]"
    
        otherDataTile3 += [
        
                            ("----------------------------------------------------------------------------------------------------", {"font-size": 9}),
                            (f'<div style="white-space: pre; font-family: monospace">{contact_event_label}</div>', {"font-size": 9}),
    
                          ]
        
    otherDataTile3 += [
         
                        ("----------------------------------------------------------------------------------------------------", {"font-size": 9}),
                        ("",{"blank": True}),
                        ("",{"blank": True}),
                        ("Average Contact Time Per Orbital Period:", {"font-size": 10, "bold": True}),
    
                       ]
    
    for key,value in contact_durations_avg_per.items():
    
        contact_event_label = (f"{key}:").ljust(label_width) + (f"{round(value,4)}").ljust(12) + "[seconds]"
    
        otherDataTile3 += [
        
                            ("----------------------------------------------------------------------------------------------------", {"font-size": 9}),
                            (f'<div style="white-space: pre; font-family: monospace">{contact_event_label}</div>', {"font-size": 9}),
    
                          ]
        
    otherDataTile3 += [
         
                        ("----------------------------------------------------------------------------------------------------", {"font-size": 9}),
                        ("",{"blank": True}),
                        ("",{"blank": True}),
                        ("Total Contact Time:", {"font-size": 10, "bold": True}),
    
                       ]

    for key,value in contact_durations.items():
    
        contact_event_label = (f"{key}:").ljust(label_width) + (f"{round(value,4)}").ljust(12) + "[seconds]"
    
        otherDataTile3 += [
            
                            ("----------------------------------------------------------------------------------------------------", {"font-size": 9}),
                            (f'<div style="white-space: pre; font-family: monospace">{contact_event_label}</div>', {"font-size": 9}),
    
                          ]
    
    otherDataTile3 += [
         
                        ("----------------------------------------------------------------------------------------------------", {"font-size": 9}),
    
                       ]

contactEventsTile = contactEventsTile1 + contactEventsTile2
shadowEventsTile = shadowEventsTile1 + shadowEventsTile2
otherDataTile = otherDataTile1 + otherDataTile2 + otherDataTile3

contact_events_toggle = len(contact_events)!=10 and DataOutput_ContactEvents
shadow_events_toggle = len(shadow_events)!=10 and DataOutput_ShadowEvents

app = QApplication([])

print()
cprint("Creating Output Window","blue")

outputWindow = create_output_window(
                                    contactEventsTile, shadowEventsTile, otherDataTile, app,
                                    contact_events_toggle, shadow_events_toggle
                                   )
    
app.exec_()