import json
import sys
import re
import math
import subprocess
from datetime import datetime, timedelta
import time
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import root_scalar
import warnings
from numba import njit
import Monte as M
import mpy.io.data as defaultData
import mpy.traj.force.grav.basic as basicGrav
from mpy.units import *
from mpy.io.stuf import *

# ----------------------------------------------------------------------
# IMPORT DATA FROM USER INTERFACE
# ----------------------------------------------------------------------

def sanitize_key(key):
    return re.sub(r'[^0-9a-zA-Z_]', '', key)

if len(sys.argv) < 2:
    print("Usage: python monteSimulation.py input_data.json")
    sys.exit(1)

input_path = sys.argv[1]

with open(input_path, "r") as f:
    try:
        data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
        sys.exit(1)

for section, values in data.items():
    if isinstance(values, dict):
        for key, val in values.items():
            safe_key = sanitize_key(key)
            globals()[safe_key] = val
    else:
        safe_key = sanitize_key(section)
        globals()[safe_key] = values
        #print(f"{safe_key} = {values}")

warnings.filterwarnings("ignore", message = "A NumPy version >=")

# ----------------------------------------------------------------------
# SETUP BOA AND INITIAL STATE
# ----------------------------------------------------------------------

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

print()
cprint("MONTE Simulation Started","blue")

start = time.time()

boa = M.BoaLoad()

defaultData.loadInto(boa, [
    "frame",
    "body",
    "ephem/satellite/mars097",
    "ephem/planet/de405",
    "station"
    ])

primary = PrimaryandPerturbations_PrimaryBody
frame = InitialOrbitalElements_ReferenceFrame
orbitalElements = InitialOrbitalElements_Type

if primary == "Mercury":

    primary_j2_norm = M.SphHarmonicsBoa.read(boa,"MESS30A").j()[2]
    primary_j3_norm = M.SphHarmonicsBoa.read(boa,"MESS30A").j()[3]

if primary == "Venus":

    primary_j2_norm = M.SphHarmonicsBoa.read(boa,"MGNP180U").j()[2]
    primary_j3_norm = M.SphHarmonicsBoa.read(boa,"MGNP180U").j()[3]

if primary == "Earth":
    
    primary_j2_norm = M.SphHarmonicsBoa.read(boa,"EGM96").j()[2]
    primary_j3_norm = M.SphHarmonicsBoa.read(boa,"EGM96").j()[3]

if primary == "Mars":
    
    primary_j2_norm = M.SphHarmonicsBoa.read(boa,"Mars").j()[2]
    primary_j3_norm = M.SphHarmonicsBoa.read(boa,"Mars").j()[3]

primary_j2 = primary_j2_norm*math.sqrt(5)
primary_j3 = primary_j3_norm*math.sqrt(7)

primaryBodyDataBoa = M.BodyDataBoa.read(boa,primary)
primaryBodyData = M.BodyData(boa, primary,primaryBodyDataBoa.frame(),primary)

primary_equitorial_radius = M.UnitDbl.value(M.BodyData.radius(primaryBodyData))
primary_polar_radius = primary_equitorial_radius*math.sqrt(1-19/3*primary_j2)

scName = "spacecraft1"

def time_step_sec(time_step,time_step_unit):

    if time_step_unit == "Seconds":

        return time_step

    if time_step_unit == "Minutes":

        return time_step*60

    if time_step_unit == "Hours":

        return time_step*3600

    if time_step_unit == "Days":

        return time_step*24*3600
    
time_step_seconds =  time_step_sec(float(EpochDuration_SimulationTimeStep),EpochDuration_SimulationTimeStep_Units)

# ----------------------------------------------------------------------
# SPACECRAFT INITIAL STATE FUNCTIONS
# ----------------------------------------------------------------------

def create_initial_State(globalVars,boa,primary,frame,orbitalElements,t,primary_j2,primary_j3):

    sunBodyDataBoa = M.BodyDataBoa.read(boa,"Sun")
    sunBodyData = M.BodyData(boa, "Sun",sunBodyDataBoa.frame(),"Sun")
    muSun = M.UnitDbl.value(M.BodyData.gm(sunBodyData))

    primaryTraj = M.TrajQuery(boa,primary,"Sun",frame)
    primaryState = primaryTraj.state(t, 2)
    primarySunOrbitalRadius = primaryState.posMag()
    primarySunOrbitalSpeed = primaryState.velMag()
    primarySunSMA = (2/M.UnitDbl.value(primarySunOrbitalRadius)-M.UnitDbl.value(primarySunOrbitalSpeed)**2/M.UnitDbl.value(M.BodyData.gm(sunBodyData)))**-1

    primaryBodyDataBoa = M.BodyDataBoa.read(boa,primary)
    primaryBodyData = M.BodyData(boa, primary,primaryBodyDataBoa.frame(),primary)
    muPrimary = M.UnitDbl.value(M.BodyData.gm(primaryBodyData))
    primaryEquitorialRadius = M.UnitDbl.value(M.BodyData.radius(primaryBodyData))

    sunTraj = M.TrajQuery(boa,"Sun",primary,f"IAU {primary} Fixed")
    sunState = sunTraj.state(t, 2)
    sunLongitude = M.UnitDbl.value(M.Geodetic.longitude(sunState))

    if orbitalElements == "Keplarian":

        element_names = ["Semi-Major Axis [km]", "Eccentricity", "Inclination [deg]",
                         "RAAN [deg]", "ARGP [deg]", "True Anomaly [deg]"]

        initialState = M.State(
            boa, scName, primary,
            M.Conic.semiMajorAxis(M.UnitDbl(float(globalVars['InitialOrbitalElements_SemiMajorAxiskm']), 'km')),
            M.Conic.eccentricity(float(globalVars['InitialOrbitalElements_Eccentricity'])),
            M.Conic.inclination(M.UnitDbl(float(globalVars['InitialOrbitalElements_Inclinationdeg']), 'deg')),
            M.Conic.argumentOfLatitude(M.UnitDbl(float(globalVars['InitialOrbitalElements_ArgumentofPeriapsisdeg']), 'deg')),
            M.Conic.longitudeOfNode(M.UnitDbl(float(globalVars['InitialOrbitalElements_RAANdeg']), 'deg')),
            M.Conic.trueAnomaly(M.UnitDbl(float(globalVars['InitialOrbitalElements_TrueAnomalydeg']), 'deg'))
        )

    if orbitalElements == "Cartesian":

        element_names = ["X Position [km]", "Y Position [km]", "Z Position [km]",
                         "X Velocity [km/s]", "Y Velocity [km/s]", "Z Velocity [km/s]"]

        initialState = M.State(
            boa, scName, primary,
            M.Cartesian.x(M.UnitDbl(float(globalVars['InitialOrbitalElements_XPositionkm']), 'km')),
            M.Cartesian.y(M.UnitDbl(float(globalVars['InitialOrbitalElements_YPositionkm']), 'km')),
            M.Cartesian.z(M.UnitDbl(float(globalVars['InitialOrbitalElements_ZPositionkm']), 'km')),
            M.Cartesian.dx(M.UnitDbl(float(globalVars['InitialOrbitalElements_XVelocitykms']), 'km/sec')),
            M.Cartesian.dy(M.UnitDbl(float(globalVars['InitialOrbitalElements_YVelocitykms']), 'km/sec')),
            M.Cartesian.dz(M.UnitDbl(float(globalVars['InitialOrbitalElements_ZVelocitykms']), 'km/sec'))
        )

    if orbitalElements == "Spherical":

        element_names = ["Radius [km]", "Latitude [deg]", "Longitude [deg]", "Radial Velocity [km/s]", 
                         "Latitudinal Velocity [deg/s]","Longitudinal Velocity [deg/s]"]

        initialState = M.State(
            boa, scName, primary,
            M.Spherical.radius(M.UnitDbl(float(globalVars['InitialOrbitalElements_XPositionkm']), 'km')),
            M.Spherical.dradius(M.UnitDbl(float(globalVars['InitialOrbitalElements_YPositionkm']), 'km/sec')),
            M.Spherical.latitude(M.UnitDbl(float(globalVars['InitialOrbitalElements_ZPositionkm']), 'deg')),
            M.Spherical.dlatitude(M.UnitDbl(float(globalVars['InitialOrbitalElements_XVelocitykms']), 'deg/sec')),
            M.Spherical.longitude(M.UnitDbl(float(globalVars['InitialOrbitalElements_YVelocitykms']), 'deg')),
            M.Spherical.dlongitude(M.UnitDbl(float(globalVars['InitialOrbitalElements_ZVelocitykms']), 'deg/sec'))
        )

    if orbitalElements == "Custom":

        oe_inputs = {InitialOrbitalElements_CustomElement1, InitialOrbitalElements_CustomElement2, InitialOrbitalElements_CustomElement3,
                     InitialOrbitalElements_CustomElement4, InitialOrbitalElements_CustomElement5, InitialOrbitalElements_CustomElement6,
                     InitialOrbitalElements_CustomElement7, InitialOrbitalElements_CustomElement8}
        
        oe_input_types = []
        oe_input_units = []
        oe_input_vals = []

        for oe_input in oe_inputs:

            if len(oe_input)!=0:

                oe_input_parts = oe_input.split()

                if len(oe_input_parts)==2:

                    oe_input_type = oe_input_parts[0][:-1]
                    oe_input_unit = 1
                    oe_input_val = oe_input_parts[1]
                    oe_input_types.append(oe_input_type)
                    oe_input_units.append(oe_input_unit)
                    oe_input_vals.append(oe_input_val)

                if len(oe_input_parts)==3:

                    oe_input_type = oe_input_parts[0]
                    oe_input_unit = oe_input_parts[1][1:-2]
                    oe_input_val = oe_input_parts[2]
                    oe_input_types.append(oe_input_type)
                    oe_input_units.append(oe_input_unit)
                    oe_input_vals.append(oe_input_val)

        orbitalElements = InitialOrbitalElements_CustomElementsType

        element_names = oe_input_types

        dynamic_arguments = []

        for i in range(len(oe_input_types)):

            method_to_call = getattr(getattr(M, orbitalElements), oe_input_types[i])
            argument_value = M.UnitDbl(float(oe_input_vals[i]), oe_input_units[i])

            dynamic_arguments.append(method_to_call(argument_value))

        initialState = M.State(
            boa, scName, primary,
            *dynamic_arguments
        )

        element_names = ["Semi-Major Axis [km]", "Eccentricity", "Inclination [deg]",
                         "RAAN [deg]", "ARGP [deg]", "True Anomaly [deg]"]

    if orbitalElements == "Sun Synchronous Orbit":

        element_names = ["Semi-Major Axis [km]", "Eccentricity", "Inclination [deg]",
                         "RAAN [deg]", "ARGP [deg]", "True Anomaly [deg]"]
        
        eccentricity = 0
        altitude = float(globalVars['InitialOrbitalElements_Altitudekm'])
        semiMajorAxis =  M.UnitDbl.value(primaryBodyData.radius()) + altitude
        inclination = math.degrees(math.acos(-2/3*(1-eccentricity**2)**2*semiMajorAxis**(7/2)/(math.sqrt(muPrimary*primarySunSMA**3/muSun)*primary_j2*primaryEquitorialRadius**2)))
        RAAN_deg = sunLongitude - 15*(float(globalVars['InitialOrbitalElements_MLTANhours'])-6)

        initialState = M.State(
            boa, scName, primary,
            M.Conic.semiMajorAxis(M.UnitDbl(semiMajorAxis, 'km')),
            M.Conic.eccentricity(0),
            M.Conic.inclination(M.UnitDbl(inclination, 'deg')),
            M.Conic.argumentOfLatitude(M.UnitDbl(0, 'deg')),
            M.Conic.longitudeOfNode(M.UnitDbl(RAAN_deg, 'deg')),
            M.Conic.trueAnomaly(M.UnitDbl(float(globalVars['InitialOrbitalElements_TrueAnomalydeg']), 'deg'))
        )

    if orbitalElements == "Frozen Orbit":

        element_names = ["Semi-Major Axis [km]", "Eccentricity", "Inclination [deg]",
                         "RAAN [deg]", "ARGP [deg]", "True Anomaly [deg]"]
        
        inclination = float(globalVars['InitialOrbitalElements_Inclinationdeg'])
        semiMajorAxis = float(globalVars['InitialOrbitalElements_SemiMajorAxiskm'])
        eccentricity = -(primary_j3*math.sin(math.radians(inclination)))/(primary_j2*2*semiMajorAxis)

        initialState = M.State(
            boa, scName, primary,
            M.Conic.semiMajorAxis(M.UnitDbl(float(globalVars['InitialOrbitalElements_SemiMajorAxiskm']), 'km')),
            M.Conic.eccentricity(eccentricity),
            M.Conic.inclination(M.UnitDbl(float(globalVars['InitialOrbitalElements_Inclinationdeg']), 'deg')),
            M.Conic.argumentOfLatitude(M.UnitDbl(float(globalVars['InitialOrbitalElements_ArgumentofPeriapsisdeg']), 'deg')),
            M.Conic.longitudeOfNode(M.UnitDbl(float(globalVars['InitialOrbitalElements_RAANdeg']), 'deg')),
            M.Conic.trueAnomaly(M.UnitDbl(float(globalVars['InitialOrbitalElements_TrueAnomalydeg']), 'deg'))
        )

    if orbitalElements == "Repeat Ground Track":

        element_names = ["Semi-Major Axis [km]", "Eccentricity", "Inclination [deg]",
                         "RAAN [deg]", "ARGP [deg]", "True Anomaly [deg]"]
        
        def repeat_groundtrack( boa, primary, inclination_deg, num_periods, num_days):

            inclination_rad = math.radians(inclination_deg)
            Tprimary = 2*math.pi*math.sqrt(primarySunSMA**3/muSun)

            sunTraj = M.TrajQuery(boa,"Sun",primary,f"IAU {primary} Fixed")
            tArray = M.Epoch.range(M.Epoch("01-JAN-2000 00:00:00 ET"),M.Epoch("03-JAN-2000 00:00:00 ET"),M.UnitDbl(1,"seconds"))
            sunLongitudes = []

            for t in tArray:
            
                sunState = sunTraj.state(t, 2)
                sunLongitude = M.UnitDbl.value(M.Geodetic.longitude(sunState))
                sunLongitudes.append(sunLongitude)

            sunLongitudesArray = np.array(sunLongitudes)
            minima_indices, _ = find_peaks(-sunLongitudesArray)
            primarySolarDay = (minima_indices[1] - minima_indices[0])
            primarySideralDay = Tprimary/(Tprimary/primarySolarDay + 1)

            T_sat_desired = (num_days * primarySideralDay) / num_periods

            def sma_eqn(sma):
                mean_motion = 2 * math.pi / T_sat_desired
                omega_dot = (-3 / 2) * math.sqrt(muPrimary) * primary_j2 * primary_equitorial_radius**2 * math.cos(inclination_rad) \
                    / ((1 - 0**2)**2 * sma**(3.5))
                return mean_motion - (math.sqrt(muPrimary / sma**3) + omega_dot)

            sol = root_scalar(sma_eqn, bracket=[primary_equitorial_radius + 1, 1e9], method='brentq')
            if not sol.converged:
                raise RuntimeError("Failed to solve for semi-major axis")

            sma_solution = sol.root

            return sma_solution 
        
        semiMajorAxis = repeat_groundtrack( boa, primary, 
                                           float(globalVars['InitialOrbitalElements_Inclinationdeg']), 
                                           float(globalVars['InitialOrbitalElements_RepeatCycles']), 
                                           float(globalVars['InitialOrbitalElements_RepeatTimedays']))
        
        initialState = M.State(
            boa, scName, primary,
            M.Conic.semiMajorAxis(M.UnitDbl(semiMajorAxis, 'km')),
            M.Conic.eccentricity(float(globalVars['InitialOrbitalElements_Eccentricity'])),
            M.Conic.inclination(M.UnitDbl(float(globalVars['InitialOrbitalElements_Inclinationdeg']), 'deg')),
            M.Conic.argumentOfLatitude(M.UnitDbl(float(globalVars['InitialOrbitalElements_ArgumentofPeriapsisdeg']), 'deg')),
            M.Conic.longitudeOfNode(M.UnitDbl(float(globalVars['InitialOrbitalElements_RAANdeg']), 'deg')),
            M.Conic.trueAnomaly(M.UnitDbl(float(globalVars['InitialOrbitalElements_TrueAnomalydeg']), 'deg'))
        )

    if orbitalElements == "Geosynchronous":

        element_names = ["Semi-Major Axis [km]", "Eccentricity", "Inclination [deg]",
                         "RAAN [deg]", "ARGP [deg]", "True Anomaly [deg]"]
        
        print(orbitalElements)

    if orbitalElements == "Molniya":

        element_names = ["Semi-Major Axis [km]", "Eccentricity", "Inclination [deg]",
                         "RAAN [deg]", "ARGP [deg]", "True Anomaly [deg]"]
        
        print(orbitalElements)

    if orbitalElements == "Tundra":

        element_names = ["Semi-Major Axis [km]", "Eccentricity", "Inclination [deg]",
                         "RAAN [deg]", "ARGP [deg]", "True Anomaly [deg]"]
        
        print(orbitalElements)

    return initialState, element_names

def shift_datetime_string(datetime_str: str, num_seconds: int, mode: str) -> str:

    try:
        base_str, time_system = datetime_str.rsplit(' ', 1)
    except ValueError:
        raise ValueError("Input string must end with a space and a time system (e.g. 'ET')")

    try:
        dt = datetime.strptime(base_str, "%d-%b-%Y %H:%M:%S")
    except ValueError:
        raise ValueError("Datetime format must be 'DD-MMM-YYYY HH:MM:SS'")

    delta = timedelta(seconds=num_seconds)
    if mode == "initial":
        new_dt = dt + delta
    elif mode == "final":
        new_dt = dt - delta
    else:
        raise ValueError("Mode must be either 'initial' or 'final'")

    new_dt_str = new_dt.strftime("%d-%b-%Y %H:%M:%S")
    return f"{new_dt_str} {time_system}"


# -------------------------------------------------------------------------------------
# CREATE SPACECRAFT INITIAL STATE
# -------------------------------------------------------------------------------------

if 'EpochDuration_InitialEpoch' in globals() and ('EpochDuration_NumberofOrbitalPeriods' in globals() or 'EpochDuration_Timespan' in globals()):

    t0 = M.Epoch(EpochDuration_InitialEpoch)

    initialState, element_names = create_initial_State(globals(),boa,primary,frame,orbitalElements,t0,primary_j2,primary_j3)

    if 'EpochDuration_NumberofOrbitalPeriods' in globals():

        pos = M.UnitDbl.value(initialState.posMag())
        vel = M.UnitDbl.value(initialState.velMag())
        primaryBodyDataBoa = M.BodyDataBoa.read(boa,primary)
        primaryBodyData = M.BodyData(boa, primary,primaryBodyDataBoa.frame(),primary)
        muPrimary = M.UnitDbl.value(M.BodyData.gm(primaryBodyData))
        eps = vel**2/2-muPrimary/pos
        sma = -muPrimary/(2*eps)
        T = 2*math.pi*math.sqrt(sma**3/muPrimary)
        tf = M.Epoch(shift_datetime_string(EpochDuration_InitialEpoch, T*float(EpochDuration_NumberofOrbitalPeriods), "initial"))

    if 'EpochDuration_Timespan' in globals():
        
        if EpochDuration_Timespan_Units == "Seconds":

            timespanUnit = 1

        if EpochDuration_Timespan_Units == "Minutes":

            timespanUnit = 60

        if EpochDuration_Timespan_Units == "Hours":

            timespanUnit = 3600

        if EpochDuration_Timespan_Units == "Days":

            timespanUnit = 24*3600

        tf = M.Epoch(shift_datetime_string(EpochDuration_InitialEpoch, timespanUnit*float(EpochDuration_Timespan), "initial"))

if 'EpochDuration_InitialEpoch' in globals() and 'EpochDuration_FinalEpoch' in globals():

    t0 = M.Epoch(EpochDuration_InitialEpoch)
    tf = M.Epoch(EpochDuration_FinalEpoch)

    initialState, element_names = create_initial_State(globals(),boa,primary,frame,orbitalElements,t0,primary_j2,primary_j3)

    pos = M.UnitDbl.value(initialState.posMag())
    vel = M.UnitDbl.value(initialState.velMag())
    primaryBodyDataBoa = M.BodyDataBoa.read(boa,primary)
    primaryBodyData = M.BodyData(boa, primary,primaryBodyDataBoa.frame(),primary)
    muPrimary = M.UnitDbl.value(M.BodyData.gm(primaryBodyData))
    eps = vel**2/2-muPrimary/pos
    sma = -muPrimary/(2*eps)
    T = 2*math.pi*math.sqrt(sma**3/muPrimary)

if ('EpochDuration_NumberofOrbitalPeriods' in globals() or 'EpochDuration_Timespan' in globals()) and 'EpochDuration_FinalEpoch' in globals():

    tf = M.Epoch(EpochDuration_FinalEpoch)
    finalState, element_names = create_initial_State(globals(),boa,primary,frame,orbitalElements,tf,primary_j2,primary_j3)

    if 'EpochDuration_NumberofOrbitalPeriods' in globals():

        pos = M.UnitDbl.value(finalState.posMag())
        vel = M.UnitDbl.value(finalState.velMag())
        primaryBodyDataBoa = M.BodyDataBoa.read(boa,primary)
        primaryBodyData = M.BodyData(boa, primary,primaryBodyDataBoa.frame(),primary)
        muPrimary = M.UnitDbl.value(M.BodyData.gm(primaryBodyData))
        eps = vel**2/2-muPrimary/pos
        sma = -muPrimary/(2*eps)
        T = 2*math.pi*math.sqrt(sma**3/muPrimary)

        t0 = M.Epoch(shift_datetime_string(EpochDuration_FinalEpoch, T*float(EpochDuration_NumberofOrbitalPeriods), "final"))

        initialState, element_names = create_initial_State(globals(),boa,primary,frame,orbitalElements,t0,primary_j2,primary_j3)

    if 'EpochDuration_Timespan' in globals():

        if EpochDuration_Timespan_Units == "Seconds":

            timespanUnit = 1

        if EpochDuration_Timespan_Units == "Minutes":

            timespanUnit = 60

        if EpochDuration_Timespan_Units == "Hours":

            timespanUnit = 3600

        if EpochDuration_Timespan_Units == "Days":

            timespanUnit = 24*3600

        t0 = M.Epoch(shift_datetime_string(EpochDuration_FinalEpoch, timespanUnit*float(EpochDuration_Timespan), "final"))
        initialState, element_names = create_initial_State(globals(),boa,primary,frame,orbitalElements,t0,primary_j2,primary_j3)

inital_epoch_str = str(t0)

sunTraj = M.TrajQuery(boa,"Sun",primary,f"IAU {primary} Fixed")
sunState = sunTraj.state(tf, 2)
sunLongitude = M.UnitDbl.value(M.Geodetic.longitude(sunState))*180/math.pi

if orbitalElements not in ["Keplarian","Cartesian","Spherical","Equinoctial"]:

    orbitalElements = "Keplarian"


# ----------------------------------------------------------------------
# ADD FORCES
# ----------------------------------------------------------------------

forces = [
    M.GravityForce(boa,scName),
    #M.AtmDragForce(boa,scName)
    ]

basicGrav.add(boa, scName, ["Sun", primary])


# ----------------------------------------------------------------------
# INTEGRATE TRAJECTORY
# ----------------------------------------------------------------------

if primary == "Mercury":

    inertialFrame = "IAU Mercury Pole"

elif primary == "Venus":

    inertialFrame = "IAU Venus Pole"

elif primary == "Earth":

    inertialFrame = "EME2000"

elif primary == "Mars":

    inertialFrame = "Mars Inertial"

integInitialState = M.IntegState( boa, t0, tf, [], scName, primary,
                                  inertialFrame, inertialFrame, initialState,
                                  forces, False, [], [] )

integ = M.IntegSetup(boa)
integ.add(integInitialState)

prop = M.DivaPropagator(boa, "DIVA", integ)
prop.create(boa, t0, tf)

trajQuery = M.TrajQuery(boa, scName, primary,inertialFrame)
trajQueryGeo = M.TrajQuery(boa, scName, primary,f"IAU {primary} Fixed")


# ----------------------------------------------------------------------
# COLLECT STATES FROM TRAJECTORY AND BUILD POSITION ARRAYS
# ----------------------------------------------------------------------

tArray = M.Epoch.range(t0,tf,M.UnitDbl(float(EpochDuration_SimulationTimeStep), EpochDuration_SimulationTimeStep_Units))

states = []

latitudes = []
longitudes = []
heights = []

latitudes1 = []
longitudes1 = []
heights1 = []

xPositions = []
yPositions = []
zPositions = []

SemiMajorAxis = []
Eccentricity = []
Inclination = []
RAAN = []
ARGP = []
TrueAnomaly = []

XPosition = []
YPosition = []
ZPosition = []
XVelocity = []
YVelocity = []
ZVelocity = []

Radius = []
RadialVelocity = []
Latitude = []
LatitudinalVelocity = []
Longitude = []
LongitudinalVelocity = []

H = []
K = []
P = []
Q = []

for t in tArray:

    state = trajQuery.state(t)
    stateGeo = trajQueryGeo.state(t) 

    states.append(state)

    latitude = M.UnitDbl.value(M.Geodetic.latitude(stateGeo))*180/math.pi
    longitude = M.UnitDbl.value(M.Geodetic.longitude(stateGeo))*180/math.pi
    height = M.UnitDbl.value(M.Geodetic.height(stateGeo))
    
    latitudes.append(latitude)
    longitudes.append(longitude)
    heights.append(height)

    if Plotting_3DVisualization:

        xPos = state.pos()[0]
        yPos = state.pos()[1]
        zPos = state.pos()[2]

        xPositions.append(xPos)
        yPositions.append(yPos)
        zPositions.append(zPos)

    if Plotting_OrbitalElements:

        if (orbitalElements == "Keplarian") | (orbitalElements == "Conic"):

            sma = M.Conic.semiMajorAxis(state)
            e = M.Conic.eccentricity(state)
            inc = M.Conic.inclination(state)
            argp = M.Conic.argumentOfLatitude(state)
            raan = M.Conic.longitudeOfNode(state)
            tAnom = M.Conic.trueAnomaly(state)

            SemiMajorAxis.append(M.UnitDbl.value(sma))
            Eccentricity.append(M.UnitDbl.value(e))
            Inclination.append(M.UnitDbl.value(inc)*180/math.pi)
            RAAN.append(M.UnitDbl.value(argp)*180/math.pi)
            ARGP.append(M.UnitDbl.value(raan)*180/math.pi)
            TrueAnomaly.append(M.UnitDbl.value(tAnom)*180/math.pi)

        if orbitalElements == "Cartesian":

            x = M.Cartesian.x(state)
            y = M.Cartesian.y(state)
            z = M.Cartesian.z(state)
            dX = M.Cartesian.dx(state)
            dY = M.Cartesian.dy(state)
            dZ = M.Cartesian.dz(state)

            XPosition.append(M.UnitDbl.value(x))
            YPosition.append(M.UnitDbl.value(y))
            ZPosition.append(M.UnitDbl.value(z))
            XVelocity.append(M.UnitDbl.value(dX))
            YVelocity.append(M.UnitDbl.value(dY))
            ZVelocity.append(M.UnitDbl.value(dZ))

        if orbitalElements == "Spherical":

            r = M.Spherical.radius(state)
            dR = M.Spherical.dradius(state)
            lat = M.Spherical.latitude(state)
            dLat = M.Spherical.dlatitude(state)
            long = M.Spherical.longitude(state)
            dLong = M.Spherical.dlongitude(state)

            Radius.append(M.UnitDbl.value(r))
            RadialVelocity.append(M.UnitDbl.value(dR))
            Latitude.append(M.UnitDbl.value(lat))
            LatitudinalVelocity.append(M.UnitDbl.value(dLat))
            Longitude.append(M.UnitDbl.value(long))
            LongitudinalVelocity.append(M.UnitDbl.value(dLong))

        if orbitalElements == "Equinoctial":

            h = M.Conic.equinoctialH(state)
            k = M.Conic.equinoctialK(state)
            p = M.Conic.equinoctialP(state)
            q = M.Conic.equinoctialQ(state)

            H.append(M.UnitDbl.value(h))
            K.append(M.UnitDbl.value(k))
            P.append(M.UnitDbl.value(p))
            Q.append(M.UnitDbl.value(q))


# ----------------------------------------------------------------------
# FIND GROUND STATION EVENTS FUNCTIONS
# ----------------------------------------------------------------------

def calculate_contact_durations(contact_events_dict,T,t0,tf):

    j2000 = datetime.strptime("01-JAN-2000 00:00:00.0000", "%d-%b-%Y %H:%M:%S.%f")

    initial_time_str_parts = str(t0).split()
    initial_time_str = f"{initial_time_str_parts[0]} {initial_time_str_parts[1]}"
    initial_time_dt = datetime.strptime(initial_time_str, "%d-%b-%Y %H:%M:%S.%f")
    initial_time = (initial_time_dt - j2000).total_seconds()

    final_time_str_parts = str(tf).split()
    final_time_str = f"{final_time_str_parts[0]} {final_time_str_parts[1]}"
    final_time_dt = datetime.strptime(final_time_str, "%d-%b-%Y %H:%M:%S.%f")
    final_time = (final_time_dt - j2000).total_seconds()

    contact_durations_avg = {}
    contact_durations_avg_per = {}
    contact_durations = {}

    num_periods = (final_time_dt-initial_time_dt).total_seconds()/T

    for key, values in contact_events_dict.items():

        times = {}

        for value in values:

            type_line_parts = str(value).splitlines()[2].split()
            type_line = f"{type_line_parts[1]} {type_line_parts[2]}"

            epoch_line_parts = str(value).splitlines()[3].split()
            epoch_line = f"{epoch_line_parts[1]} {epoch_line_parts[2]}"

            epoch = datetime.strptime(epoch_line, "%d-%b-%Y %H:%M:%S.%f")
            epoch_sec = (epoch - j2000).total_seconds()

            times[epoch_sec] = type_line

        i=0

        total_time = 0

        for epoch_sec in times:

            i+=1

            if i==1 and times[epoch_sec] == ": Set":

                total_time += epoch_sec - initial_time

            elif times[epoch_sec] == ": Rise" and i != len(times):

                rise_epoch = epoch_sec

            elif times[epoch_sec] == ": Set":

                total_time += epoch_sec - rise_epoch

            elif times[epoch_sec] == ": Rise" and i == len(times):

                total_time += final_time - epoch_sec
        
        if len(times) != 0:
            contact_durations_avg[key] = total_time/math.ceil(len(times))*2
        else:
            contact_durations_avg[key] = 0
        contact_durations_avg_per[key] = total_time/num_periods
        contact_durations[key] = total_time

    return contact_durations_avg, contact_durations_avg_per, contact_durations

def build_contact_array(contact_events_dict,time_step_seconds,num_entries):

    j2000 = datetime.strptime("01-JAN-2000 00:00:00.0000", "%d-%b-%Y %H:%M:%S.%f")

    contact = np.zeros(num_entries, dtype=bool)

    i=0

    for key, values in contact_events_dict.items():

        i+=1

        rise_times = []
        set_times = []

        for value in values:

            type_line_parts = str(value).splitlines()[2].split()
            type_line = f"{type_line_parts[1]} {type_line_parts[2]}"

            epoch_line_parts = str(value).splitlines()[3].split()
            epoch_line = f"{epoch_line_parts[1]} {epoch_line_parts[2]}"

            epoch = datetime.strptime(epoch_line, "%d-%b-%Y %H:%M:%S.%f")
            epoch_sec = (epoch - j2000).total_seconds()

            if type_line == ": Rise":

                rise_times.append(epoch_sec)

            elif type_line == ": Set":

                set_times.append(epoch_sec)

        globals()[f"station_contact{i}"] = np.zeros(num_entries, dtype=bool)

        enters = np.sort(np.array(rise_times, dtype=float))
        exits  = np.sort(np.array(set_times, dtype=float))

        i_rise, i_set = 0, 0
        in_contact = False

        for k in range(num_entries):
            t = k * time_step_seconds

            while i_rise < len(enters) and enters[i_rise] <= t:
                in_contact = True
                i_rise += 1
            while i_set < len(exits) and exits[i_set] <= t:
                in_contact = False
                i_set += 1

            globals()[f"station_contact{i}"][k] = in_contact

    for j in range(1,i+1):

        contact = np.logical_or(contact, globals()[f"station_contact{j}"])

    return contact


# ----------------------------------------------------------------------
# FIND GROUND STATION EVENTS
# ----------------------------------------------------------------------

contact_events_dict = {}
search_interval = M.TimeInterval( t0, tf )

M.DefaultHorizonMask.addAll( boa )

i=0
stations = {}

for station in GroundStations_Predefined:

    i+=1

    if not station[:3]=="DSS":
        station = station + "M1"

    globals()[f"trajQuery{i}"] = M.TrajQuery(boa, scName, station)
    globals()[f"groundStation{i}InView"] = M.HorizonMaskEvent(globals()[f"trajQuery{i}"], M.HorizonMaskEvent.CROSSING )
    globals()[f"station{i}contactEvents"] = globals()[f"groundStation{i}InView"].search(search_interval, time_step_seconds*sec)
    globals()[f"stationTrajQuery{i}"] = M.TrajQuery(boa, station, primary, f"IAU {primary} Fixed")

    station_lat = M.UnitDbl.value(M.Geodetic.latitude(globals()[f"stationTrajQuery{i}"].state(t0)))*180/math.pi
    station_long = M.UnitDbl.value(M.Geodetic.longitude(globals()[f"stationTrajQuery{i}"].state(t0)))*180/math.pi

    stations[station] = station_lat,station_long
    contact_events_dict[station] = globals()[f"station{i}contactEvents"]

contact_durations_avg, contact_durations_avg_per, contact_durations = calculate_contact_durations(contact_events_dict,T,str(t0),str(tf))
contact_bool = build_contact_array(contact_events_dict,time_step_seconds,len(states))


# ----------------------------------------------------------------------
# FIND SHADOW EVENTS FUNCTIONS
# ----------------------------------------------------------------------

def time_dif_sec(str1,str2):

    def parse_time_str(s):

        parts = s.strip().rsplit(' ', 1)

        if len(parts) != 2:

            raise ValueError(f"Invalid time string format: {s}")
        
        time_part = parts[0]

        if len(time_part) < 22:

            time_part += ".0000"
        return datetime.strptime(time_part, "%d-%b-%Y %H:%M:%S.%f")

    t1 = parse_time_str(str1.upper())
    t2 = parse_time_str(str2.upper())
    return (t2-t1).total_seconds()

def create_shadow_array(enter_times, exit_times, time_step, num_entries):

    shadow = np.zeros(num_entries, dtype=bool)

    enters = np.sort(np.array(enter_times, dtype=float))
    exits  = np.sort(np.array(exit_times, dtype=float))

    i_enter, i_exit = 0, 0
    in_shadow = False

    for k in range(num_entries):
        t = k * time_step

        while i_enter < len(enters) and enters[i_enter] <= t:
            in_shadow = True
            i_enter += 1
        while i_exit < len(exits) and exits[i_exit] <= t:
            in_shadow = False
            i_exit += 1

        shadow[k] = in_shadow

    return shadow

def build_shadow_array(events,initial_time_str,time_step_seconds,states):

    entry_epochs = []
    exit_epochs = []

    for event in events:
        lines = str(event).splitlines()
        entry_epoch_str = lines[3][7:]
        exit_epoch_str = lines[4][7:]

        entry_epoch_sec = time_dif_sec(initial_time_str,entry_epoch_str)
        exit_epoch_sec = time_dif_sec(initial_time_str,exit_epoch_str)

        entry_epochs.append(entry_epoch_sec)
        exit_epochs.append(exit_epoch_sec)

    return create_shadow_array(entry_epochs,exit_epochs,time_step_seconds,len(states))


# ----------------------------------------------------------------------
# FIND SHADOW EVENTS
# ----------------------------------------------------------------------

shadow_events_dict = {}

primary_umbra_events = M.ShadowEvent( boa, primary, scName, M.ShadowEvent.IN_UMBRA ).search( search_interval, time_step_seconds*sec )
primary_penumbra_events = M.ShadowEvent( boa, primary, scName, M.ShadowEvent.IN_PENUMBRA ).search( search_interval, time_step_seconds*sec )

primary_umbra_array = build_shadow_array(primary_umbra_events,EpochDuration_InitialEpoch,time_step_seconds,states)
primary_penumbra_array = build_shadow_array(primary_penumbra_events,EpochDuration_InitialEpoch,time_step_seconds,states)

primary_shadow_array = primary_umbra_array | primary_penumbra_array

if primary == "Earth":

    bodies = [primary,"Moon"]

    moon_umbra_events = M.ShadowEvent( boa, "Moon", scName, M.ShadowEvent.IN_UMBRA ).search( search_interval, time_step_seconds*sec )
    moon_penumbra_events = M.ShadowEvent( boa, "Moon", scName, M.ShadowEvent.IN_PENUMBRA ).search( search_interval, time_step_seconds*sec )

    moon_umbra_array = build_shadow_array(moon_umbra_events,EpochDuration_InitialEpoch,time_step_seconds,states)
    moon_penumbra_array = build_shadow_array(moon_penumbra_events,EpochDuration_InitialEpoch,time_step_seconds,states)

    moon_shadow_array = moon_umbra_array | moon_penumbra_array

    shadow_array = primary_shadow_array | moon_shadow_array
    
elif primary == "Mars":

    bodies = [primary,"Phobos","Deimos"]

    phobos_umbra_events = M.ShadowEvent( boa, "Phobos", scName, M.ShadowEvent.IN_UMBRA ).search( search_interval, time_step_seconds*sec )
    phobos_penumbra_events = M.ShadowEvent( boa, "Phobos", scName, M.ShadowEvent.IN_PENUMBRA ).search( search_interval, time_step_seconds*sec )

    phobos_umbra_array = build_shadow_array(phobos_umbra_events,EpochDuration_InitialEpoch,time_step_seconds,states)
    phobos_penumbra_array = build_shadow_array(phobos_penumbra_events,EpochDuration_InitialEpoch,time_step_seconds,states)

    phobos_shadow_array = phobos_umbra_array | phobos_penumbra_array

    deimos_umbra_events = M.ShadowEvent( boa, "Deimos", scName, M.ShadowEvent.IN_UMBRA ).search( search_interval, time_step_seconds*sec )
    deimos_penumbra_events = M.ShadowEvent( boa, "Deimos", scName, M.ShadowEvent.IN_PENUMBRA ).search( search_interval, time_step_seconds*sec )

    deimos_umbra_array = build_shadow_array(deimos_umbra_events,EpochDuration_InitialEpoch,time_step_seconds,states)
    deimos_penumbra_array = build_shadow_array(deimos_penumbra_events,EpochDuration_InitialEpoch,time_step_seconds,states)
    
    deimos_shadow_array = deimos_umbra_array | deimos_penumbra_array

    shadow_array = primary_shadow_array | phobos_shadow_array | deimos_shadow_array

else:

    bodies = primary

    shadow_array = primary_shadow_array

i=0

for body in bodies:

    if body == primary:

        shadow_events_dict[body] = globals()["primary_umbra_events"] + globals()["primary_penumbra_events"]

    else:

        shadow_events_dict[body] = globals()[f"{body.lower()}_umbra_events"] + globals()[f"{body.lower()}_penumbra_events"]


# ----------------------------------------------------------------------
# CALCULATE CONICAL SENSOR GROUND COVERAGE FUNCTIONS
# ----------------------------------------------------------------------

@njit
def points_in_polygon_numba(x, y, poly_lon, poly_lat):
    n_vert = len(poly_lon)
    inside = np.zeros(x.shape[0], dtype=np.bool_)
    for k in range(x.shape[0]):
        px = x[k]
        py = y[k]
        c = False
        for i in range(n_vert):
            j = (i - 1) % n_vert
            xi, yi = poly_lon[i], poly_lat[i]
            xj, yj = poly_lon[j], poly_lat[j]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
                c = not c
        inside[k] = c
    return inside

def great_circle_offset(lat, lon, azimuth_deg, angular_distance_rad):
    lat1 = np.radians(lat)
    lon1 = np.radians(lon)
    az = np.radians(azimuth_deg)

    sin_lat1 = np.sin(lat1)
    cos_lat1 = np.cos(lat1)
    sin_d = np.sin(angular_distance_rad)
    cos_d = np.cos(angular_distance_rad)

    lat2 = np.arcsin(sin_lat1 * cos_d + cos_lat1 * sin_d * np.cos(az))
    lon2 = lon1 + np.arctan2(
        np.sin(az) * sin_d * cos_lat1,
        cos_d - sin_lat1 * np.sin(lat2)
    )

    return np.degrees(lat2), (np.degrees(lon2) + 540) % 360 - 180

def compute_swath_edges_great_circle(
    latitudes, longitudes, altitudes_km,
    sensor_fov_deg, radius_eq_km, radius_pole_km
):
    latitudes = np.asarray(latitudes)
    longitudes = np.unwrap(np.radians(longitudes))
    longitudes = np.degrees(longitudes)
    altitudes_km = np.asarray(altitudes_km)

    dlat = np.diff(latitudes, append=latitudes[-1])
    dlon = np.diff(longitudes, append=longitudes[-1])
    azimuths = (np.degrees(np.arctan2(dlon, dlat)) + 360) % 360

    lat_rad = np.radians(latitudes)
    r_surface = np.sqrt((radius_eq_km * np.cos(lat_rad))**2 +
                        (radius_pole_km * np.sin(lat_rad))**2)

    half_fov_rad = np.radians(sensor_fov_deg / 2)
    swath_offset_rad = np.tan(half_fov_rad) * altitudes_km / r_surface

    left_lat, left_lon = great_circle_offset(latitudes, longitudes, azimuths - 90, swath_offset_rad)
    right_lat, right_lon = great_circle_offset(latitudes, longitudes, azimuths + 90, swath_offset_rad)

    return left_lat, left_lon, right_lat, right_lon

def build_swath_polygons_from_track_pairwise(
    latitudes, longitudes, altitudes_km,
    sensor_fov_deg, radius_eq_km, radius_pole_km
):
    left_lat, left_lon, right_lat, right_lon = compute_swath_edges_great_circle(
        latitudes, longitudes, altitudes_km,
        sensor_fov_deg, radius_eq_km, radius_pole_km
    )

    swath_polygons = []

    for i in range(len(latitudes) - 1):

        lon_pair = np.unwrap(np.radians([left_lon[i], left_lon[i+1], right_lon[i+1], right_lon[i]]))
        lon_pair = np.degrees(lon_pair)

        poly_lat = np.array([
            left_lat[i],
            left_lat[i+1],
            right_lat[i+1],
            right_lat[i]
        ])
        poly_lon = lon_pair

        swath_polygons.append((poly_lat, poly_lon))

    return swath_polygons

def calculate_coverage(
    latitudes, longitudes, altitudes_km,
    in_shadow,
    sensor_fov_deg,
    lat_res_deg, lon_res_deg,
    radius_eq_km, radius_pole_km
):
    latitudes = np.asarray(latitudes)
    longitudes = np.asarray(longitudes)
    altitudes_km = np.asarray(altitudes_km)
    in_shadow = np.asarray(in_shadow)

    lat_grid = np.arange(90, -90 - lat_res_deg, -lat_res_deg)
    lon_grid = np.arange(-180, 180, lon_res_deg)
    lat_mesh, lon_mesh = np.meshgrid(lat_grid, lon_grid, indexing='ij')
    lat_flat = lat_mesh.ravel()
    lon_flat = lon_mesh.ravel()
    total_cells = lat_flat.size

    coverage = np.zeros_like(lat_mesh, dtype=np.uint8)
    coverage_lit_only = np.zeros_like(lat_mesh, dtype=np.uint8)

    def apply_polygons(polygons, cov_matrix):
        for poly_lat, poly_lon in polygons:
            lat_min, lat_max = poly_lat.min(), poly_lat.max()
            lon_min, lon_max = poly_lon.min(), poly_lon.max()

            if lon_max - lon_min > 180:
                lon_wrap = ((lon_flat - lon_min + 360) % 360) + lon_min
            else:
                lon_wrap = lon_flat

            mask_bbox = (
                (lat_flat >= lat_min) & (lat_flat <= lat_max) &
                (lon_wrap >= lon_min) & (lon_wrap <= lon_max)
            )

            reduced_lats = lat_flat[mask_bbox]
            reduced_lons = lon_wrap[mask_bbox]
            inside_bbox = points_in_polygon_numba(reduced_lons, reduced_lats, poly_lon, poly_lat)

            cov_matrix.ravel()[np.where(mask_bbox)[0][inside_bbox]] = 1

    all_polys = build_swath_polygons_from_track_pairwise(
        latitudes, longitudes, altitudes_km,
        sensor_fov_deg, radius_eq_km, radius_pole_km
    )
    apply_polygons(all_polys, coverage)

    lat_lit = latitudes[~in_shadow]
    lon_lit = longitudes[~in_shadow]
    alt_lit = altitudes_km[~in_shadow]

    if len(lat_lit) > 1:
        lit_polys = build_swath_polygons_from_track_pairwise(
            lat_lit, lon_lit, alt_lit,
            sensor_fov_deg, radius_eq_km, radius_pole_km
        )
        apply_polygons(lit_polys, coverage_lit_only)

    percent_all = 100.0 * np.sum(coverage) / total_cells
    percent_lit = 100.0 * np.sum(coverage_lit_only) / total_cells

    return coverage, coverage_lit_only, percent_all, percent_lit


# ----------------------------------------------------------------------
# CALCULATE CONICAL SENSOR GROUND COVERAGE
# ----------------------------------------------------------------------

latitudinal_resolution = .125
longitudinal_resolution = .25

a,b,c,d = calculate_coverage(latitudes, longitudes, heights, shadow_array,            
                             float(SpacecraftPhysicalProperties_ConicalSensorFOVdeg),       
                            longitudinal_resolution, latitudinal_resolution, 
                            primary_equitorial_radius, primary_polar_radius)

total_coverage, lit_coverage, total_percent, lit_percent = a,b,c,d


# ----------------------------------------------------------------------
# CALCULATE GROUNDTRACK REPEAT TIME AND NODAL SPACING
# ----------------------------------------------------------------------

repeatState = trajQuery.state(t0)

semimajoraxis = M.UnitDbl.value(M.Conic.semiMajorAxis(repeatState))
eccentricity = M.UnitDbl.value(M.Conic.eccentricity(repeatState))
inclination = M.UnitDbl.value(M.Conic.inclination(repeatState))

pos = M.UnitDbl.value(initialState.posMag())
vel = M.UnitDbl.value(initialState.velMag())
primaryBodyDataBoa = M.BodyDataBoa.read(boa,primary)
primaryBodyData = M.BodyData(boa, primary,primaryBodyDataBoa.frame(),primary)
muPrimary = M.UnitDbl.value(M.BodyData.gm(primaryBodyData))
eps = vel**2/2-muPrimary/pos
sma = -muPrimary/(2*eps)
Tsatellite = 2*math.pi*math.sqrt(sma**3/muPrimary)

sunBodyDataBoa = M.BodyDataBoa.read(boa,"Sun")
sunBodyData = M.BodyData(boa, "Sun",sunBodyDataBoa.frame(),"Sun")
muSun = M.UnitDbl.value(M.BodyData.gm(sunBodyData))
primaryTraj = M.TrajQuery(boa,primary,"Sun",frame)
primaryState = primaryTraj.state(t, 2)
primarySunOrbitalRadius = primaryState.posMag()
primarySunOrbitalSpeed = primaryState.velMag()
primarySunSMA = (2/M.UnitDbl.value(primarySunOrbitalRadius)-M.UnitDbl.value(primarySunOrbitalSpeed)**2/M.UnitDbl.value(M.BodyData.gm(sunBodyData)))**-1
Tprimary = 2*math.pi*math.sqrt(primarySunSMA**3/muSun)

sunTraj = M.TrajQuery(boa,"Sun",primary,f"IAU {primary} Fixed")
tArray = M.Epoch.range(M.Epoch("01-JAN-2000 00:00:00 ET"),M.Epoch("03-JAN-2000 00:00:00 ET"),M.UnitDbl(1,"seconds"))
sunLongitudes = []

for t in tArray:

    sunState = sunTraj.state(t, 2)
    sunLongitude = M.UnitDbl.value(M.Geodetic.longitude(sunState))
    sunLongitudes.append(sunLongitude)

sunLongitudesArray = np.array(sunLongitudes)
minima_indices, _ = find_peaks(-sunLongitudesArray)
primarySolarDay = (minima_indices[1] - minima_indices[0])
primarySideralDay = Tprimary/(Tprimary/primarySolarDay + 1)

omegaDot = math.cos(inclination)*(-3/2*math.sqrt(muPrimary)*primary_j2*primary_equitorial_radius**2/((1-eccentricity**2)**2*semimajoraxis**3.5))

NdArray = range(1,100001)
NsArray = []

for Nd in NdArray:

    Ns = Nd*primarySideralDay*(1+omegaDot)/Tsatellite
    NsArray.append(Ns)


diffs = []

for Ns in NsArray:

    i+=1
    NsInt = round(Ns)
    diff = abs(Ns-NsInt)
    diffs.append(diff)

diffArray = np.array(diffs)
minDiffIdx = np.argmin(diffArray)

if InitialOrbitalElements_Type == "Repeat Ground Track":

    repeat_num_days = InitialOrbitalElements_RepeatTimedays
    repeat_num_periods = str(round(NsArray[minDiffIdx]/minDiffIdx*float(InitialOrbitalElements_RepeatTimedays)))
    nodal_spacing = str(round(360*minDiffIdx/NsArray[minDiffIdx],2)%360)

else:

    repeat_num_days = str(round(minDiffIdx))
    repeat_num_periods = str(round(NsArray[minDiffIdx]))
    nodal_spacing = str(round(360*minDiffIdx/NsArray[minDiffIdx],2)%360)
    

# ----------------------------------------------------------------------
# EXPORT SIMULATION DATA TO JSON
# ----------------------------------------------------------------------

stop = time.time()
duration = str(stop - start)

print()
cprint(f"MONTE Simulation Complete. Elapsed Time: {duration[:4]} [Seconds] ","green")

def make_dict_serializable(d):
    """
    Recursively convert a dictionary to a JSON-serializable form.
    - Sets/tuples become lists.
    - NumPy arrays become lists.
    """
    serializable_dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            serializable_dict[k] = make_dict_serializable(v)
        elif isinstance(v, (set, tuple)):
            serializable_dict[k] = list(v)
        elif isinstance(v, np.ndarray):
            serializable_dict[k] = v.tolist()
        else:
            try:
                json.dumps(v)
                serializable_dict[k] = v
            except (TypeError, OverflowError):
                serializable_dict[k] = str(v)
    return serializable_dict

def is_json_serializable(obj):
    try:
        json.dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False

def export_globals_to_json(path):

    global_vars = globals()
    export_dict = {}

    for name, value in global_vars.items():
        if name.startswith("__"):
            continue
        if callable(value):
            continue

        if isinstance(value, dict):
            export_dict[name] = make_dict_serializable(value)
        elif isinstance(value, np.ndarray):
            export_dict[name] = value.tolist()
        elif is_json_serializable(value):
            export_dict[name] = value

    with open(path, "w") as f:
        json.dump(export_dict, f, indent=2)

export_globals_to_json("monte_data.json")


# ----------------------------------------------------------------------
# CREATE PLOTS AND DATA OUTPUT WINDOW
# ----------------------------------------------------------------------

commands = [
    ["python3.9", "3dPlot.py", "input_data.json", "monte_data.json"],
    ["python3.9", "groundTrackPlot.py", "input_data.json", "monte_data.json"],
    ["python3.9", "orbitalElementsPlot.py", "input_data.json", "monte_data.json"],
    ["python3.9", "dataWindow.py", "input_data.json", "monte_data.json"]
]

processes = [subprocess.Popen(cmd) for cmd in commands]

for p in processes:
    p.wait()