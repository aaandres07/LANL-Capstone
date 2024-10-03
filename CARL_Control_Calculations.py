import math
import numpy as np
from scipy import integrate
import statistics
from scipy.interpolate import interp1d

''''
#######################################

Dictionaries

  1. experiment1
        Experiment
        -----------
        time range:             array 
        intial conditions:      array  [0] : rover velocity [m/s] [1] : rover position [m]
        alpha distance:         array
        alpha degrees:          array
        Rolling coefficient:    0.1

        End Event
        ----------
        Max Distance           1000 [meters]
        Max Time               5000 [seconds]
        Min velocity           0.01 [m/s]
        
  
  2. define_rover_1
       Wheel:
       -----------
        radius                0.3  [meters]
        mass                  1    [kg]
    speed reducer:  type, diameter pinion & gear, mass
       Motor:  
       -------             
        torque stall          170   [N-m]
        torque no load        0     [N-m]
        speed no load         3.80  [rad/s]
        mass                  5.0   [kg]
        efficiency tau        array
        efficiency            array
      Chassis
      -------
        mass                  659 [kg]
      Science Payload
      ---------------
        mass                  75 [kg]
      Power Subassembly
      ----------------
        mass                  90 [kg]

      Wheel Assembly          wheel, speed_reducer, motor
      Rover                   wheel_assembly, chassis, science_payload, power_subsys

      planet                  9.81 [m/s^2]

      
      
########################################
'''
 
def experiment1():
    
    experiment = {'time_range' : np.array([0,20000]),
                  'initial_conditions' : np.array([0.01,0]),
                  'alpha_dist' : np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]),
                  'alpha_deg' : np.array([0, 0, 0, 0, \
                                        0, 0, 0, 0, \
                                        0, 0, 0]),
                  'Crr' : 0.05}
    
    
    # Below are default values for example only:
    end_event = {'max_distance' : 5000,
                 'max_time' : 600, #10 minutes
                 'min_velocity' : 0.00}
    
    return experiment, end_event


def define_rover_1():
    # Initialize Rover dict for testing
    wheel = {'radius':0.127, #10-inch wheels
             'mass': 1.317} #https://trampaboards.com/10-inch-megastars-wheel-with-offset-bearing-position--any-10-inch-tyre-p-27384.html
    speed_reducer = {'type':'reverted',
                     'diam_pinion':1, #No gear ratio
                     'diam_gear':1,
                     'mass':0}
    motor = {'torque_stall': 12, #Operating torque 0.49 N-m
             'torque_noload':0,  
             'speed_noload': 29.94985,  #286 rpm
             'mass': 1.36, #3 lbs probably change based on motor selection
             'effcy_tau':np.array([0,0.10,0.45,0.49,0.08,12]), #changes based on motor specs
             'effcy':np.array([0,0.55,0.75,0.71,0.50,0.05])}
    
        
    chassis = {'mass': 54.4311} # Aluminum (120 lbs)
    science_payload = {'mass': 0.907185} # Senor & hardware equipment (2 lbs)
    power_subsys = {'mass': 9.07185} # Battery (20 lbs)
    
    wheel_assembly = {'wheel':wheel,
                      'speed_reducer':speed_reducer,
                      'motor':motor}
    
    rover = {'wheel_assembly':wheel_assembly,
             'chassis':chassis,
             'science_payload':science_payload,
             'power_subsys':power_subsys}
    
    planet = {'g':9.81}
    
    
    # return everything we need
    return rover, planet

experiment, end_event = experiment1()
rover, planet = define_rover_1()


def effcyfun():
    """General Description:
    Creates a function using the motor torque effiency 
    """
    
    effcy_tau = rover['wheel_assembly']['motor']['effcy_tau']
    effcy = rover['wheel_assembly']['motor']['effcy']
    effcy_fun = interp1d(effcy_tau, effcy, kind = 'cubic') # fit the cubic spline
    
    return effcy_fun


def alphafun():
    """General Description:
    Creates a function using the terrain angle 
    """ 
    
    alpha_dist = experiment['alpha_dist']
    alpha_deg = experiment['alpha_deg']
    #create interpolation function using distance and deg data points
    alpha_fun = interp1d(alpha_dist, alpha_deg, kind = 'cubic', fill_value='extrapolate') #fit the cubic spline
    
    return alpha_fun


def get_mass(rover):
    """General Description:
    Calculates total mass of rover
    
    Inputs:  rover:  dict      Data structure containing rover parameters
    
    Outputs:     m:  scalar    Rover mass [kg].
    """
    
    # Check that the input is a dict
    if type(rover) != dict:
        raise Exception('Input must be a dict')
    
    # add up mass of chassis, power subsystem, science payload, 
    # and components from all six wheel assemblies
    m = rover['chassis']['mass'] \
        + rover['power_subsys']['mass'] \
        + rover['science_payload']['mass'] \
        + 6*rover['wheel_assembly']['motor']['mass'] \
        + 6*rover['wheel_assembly']['speed_reducer']['mass'] \
        + 6*rover['wheel_assembly']['wheel']['mass'] \
    
    return m


def get_gear_ratio(speed_reducer):
    """General Description:
    Finds gear ratio of gear box
    
    Inputs:  speed_reducer:  dict      Data dictionary specifying speed
                                        reducer parameters
                                        
    Outputs:            Ng:  scalar    Speed ratio from input pinion shaft
                                        to output gear shaft. Unitless.
    """
    
    # Check that the input is a dict
    if type(speed_reducer) != dict:
        raise Exception('Input must be a dict')
    
    # Check 'type' field (not case sensitive)
    if speed_reducer['type'].lower() != 'reverted':
        raise Exception('The speed reducer type is not recognized.')
    
    # Main code
    d1 = speed_reducer['diam_pinion']
    d2 = speed_reducer['diam_gear']
    
    Ng = (d2/d1)**2
    
    return Ng


def tau_dcmotor(omega, motor):
    """General Description:
    Calculates torque of motor
    
    Inputs:  omega:  numpy array      Motor shaft speed [rad/s]
             motor:  dict             Data dictionary specifying motor parameters
             
    Outputs:   tau:  numpy array      Torque at motor shaft [Nm].  Return argument
                                      is same size as first input argument.
    """
    
    # Check that the first input is a scalar or a vector
    if (type(omega) != int) and (type(omega) != float) and (not isinstance(omega, np.ndarray)):
        raise Exception('First input must be a scalar or a vector. If input is a vector, it should be defined as a numpy array.')
    elif not isinstance(omega, np.ndarray):
        omega = np.array([omega],dtype=float) # make the scalar a numpy array
    elif len(np.shape(omega)) != 1:
        raise Exception('First input must be a scalar or a vector. Matrices are not allowed.')

    # Check that the second input is a dict
    if type(motor) != dict:
        raise Exception('Second input must be a dict')
        
    # Main code
    tau_s    = motor['torque_stall']
    tau_nl   = motor['torque_noload']
    omega_nl = motor['speed_noload']
    
    # initialize
    tau = np.zeros(len(omega),dtype = float)
    for ii in range(len(omega)):
        if omega[ii] >= 0 and omega[ii] <= omega_nl:
            tau[ii] = tau_s - (tau_s-tau_nl)/omega_nl *omega[ii]
        elif omega[ii] < 0:
            tau[ii] = tau_s
        elif omega[ii] > omega_nl:
            tau[ii] = 0
        
    return tau
    
    


def F_rolling(omega, terrain_angle, rover, planet, Crr):
    """General Description:
    Calculates rolling force on the rover
    
    Inputs:           omega:  numpy array     Motor shaft speed [rad/s]
              terrain_angle:  numpy array     Array of terrain angles [deg]
                      rover:  dict            Data structure specifying rover 
                                              parameters
                    planet:  dict            Data dictionary specifying planetary 
                                              parameters
                        Crr:  scalar          Value of rolling resistance coefficient [-]
    
    Outputs:           Frr:  numpy array     Array of forces [N]
    """
    
    # Check that the first input is a scalar or a vector
    if (type(omega) != int) and (type(omega) != float) and (not isinstance(omega, np.ndarray)):
        raise Exception('First input must be a scalar or a vector. If input is a vector, it should be defined as a numpy array.')
    elif not isinstance(omega, np.ndarray):
        omega = np.array([omega],dtype=float) # make the scalar a numpy array
    elif len(np.shape(omega)) != 1:
        raise Exception('First input must be a scalar or a vector. Matrices are not allowed.')
        
    # Check that the second input is a scalar or a vector
    if (type(terrain_angle) != int) and (type(terrain_angle) != float) and (not isinstance(terrain_angle, np.ndarray)):
        raise Exception('Second input must be a scalar or a vector. If input is a vector, it should be defined as a numpy array.')
    elif not isinstance(terrain_angle, np.ndarray):
        terrain_angle = np.array([terrain_angle],dtype=float) # make the scalar a numpy array
    elif len(np.shape(terrain_angle)) != 1:
        raise Exception('Second input must be a scalar or a vector. Matrices are not allowed.')
        
    # Check that the first two inputs are of the same size
    if len(omega) != len(terrain_angle):
        raise Exception('First two inputs must be the same size')
    
    # Check that values of the second input are within the feasible range  
    if max([abs(x) for x in terrain_angle]) > 75:    
        raise Exception('All elements of the second input must be between -75 degrees and +75 degrees')
        
    # Check that the third input is a dict
    if type(rover) != dict:
        raise Exception('Third input must be a dict')
        
    # Check that the fourth input is a dict
    if type(planet) != dict:
        raise Exception('Fourth input must be a dict')
        
    # Check that the fifth input is a scalar and positive
    if (type(Crr) != int) and (type(Crr) != float):
        raise Exception('Fifth input must be a scalar')
    if Crr <= 0:
        raise Exception('Fifth input must be a positive number')
        
    # Main Code
    m = get_mass(rover)
    g = planet['g']
    r = rover['wheel_assembly']['wheel']['radius']
    Ng = get_gear_ratio(rover['wheel_assembly']['speed_reducer'])
    
    v_rover = r*omega/Ng
    
    Fn = np.array([m*g*math.cos(math.radians(x)) for x in terrain_angle],dtype=float) # normal force
    Frr_simple = -Crr*Fn # simple rolling resistance
    
    Frr = np.array([math.erf(40*v_rover[ii]) * Frr_simple[ii] for ii in range(len(v_rover))], dtype = float)
    
    return Frr


def F_gravity(terrain_angle, rover, planet):
    """General Description:
    Calculates force of gravity on the rover
    
    Inputs:  terrain_angle:  numpy array   Array of terrain angles [deg]
                     rover:  dict          Data structure specifying rover 
                                            parameters
                    planet:  dict          Data dictionary specifying planetary 
                                            parameters
    
    Outputs:           Fgt:  numpy array   Array of forces [N]
    """
    
    # Check that the first input is a scalar or a vector
    if (type(terrain_angle) != int) and (type(terrain_angle) != float) and (not isinstance(terrain_angle, np.ndarray)):
        raise Exception('First input must be a scalar or a vector. If input is a vector, it should be defined as a numpy array.')
    elif not isinstance(terrain_angle, np.ndarray):
        terrain_angle = np.array([terrain_angle],dtype=float) # make the scalar a numpy array
    elif len(np.shape(terrain_angle)) != 1:
        raise Exception('First input must be a scalar or a vector. Matrices are not allowed.')
        
    # Check that values of the first input are within the feasible range  
    if max([abs(x) for x in terrain_angle]) > 75:    
        raise Exception('All elements of the first input must be between -75 degrees and +75 degrees')

    # Check that the second input is a dict
    if type(rover) != dict:
        raise Exception('Second input must be a dict')
    
    # Check that the third input is a dict
    if type(planet) != dict:
        raise Exception('Third input must be a dict')
        
    # Main Code
    m = get_mass(rover)
    g = planet['g']
    
    Fgt = np.array([-m*g*math.sin(math.radians(x)) for x in terrain_angle], dtype = float)
        
    return Fgt


def F_drive(omega, rover):
    """General Description:
    Calculates required drice force to accelerate rover
    
    Inputs:  omega:  numpy array   Array of motor shaft speeds [rad/s]
             rover:  dict          Data dictionary specifying rover parameters
    
    Outputs:    Fd:  numpy array   Array of drive forces [N]
    """
    
    # Check that the first input is a scalar or a vector
    if (type(omega) != int) and (type(omega) != float) and (not isinstance(omega, np.ndarray)):
        raise Exception('First input must be a scalar or a vector. If input is a vector, it should be defined as a numpy array.')
    elif not isinstance(omega, np.ndarray):
        omega = np.array([omega],dtype=float) # make the scalar a numpy array
    elif len(np.shape(omega)) != 1:
        raise Exception('First input must be a scalar or a vector. Matrices are not allowed.')

    # Check that the second input is a dict
    if type(rover) != dict:
        raise Exception('Second input must be a dict')
    
    # Main code
    Ng = get_gear_ratio(rover['wheel_assembly']['speed_reducer'])
    
    tau = tau_dcmotor(omega, rover['wheel_assembly']['motor'])
    tau_out = tau*Ng
    
    r = rover['wheel_assembly']['wheel']['radius']
    
    # Drive force for one wheel
    Fd_wheel = tau_out/r 
    
    # Drive force for all six wheels
    Fd = 6*Fd_wheel
    
    return Fd


def F_net(omega, terrain_angle, rover, planet, Crr):
    """General Description:
    Calculates net force acting on the rover
    
    Inputs:           omega:  list     Motor shaft speed [rad/s]
              terrain_angle:  list     Array of terrain angles [deg]
                      rover:  dict     Data structure specifying rover 
                                      parameters
                     planet:  dict     Data dictionary specifying planetary 
                                      parameters
                        Crr:  scalar   Value of rolling resistance coefficient
                                      [-]
    
    Outputs:           Fnet:  list     Array of forces [N]
    """
    
    # Check that the first input is a scalar or a vector
    if (type(omega) != int) and (type(omega) != float) and (not isinstance(omega, np.ndarray)):
    # if (not isinstance(omega, np.ndarray)):
        raise Exception('First input must be a scalar or a vector. If input is a vector, it should be defined as a numpy array.')
    elif not isinstance(omega, np.ndarray):
        omega = np.array([omega],dtype=float) # make the scalar a numpy array
    elif len(np.shape(omega)) != 1:
        raise Exception('First input must be a scalar or a vector. Matrices are not allowed.')
        
    # Check that the second input is a scalar or a vector
    if (type(terrain_angle) != int) and (type(terrain_angle) != float) and (not isinstance(terrain_angle, np.ndarray)):
        raise Exception('Second input must be a scalar or a vector. If input is a vector, it should be defined as a numpy array.')
    elif not isinstance(terrain_angle, np.ndarray):
        terrain_angle = np.array([terrain_angle],dtype=float) # make the scalar a numpy array
    elif len(np.shape(terrain_angle)) != 1:
        raise Exception('Second input must be a scalar or a vector. Matrices are not allowed.')
        
    # Check that the first two inputs are of the same size
    if len(omega) != len(terrain_angle):
        raise Exception('First two inputs must be the same size')
    
    # Check that values of the second input are within the feasible range  
    if max([abs(x) for x in terrain_angle]) > 75:    
        raise Exception('All elements of the second input must be between -75 degrees and +75 degrees')
        
    # Check that the third input is a dict
    if type(rover) != dict:
        raise Exception('Third input must be a dict')
        
    # Check that the fourth input is a dict
    if type(planet) != dict:
        raise Exception('Fourth input must be a dict')
        
    # Check that the fifth input is a scalar and positive
    if (type(Crr) != int) and (type(Crr) != float):
        raise Exception('Fifth input must be a scalar')
    if Crr <= 0:
        raise Exception('Fifth input must be a positive number')
    
    # Main Code
    Fd = F_drive(omega, rover)
    Frr = F_rolling(omega, terrain_angle, rover, planet, Crr)
    Fg = F_gravity(terrain_angle, rover, planet)
    
    Fnet = Fd + Frr + Fg # signs are handled in individual functions
    
    return Fnet


def motorW(v, rover):
    """General Description:
    Compute the rotational speed of the motor shaft [rad/s] given the translational velocity of the rover and the rover dictionary.
    
    Inputs:         v:       1D numpy array or scalar float/int     Rover translational velocity [m/s]
                rover:       dictionary                             Data structure containing rover parameters
    
    Outputs:        w:       1D numpy array or scalar float/int     Motor speed [rad/s]. The return value 
                                                                    should be the same size as input v.
    """

    # Validate the first input (v)
    if (type(v) != int) and (type(v) != float) and (not isinstance(v, np.ndarray)):
        raise Exception('First input must be a scalar or a vector. If input is a vector, it should be defined as a numpy array.')
    elif not isinstance(v, np.ndarray):
        v = np.array([v],dtype=float) # make the scalar a numpy array
    elif len(np.shape(v)) != 1:
        raise Exception('First input must be a scalar or a vector. Matrices are not allowed.')

    # Validate the second input (rover) to be a dictionary
    if type(rover) != dict:
        raise Exception('Second input must be a dict')
    
    # Calculate gear ratio and radius of wheel
    Ng = get_gear_ratio(rover['wheel_assembly']['speed_reducer'])
    r = rover['wheel_assembly']['wheel']['radius']
    
    # Compute rotation speed of motor shaft
    w = np.array([])
    for i in v:
        w = np.append(w,(i*Ng)/r )
        
    return w


def rover_dynamics(t, y, rover, planet, experiment):
    """General Description:
    This function computes the derivative of the state vector ([velocity, position]) for the rover given its current state. 
        
    Inputs:              t:  scalar        Time sample [s]
                         y:  numpy array   Two-element array of depenedent variables (rover velocity [m/s],rover position [m]) 
                    
                     rover:  dict          Data dictionary specifying rover parameters
                    planet:  dict          Data dictionary specifying planetary parameters
                experiment:  dict          Data dictionary specifying experiment paramaters
    
    Outputs:          dydt:  numpy array   Two-element array of depenedent variables (rover acceleration [m/s^2],rover velocity [m/s])
    """
    
    #Check that the first input is a postive scalar
    if (type(t) != int) and (type(t) != float) and (type(t) != np.float64):
        raise Exception('First input must be a scalar.')
    if t < 0:
        raise Exception('First input must be a positive number.')
    
    #Check that your second input is a two element numpy array
    if not isinstance(y, np.ndarray):
        raise Exception('Second input must be a numpy array.')
    elif len(y) != 2:
        raise Exception('Second input must be a vector of two elements only.')
    
    # Check that the third input is a dict
    if type(rover) != dict:
        raise Exception('Third input must be a dict')
        
    # Check that the fourth input is a dict
    if type(planet) != dict:
        raise Exception('Fourth input must be a dict')
        
    # Check that the fifth input is a dict
    if type(experiment) != dict:
        raise Exception('Fifth input must be a dict')
        
    # Calculate angular velocity of the motor based on the current velocity and rover parameters
    w = motorW(float(y[0]),rover)
    
    #Get Crr for rolling force
    Crr = experiment['Crr']
    
    # Get the terrain slope angle based on the current position
    alpha_fun = alphafun()
    terrain_angle = float(alpha_fun(y[1]))
    
    # Calculate the net force on the rover considering motor power, terrain angle, and other parameters
    Fnet = F_net(w,terrain_angle,rover,planet,Crr)
    
    # Extract rover's mass using its parameters
    m = get_mass(rover)

    # Compute acceleration by dividing net force by the rover's mass
    accel = float(Fnet/m)

    #velocity is given through input state vector
    vel = float(y[0])

    # Construct the derivative state vector containing acceleration and velocity
    dydt = np.array([accel,vel])
    
    return dydt



def mechpower(v,rover):
    """General Description:
    This function computes the instantaneous mechanical power output by a single DC motor at each point in a given velocity profile.
    
    Inputs:         v:      1D numpy array or scalar float/int  - Rover velocity data obtained from a simulation [m/s]
                rover:      dict                                - Data structure containing rover definition
    
    Outputs:        P:      1D numpy array or scalar float/int  - Instantaneous power output of a single motor corresponding to each element in v [W]
    """
    
    # Validate that the input 'v' is either a scalar or a 1D numpy array
    if (type(v) != int) and (type(v) != float) and (not isinstance(v, np.ndarray)):
        raise Exception('First input must be a scalar or a vector. If input is a vector, it should be defined as a numpy array.')
        
    elif not isinstance(v, np.ndarray):
        v = np.array([v],dtype=float) # make the scalar a numpy array
        
    elif len(np.shape(v)) != 1:
        raise Exception('First input must be a scalar or a vector. Matrices are not allowed.')

    # Validate that the second input is a dictionary
    if type(rover) != dict:
        raise Exception('Second input must be a dict')
        
    # Extract motor data from the rover dictionary
    motor = rover['wheel_assembly']['motor']    
    w = motorW(v,rover)             # Get omega for the given velocity
    tau = tau_dcmotor(w, motor)     # Get torque for the given omega   

    # compare length of tau and w arrays (they should be the same)
    if len(w) != len(tau):
        raise Exception('Tau and omega arrays must be the same size!')

    # compute instantaneous power output relative to time
    P = np.zeros(len(w),dtype=float)
    for i in range(len(w)):
        P[i] = w[i]*tau[i]     # Power = Torque * Velocity
    
    return P

def battenergy(t, v, rover):
    """General Description:
    Compute the total electrical energy consumed from the rover battery pack over a simulation profile.
    
    Inputs:         t:      1D numpy array     N-element array of time samples from a rover simulation [s]
                    v:      1D numpy array     N-element array of rover velocity data from a simulation [m/s]
                rover:      dict               Data structure containing rover definition
        
    Outputs:        E:      scalar             Total electrical energy consumed from the rover battery pack 
                                               over the input simulation profile. [J]
    """    
    # Validate that 't' (time) is a 1D numpy array
    if (type(t) != int) and (type(t) != float) and (not isinstance(t, np.ndarray)):
        raise Exception('First input must be a vector. If input is a vector, it should be defined as a numpy array.')
    elif not isinstance(t, np.ndarray):
        t = np.array([t], dtype=float)  # make the scalar a numpy array
    elif len(t.shape) != 1:
        raise Exception('First input must be a vector. Matrices are not allowed.')

    # Validate that 'v' (velocity) is a 1D numpy array
    if (type(v) != int) and (type(v) != float) and (not isinstance(v, np.ndarray)):
        raise Exception('Second input must be a vector. If input is a vector, it should be defined as a numpy array.')
    elif not isinstance(v, np.ndarray):
        v = np.array([v], dtype=float)
    elif len(v.shape) != 1:
        raise Exception('Second input must be a vector. Matrices are not allowed.')

    # Check that time and velocity vectors have the same length
    if t.shape[0] != v.shape[0]:
        raise Exception('The first and second input should be vectors of equal length')

    # Validate that 'rover' input is a dictionary
    if not isinstance(rover, dict):
        raise Exception('Third input must be a dict')

    # Compute mechanical power for the rover using the provided function
    P = mechpower(v, rover)

    # Extract efficiency function and motor data from the rover dictionary
    effcy_fun = effcyfun()
    motor = rover['wheel_assembly']['motor']

    # Initialize an array to store power values adjusted for motor efficiency
    P_array = np.array([])

    # Loop through each time step to compute the motor shaft rotational speed, torque, 
    # and adjust the power values based on motor efficiency
    for i in range(len(t)):
        omega = motorW(float(v[i]), rover)
        tau = tau_dcmotor(omega, motor)
        effcy = effcy_fun(tau)
        P_array = np.append(P_array, P[i]/effcy)

    # Integrate the power values over time to compute total energy consumption
    E = 6*np.trapz(P_array, t)

    return E

def simulate_rover(rover, planet, experiment, end_event):
    """General Description:
    Function to simulate the rover's movement and collect telemetry data.

    Inputs:        rover:      dict    Data structure containing rover definition
                  planet:      dict    Data structure containing planet terrain and environment
              experiment:      dict    Data structure containing experiment conditions
               end_event:      dict    Criteria to define the end of the simulation

    Outputs:       rover:      dict    Updated rover dictionary with telemetry data
    """
    # Validate that each input is a dictionary
    if(type(rover) != dict):
        raise Exception("Your first input must be a dictionary.")
    if(type(planet) != dict):
        raise Exception("Your second input must be a dictionary.")
    if(type(experiment) != dict):
        raise Exception("Your third input must be a dictionary.")
    if(type(end_event) != dict):
        raise Exception("Your fourth input must be a dictionary.")

    # Extract mission end event, time range, and initial conditions from the experiment dictionary
    event = end_of_mission_event(end_event)
    t = experiment['time_range']
    y = experiment['initial_conditions']

    # Define the rover dynamics function for integration
    function = lambda t,y: rover_dynamics(t,y,rover,planet,experiment)

    # Integrate the rover dynamics over time to get state estimates
    est = integrate.solve_ivp(function, t, y, method="RK45", events=event)

    # Extract position and velocity from state estimates
    pos = est.y[1,:]
    vel = est.y[0,:]
    time = est.t
    
    #Represent deceleration at the end of mission to catch glider (last 15 seconds)
    vel[-10:] = np.linspace(vel[-11], 0, 10)
    
    #Represent deceleration during mission due to possible turns
    vel[-10:] = np.linspace(vel[-11], 0, 10)
    
    # Compute power and energy using auxiliary functions
    P = mechpower(vel,rover)
    E = battenergy(time, vel, rover)

    # Compute average velocity and total distance traveled
    avg_vel = statistics.mean(vel)    
    dist = avg_vel*time[-1]

    # Populating telemetry dictionary
    telemetry = {
        'Time': time,
        'completion_time':time[-1],
        'velocity':vel,
        'position':pos,
        'distance_traveled':dist,
        'max_velocity': max(vel),
        'average_velocity':avg_vel,
        'power':P,
        'max_power': np.max(P),
        'battery_energy':E,
        'energy_per_distance':E/dist
        }

    # Update rover dictionary with telemetry data
    rover['telemetry'] = telemetry
    
    return rover


def end_of_mission_event(end_event):
    """General Description:
        Defines an event that terminates the mission simulation. Mission is over
        when rover reaches a certain distance, has moved for a maximum simulation 
        time or has reached a minimum velocity.
    """
    
    mission_distance = end_event['max_distance']
    mission_max_time = end_event['max_time']
    mission_min_velocity = end_event['min_velocity']
    
    # Assume that y[1] is the distance traveled
    distance_left = lambda t,y: mission_distance - y[1]
    distance_left.terminal = True
    
    time_left = lambda t,y: mission_max_time - t
    time_left.terminal = True
    
    velocity_threshold = lambda t,y: y[0] - mission_min_velocity;
    velocity_threshold.terminal = True
    velocity_threshold.direction = -1
    
    events = [distance_left, time_left, velocity_threshold]
    
    return events

time = battery_energy = simulate_rover(rover, planet, experiment, end_event)['telemetry']['Time']
battery_energy = simulate_rover(rover, planet, experiment, end_event)['telemetry']['battery_energy']
energy_per_distance = simulate_rover(rover, planet, experiment, end_event)['telemetry']['energy_per_distance']
distance_traveled = simulate_rover(rover, planet, experiment, end_event)['telemetry']['distance_traveled']
completion_time = simulate_rover(rover, planet, experiment, end_event)['telemetry']['completion_time']
velocity = simulate_rover(rover, planet, experiment, end_event)['telemetry']['velocity']
power = simulate_rover(rover, planet, experiment, end_event)['telemetry']['power']
max_power = simulate_rover(rover, planet, experiment, end_event)['telemetry']['max_power']
max_velocity = simulate_rover(rover, planet, experiment, end_event)['telemetry']['max_velocity']

#print("Battery Energy Consumed = {:.2f} J".format(battery_energy))
#print("Energy Per Distance = {:.2f} J/m".format(energy_per_distance))
print("Max Power = {:.2f} W".format(max_power))
print("Distance Traveled = {:.2f} miles".format(distance_traveled/1609))
print("Completion Time = {:.2f} min".format(completion_time/60))
print("Max Velocity = {:.2f} mph".format(max_velocity*2.237))
#print("Velocity = {}".format(velocity))
#print("Power = {}".format(power))

