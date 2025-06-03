import random
import xpc
from datetime import datetime
import time
import csv

AUTOPILOT_ALTITUDE_HOLD_MODE = 6

class FlightParameters:
    def __init__(self):
        # preflight parameters
        self.runway_heading = 0.0 # degrees
        POSI = [32.73706817626953, -117.20414733886719, 0, 0, 0, self.runway_heading, 1]

        # Flight parameters
        self.flight_time = 300.0 # seconds
        self.max_target_altitude = 350.0
        self.min_target_altitude = 300.0 # feet
        self.cruise_speed = 120.0 # knots
        self.descent_speed = -800.0 # feet per minute
        self.phase_start_time = 0.0
        self.flight_start_time = 0.0
        self.last_record_time = 0.0
        self.landing_altitude = 300.0
        
        self.time_since_climb = 0.0 
        self.CLIMB_CRUISE_TRANSITION_TIME  = 10.0

        self.failure_time = random.randint(30, int(self.flight_time) - 40)
        self.failure_idx = random.randint(1, 2) # 1 or 2, for engine failure 1 or 2
        
        # For turning
        self.is_turning = False
        self.target_heading = self.runway_heading
        self.turn_start_time = 0.0 
        self.turn_direction = 0.0 # -1 for left, 1 for right  

        self.MIN_TIME_BETWEEN_TURNS = 30.0
        self.MAX_TIME_BETWEEN_TURNS = 120.0
        self.MIN_TURN_DURATION = 5.0
        self.MAX_TURN_DURATION = 20.0 
        self.TURN_ANGLE = 15.0
        self.TURN_RUDDER_FACTOR = 0.05

        self.time_since_last_turn = 0.0 
        self.next_turn_interval = random.uniform(self.MIN_TIME_BETWEEN_TURNS, self.MAX_TIME_BETWEEN_TURNS ) # Initial interval for first turn decision



        # drefs to get
        self.get_drefs_dict = {
            "airspeed": "sim/flightmodel/position/indicated_airspeed",
            "altitude": "sim/flightmodel/position/y_agl",
            "vertical_speed": "sim/flightmodel/position/vh_ind_fpm",
            # "engine_torque": "sim/flightmodel/engine/ENGN_trq",
            "engine_torque": "sim/cockpit2/engine/indicators/torque_n_mtr",
            "n1_fan_speed": "sim/flightmodel/engine/ENGN_N1_",
            "pitch": "sim/flightmodel/position/theta",
            "roll": "sim/flightmodel/position/phi",
            "yaw": "sim/flightmodel/position/beta",
            "heading": "sim/flightmodel/position/psi",  # True heading
            "flap_position": "sim/cockpit2/controls/flap_handle_deploy_ratio",
            "throttle_ratio": "sim/flightmodel/engine/ENGN_thro",
            "engine_failure_1": "sim/operation/failures/rel_engfai0",
            "engine_failure_2": "sim/operation/failures/rel_engfai1",
            "flight_id": "sim/cockpit2/radios/actuators/flight_id",
        }
        self.get_drefs_lookup = {v: k for k, v in self.get_drefs_dict.items()}
        self.get_drefs = self.get_drefs_dict.values()
        # drefs to set
        self.set_drefs_dict = {
            "parkbrake": "sim/flightmodel/controls/parkbrake",
            "flaprqst": "sim/flightmodel/controls/flaprqst",
            "gear_handle_status": "sim/cockpit/switches/gear_handle_status",
            "ENGN_mixt": "sim/flightmodel/engine/ENGN_mixt",
            "autopilot_master_on": "sim/cockpit2/autopilot/autopilot_on",
            "autopilot_engagement_mode": "sim/cockpit/autopilot/autopilot_mode",
            "flight_director_on": "sim/cockpit2/autopilot/flight_director_on",
            "autopilot_mode": "sim/cockpit2/autopilot/altitude_mode",
            "autopilot_altitude": "sim/cockpit/autopilot/altitude",
            "autopilot_altitude_hold_on": "sim/cockpit2/autopilot/altitude_hold_armed",
            "engine_failure_1": "sim/operation/failures/rel_engfai0",
            "engine_failure_2": "sim/operation/failures/rel_engfai1",
            "joystick_pitch": "sim/joystick/FC_ptch",
        }
        self.set_drefs_lookup = {v: k for k, v in self.set_drefs_dict.items()}
        self.set_drefs = self.set_drefs_dict.values()

       


    

class AutomatedFlight:
    def __init__(self, flight_params: FlightParameters = None, xpHost='127.0.0.1', xpPort=49009, timeout=1000.0, output_filename : str = "flight_data.csv", recording_interval : float = 1.0, fail = False) :
        """
        Args:
            xpHost: The hostname of the machine running X-Plane
            xpPort: The port on which the XPC plugin is listening
            timeout: The timeout for the connection in milliseconds
            recording_interval: The interval in seconds for recording flight data
        """
                
        # Flight phases
        self.PHASE_PREFLIGHT = 0
        self.PHASE_TAKEOFF = 1
        self.PHASE_CLIMB = 2
        self.PHASE_CRUISE = 3
        self.PHASE_DESCENT = 4
        self.PHASE_LANDING = 5
        self.PHASE_TAXI = 6
        self.flying = True
        self.fail = fail
        self.failed = False
        self.flight_id = ""

        # Initialize flight parameters
        if flight_params is None:
            flight_params = FlightParameters()

        # Automatically assign all fields from flight_params
        for key, value in vars(flight_params).items():
            print(key, value)
            setattr(self, key, value)

        self.recording_interval = recording_interval


    def init_csv(self):
        """
        Args:
            filename: The name of the CSV file to save data
        """
        headers = ["time", "PHASE"] + list(self.get_drefs_dict.keys())
        with open(self.csv_filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            print(f"CSV file {self.csv_filename} initialized with headers: {headers}")
        self.f = open(self.csv_filename, 'a', newline='')
        self.writer = csv.writer(self.f)


    def record_flight_data(self):
        """Record the current flight data to CSV"""
        current_time = time.time()
        
        # Only record at specified intervals
        if current_time - self.last_record_time < self.recording_interval:
            return
            
        self.last_record_time = current_time
        
        try:
            # Get all DREFs
            values = []
            for dref in self.get_drefs:
                try:
                    # Each DREF might return an array, we take the first value
                    value = self.xpc.getDREF(dref)[0]
                    values.append(value)
                except Exception as e:
                    print(f"Error getting DREF {dref}: {e}")
                    values.append(None)
            
            # Prepare row with timestamp and phase
            timestamp = current_time - self.flight_start_time
            row = [timestamp, self.current_phase] + values
            
            # Print the row for debugging
            print(f"\nRecording flight data: PHASE: {self.current_phase} time: {timestamp:.2f}")
            for i, dref in enumerate(self.get_drefs):
                print(f"{self.get_drefs_lookup[dref]}: {values[i]}")
            
            
            # Write to CSV
            self.writer.writerow(row)
                
        except Exception as e:
            print(f"Error recording flight data: {e}")
    
    def run_flight(self):
        """Execute a complete automated flight."""
        print("Connecting to X-Plane 11...")
        
        try:
            self.current_phase = self.PHASE_PREFLIGHT
            self.xpc = xpc.XPlaneConnect()

            # Test connection. Make sure the flight id is different. If it is the same, then we landed/crashed, but XPlane hasn't restarted the flight yet.
            try:
                flight_id = self.xpc.getDREF(self.get_drefs_dict["flight_id"])[0]
                print(f"Connected to X-Plane. Current flight_id: {flight_id}")
                if flight_id != self.flight_id:
                    self.flight_id = flight_id
                    print(f"Flight ID updated to: {self.flight_id}")
                else:
                    print("...")
                    time.sleep(5)
                    return
            except:
                print("...")
                time.sleep(5)
                return
            
            # Start flight sequence
            self.flying = True
            self.flight_start_time = time.time()
            self.phase_start_time = self.flight_start_time
            self.last_record_time = self.flight_start_time
            
            print("Starting automated flight...")
        
            cur_longitude = self.xpc.getDREF("sim/flightmodel/position/longitude")[0]
            cur_latitude = self.xpc.getDREF("sim/flightmodel/position/latitude")[0]
            cur_heading = self.xpc.getDREF("sim/flightmodel/position/psi")[0]  # True heading
            self.runway_heading = cur_heading
            self.target_heading = cur_heading
            print(f"Preflight position: {cur_latitude}, {cur_longitude}, {cur_heading}")

            while self.flying:
                elapsed_time = time.time() - self.flight_start_time
               
                # Record flight data
                self.record_flight_data()

                if self.fail and elapsed_time >= self.failure_time and not self.failed:
                    print("\nIN\n")
                    self.failed = True
                    self.xpc.sendDREF(self.set_drefs_dict[f"engine_failure_{self.failure_idx}"], 6.0)  # Simulate engine failure

                
                # Check if flight time has exceeded the limit
                if elapsed_time >= self.flight_time:
                    print(f"Flight time of {self.flight_time} seconds reached.")
                    if self.current_phase < self.PHASE_DESCENT:
                        print("Starting descent for landing...")
                        self.current_phase = self.PHASE_DESCENT
                        self.phase_start_time = time.time()
               
                # Execute current flight phase
                self.execute_flight_phase()
                # Small sleep to prevent overwhelming X-Plane with commands
                time.sleep(0.3)
               
                # Check if flight is complete
                if self.current_phase == self.PHASE_TAXI and (time.time() - self.phase_start_time) > 5.0:
                    print("Flight completed successfully.")
                    break

                    
        except Exception as e:
            print(f"Error during flight: {e}")
        finally:
            print("Flight automation finished.")
            print(f"Flight data saved to {self.csv_filename}")
    
    def execute_flight_phase(self):
        """Execute the current flight phase."""
        if self.current_phase == self.PHASE_PREFLIGHT:
            self.execute_preflight()
        elif self.current_phase == self.PHASE_TAKEOFF:
            self.execute_takeoff()
        elif self.current_phase == self.PHASE_CLIMB:
            self.execute_climb()
        elif self.current_phase == self.PHASE_CRUISE:
            self.execute_cruise()
        elif self.current_phase == self.PHASE_DESCENT:
            self.execute_descent()
        elif self.current_phase == self.PHASE_LANDING:
            self.execute_landing()
        elif self.current_phase == self.PHASE_TAXI:
            self.execute_taxi()
    
    def execute_preflight(self):
        """Set up the aircraft for takeoff and data recording file"""
        self.csv_filename = f"xplane_flight_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.init_csv()

        print(f"Recording flight data every {self.recording_interval} seconds to {self.csv_filename}")

        
        print("Executing preflight...")

        
        # Set aircraft position on runway using POSI command
        # Parameters: [lat, lon, alt, pitch, roll, heading, gear]
        #print("Setting aircraft position...")
        #self.xpc.sendPOSI([32.73706817626953, -117.20414733886719, 0.0, 0.0, 0.0, 10.0, 1.0])
        
        # Disable all autopilot
        print("Disabling all autopilots")
        self.xpc.sendDREF(self.set_drefs_dict["autopilot_master_on"], 0)
        self.xpc.sendDREF(self.set_drefs_dict["autopilot_engagement_mode"], 0)
        self.xpc.sendDREF(self.set_drefs_dict["flight_director_on"], 0)

        # Set parking brakes
        print("Setting parking brakes to 1.0 ...")
        self.xpc.sendDREF(self.set_drefs_dict["parkbrake"], float(1.0))
        
        # Set flaps for takeoff
        print("Setting flaps for takeoff...")
        self.xpc.sendDREF(self.set_drefs_dict["flaprqst"], float(0.12))  # 15% flaps
        self.flaps = 0.12

        # Make sure gear is down
        print("Setting gear down...")
        self.xpc.sendDREF(self.set_drefs_dict["gear_handle_status"], float(1.0))
        
        # Set engine mixture 80to full rich
        print("Setting engine mixture to full rich...")
        self.xpc.sendDREF(self.set_drefs_dict["ENGN_mixt"], 1.0)
        


        self.xpc.sendCTRL([0.0, 0.0, 0.0 , 1.0, 1.0])
        self.throttle = 1.0
       
        # Advance to takeoff phase
        self.current_phase = self.PHASE_TAKEOFF
        self.phase_start_time = time.time()
        print("Preflight complete. Ready for takeoff.")
    
    def execute_takeoff(self):
        """Execute the takeoff phase."""
        # Get current values
        altitude = self.xpc.getDREF(self.get_drefs_dict["altitude"])[0]
        speed = self.xpc.getDREF(self.get_drefs_dict["airspeed"])[0]
        heading = self.xpc.getDREF(self.get_drefs_dict["heading"])[0] 
        roll = self.xpc.getDREF(self.get_drefs_dict["roll"])[0]  # Current roll angle
        pitch_angle = self.xpc.getDREF(self.get_drefs_dict["pitch"])[0]  # Current pitch angle
        yaw = self.xpc.getDREF(self.get_drefs_dict["yaw"])[0]
        elevator_input = 0.0
        aileron_input = 0.0
        rudder_input = 0.0

        self.xpc.sendDREF(self.set_drefs_dict["parkbrake"], float(0.0))

        if speed < 130 and altitude < 50:
            elevator_input = 0.0
            
        elif speed > 130 and altitude < 50:
            elevator_input = 0.05
        
        else:
            # Transition to climb phase
            print(f"Takeoff complete at altitude: {altitude:.2f} feet")
            self.current_phase = self.PHASE_CLIMB
            self.phase_start_time = time.time()
            pitch_target = 10.0 # degrees, adjust for a comfortable climb attitude
            pitch_error = pitch_target - pitch_angle
            elevator_input = max(min(pitch_error * 0.03, 0.1), -0.1)

            
        
        # Apply small rudder correction if needed to maintain runway heading
        # heading = self.xpc.getDREF(self.get_drefs_dict["heading"])[0]
        # heading_error = (self.target_heading - heading + 180) % 360 - 180
            
        
        # rudder_input = max(min(heading_error * 0.05, 0.15), -0.15)

        if altitude > 3:
            aileron_input = max(min(-roll * 0.08, 0.1), -0.1)  # Counters current roll
            rudder_input = max(min(-yaw * 0.05, 0.15), -0.15)
                
        self.xpc.sendCTRL([elevator_input, aileron_input, rudder_input, 1.0, 1.0])

    def execute_climb(self):
        try: 
            """Execute the climb phase."""
            # Get current values
            altitude = self.xpc.getDREF(self.get_drefs_dict["altitude"])[0]
            speed = self.xpc.getDREF(self.get_drefs_dict["airspeed"])[0]
            heading = self.xpc.getDREF(self.get_drefs_dict["heading"])[0] 
            roll = self.xpc.getDREF(self.get_drefs_dict["roll"])[0]  # Current roll angle
            pitch_angle = self.xpc.getDREF(self.get_drefs_dict["pitch"])[0]  # Current pitch angle
            yaw = self.xpc.getDREF(self.get_drefs_dict["yaw"])[0]

            elevator_input = 0.2
            aileron_input = 0.0
            rudder_input = 0.0
            
            # Gradually retract flaps as speed increases
            if speed > 110:
                self.flaps *= 0.5
                self.xpc.sendDREF(self.set_drefs_dict["flaprqst"], self.flaps)
            

            # Simple proportional control for pitch
            pitch_target = 10.0 # degrees
            pitch_error = pitch_target - pitch_angle
            elevator_input = max(min(pitch_error * 0.03, 0.1), -0.1)

            # Calculate heading error                   
            # Counters current yaw
            rudder_input = max(min(-yaw * 0.05, 0.15), -0.15)
            # rudder_input = max(min(heading_error * 0.05, 0.15), -0.15)
            
            # Counters current roll
            aileron_input = max(min(-roll * 0.08, 0.1), -0.1)  # Counters current roll


            # Send control commands with corrections
            print(f"Elevator: {elevator_input:.2f}, Aileron: {aileron_input:.3f}, Rudder: {rudder_input:.3f}")
            self.xpc.sendCTRL([elevator_input, aileron_input, rudder_input, 1.0, 1.0])


            # Transition to cruise when reaching cruise altitude
            if altitude >= self.min_target_altitude:
                print(f"Reached cruise altitude: {altitude:.2f} feet")
                self.current_phase = self.PHASE_CRUISE
                self.phase_start_time = time.time()
                self.time_since_last_turn = self.phase_start_time
                self.time_since_climb = self.phase_start_time
                self.prev_elevator_input = elevator_input
        except Exception as e:
            print(f"Error during climb phase: {e}")
            self.flying = False

    
    
                    
    def execute_cruise(self):
        """Execute the cruise phase."""
        # Get current values
        altitude = self.xpc.getDREF(self.get_drefs_dict["altitude"])[0]
        speed = self.xpc.getDREF(self.get_drefs_dict["airspeed"])[0]
        # heading = self.xpc.getDREF(self.get_drefs_dict["heading"])[0]
        roll = self.xpc.getDREF(self.get_drefs_dict["roll"])[0]
        pitch_angle = self.xpc.getDREF(self.get_drefs_dict["pitch"])[0]  # Current pitch angle
        yaw = self.xpc.getDREF(self.get_drefs_dict["yaw"])[0]
        elevator_input = 0.0
        aileron_input = 0.0
        rudder_input = 0.0
        
        current_time = time.time() 

        
        # Adjust throttle to maintain cruise speed
        self.throttle = 0.9

        # Maintain level flight with slight pitch adjustment for altitude
        # when first transitioned from c1 / bank_errorlimb phase, we want to slowly smooth the pitch angle down to 0 
        if current_time - self.time_since_climb < self.CLIMB_CRUISE_TRANSITION_TIME:
            pitch_error = 2.0 - pitch_angle # will be negative
            tmp = max(self.prev_elevator_input -  0.005, pitch_error * 0.005)
            elevator_input = max(min(tmp, 0.1), -0.1)
            self.prev_elevator_input = elevator_input 
        else:
            pitch_error = 2.0 - pitch_angle 
            elevator_input = max(min(pitch_error * 0.05, 0.1), -0.1)

        rudder_input = max(min(-yaw * 0.05, 0.15), -0.15)

        # Check if it's time to try to turn
        if not self.is_turning: 
            # Check if it's time to initiate a new turn
            if current_time - self.time_since_last_turn > self.next_turn_interval:
                self.is_turning = True
                self.turn_start_time = current_time
                self.current_turn_duration = random.uniform(self.MIN_TURN_DURATION, self.MAX_TURN_DURATION)
                self.turn_direction = random.choice([-1, 1]) # -1 for left, 1 for right
                print(f"Starting {'left' if self.turn_direction == -1 else 'right'} turn for {self.current_turn_duration:.1f} seconds.")
           

        if self.is_turning: 
            if current_time - self.turn_start_time < self.current_turn_duration: 
                bank_error = self.TURN_ANGLE - abs(roll)
                tmp = min(abs(self.prev_aileron_input) + 0.02, abs(bank_error) * 0.005) * self.turn_direction
                aileron_input = max(min(tmp , 0.1), -0.1)
                
                # rudder in 
                #rudder_input = self.turn_direction * self.TURN_RUDDER_FACTOR * abs(roll/self.TURN_ANGLE) if abs(roll) > 2 else 0
            else: # turn completed 
                self.is_turning = False 
                self.target_heading = self.xpc.getDREF(self.get_drefs_dict["heading"])[0]
                self.time_since_last_turn = current_time 
                self.next_turn_interval = random.uniform(self.MIN_TIME_BETWEEN_TURNS, self.MAX_TIME_BETWEEN_TURNS)
                # Maintain straight flight
                
                # Calculate roll correction
                aileron_input = -roll * 0.005  # Counters current roll
                # Maintain heading using simple proportional control
                # heading_error = (self.target_heading - heading + 180) % 360 - 180                                    
                # Calculate rudder correction
                # rudder_input = max(min(* 0.05, 0.15), -0.15)

                print(f"\n[*] Turn completed. New target heading: {self.target_heading:.2f}")
        else: # maintain straight flight
            # Calculate roll correction
            if current_time - self.time_since_last_turn < 8.0:      
                bank_error = 0 - abs(roll)
                tmp = min(abs(self.prev_aileron_input) + 0.02, abs(bank_error) * 0.005) * -self.turn_direction
                aileron_input = max(min(tmp , 0.1), -0.1)
            else:
                aileron_input = max(min(-roll * 0.08, 0.1), -0.1)  # Counters current roll
            # Maintain heading using simple proportional control
            # heading_error = (self.target_heading - heading + 180) % 360 - 180
            # rudder_input = max(min(heading_error * 0.05, 0.15), -0.15)
        
        # Send control commands
        
        print(f"Elevator: {elevator_input:2f}, Aileron: {aileron_input:.3f}, Rudder: {rudder_input:.3f}")
        self.prev_aileron_input = aileron_input
        self.xpc.sendCTRL([elevator_input, aileron_input, rudder_input, self.throttle, 1.0])
 
        # Calculate time spent in cruise
        total_elapsed = current_time - self.flight_start_time

       
        # Check if it's time to start descent
        remaining_time = self.flight_time - total_elapsed
        estimated_descent_time = (altitude / abs(self.descent_speed)) * 60  # in seconds
        
        if remaining_time <= estimated_descent_time + 20:  # Add 20 seconds buffer
            print("Beginning descent phase...")
            self.current_phase = self.PHASE_DESCENT
            self.phase_start_time = time.time()
        
            # Disengage autopilot
            # self.xpc.sendDREF(self.set_drefs_dict["autopilot_altitude_hold_on"], 0)
    
    def execute_descent(self):
        """Execute the descent phase."""
        # Get current values
        altitude = self.xpc.getDREF(self.get_drefs_dict["altitude"])[0]
        speed = self.xpc.getDREF(self.get_drefs_dict["airspeed"])[0]
        heading = self.xpc.getDREF("sim/flightmodel/position/psi")[0]
        roll = self.xpc.getDREF(self.get_drefs_dict["roll"])[0]
        pitch_angle = self.xpc.getDREF(self.get_drefs_dict["pitch"])[0]  # Current pitch angle
        
        yaw = self.xpc.getDREF(self.get_drefs_dict["yaw"])[0]
       
        # Calculate roll correction
        aileron_input = max(min(-roll * 0.08, 0.1), -0.1)  # Counters current roll
        
        # Maintain heading using simple proportional control
        heading_error = (self.target_heading - heading + 180) % 360 - 180

        # Calculate rudder correction
        # rudder_input = max(min(heading_error * 0.05, 0.15), -0.15)
        rudder_input = max(min(-yaw * 0.05, 0.15), -0.15)

        # Simple proportional control for pitch
        pitch_target = -7.0 # degrees, adjust for a comfortable climb attitude
        pitch_error = pitch_target - pitch_angle
        elevator_input = max(min(pitch_error * 0.05, 0.12), -0.12)

        # Reduce throttle for descent
        self.throttle = max(self.throttle * 0.97, 0.3)
        self.xpc.sendCTRL([elevator_input, aileron_input, rudder_input, self.throttle, 1.0])
    
        # Add flaps below certain speed and altitude thresholds
        if altitude < 400:
            self.xpc.sendDREF(self.set_drefs_dict["flaprqst"], 0.5)  # 50% flaps
            
        # Transition to landing phase at appropriate altitude
        if altitude < 200:
            print("Transitioning to landing phase...")
            self.current_phase = self.PHASE_LANDING
            self.phase_start_time = time.time()
            
            # Ensure gear is down for landing
            self.xpc.sendDREF(self.set_drefs_dict["gear_handle_status"], 1.0)
            self.xpc.sendCTRL([elevator_input, aileron_input, rudder_input, self.throttle, 1.0]) 
    
    def execute_landing(self):
        """Execute the landing phase."""
        # Get current values
        altitude = self.xpc.getDREF(self.get_drefs_dict["altitude"])[0]
        speed = self.xpc.getDREF(self.get_drefs_dict["airspeed"])[0]
        vertical_speed = self.xpc.getDREF(self.get_drefs_dict["vertical_speed"])[0]
        roll = self.xpc.getDREF(self.get_drefs_dict["roll"])[0]
        pitch_angle = self.xpc.getDREF(self.get_drefs_dict["pitch"])[0]  # Current pitch angle
        yaw = self.xpc.getDREF(self.get_drefs_dict["yaw"])[0]
        # Calculate roll correction
        aileron_input = max(min(-roll * 0.08, 0.1), -0.1)  # Counters current roll
        
        # Counteract current yaw 
        rudder_input = max(min(-yaw * 0.05, 0.15), -0.15)

        # Full flaps for landing
        self.xpc.sendDREF(self.set_drefs_dict["flaprqst"], 1.0)
        
        # Simple proportional control for pitch
        pitch_target = -5.0 # degrees, adjust for a comfortable descent attitude
        pitch_error = pitch_target - pitch_angle
        elevator_input = max(min(pitch_error * 0.05, 0.1), -0.1)

        
        # Reduce throttle as we get closer to the ground
        throttle = max(0.2 - (self.landing_altitude - altitude) / 2500, 0.05) if altitude < 300 else 0.2
        
        # Send control commands
        self.xpc.sendCTRL([elevator_input, aileron_input, rudder_input, throttle, 1.0])
        
        # Check for touchdown
        if altitude < 10: 
            print("Touchdown!")
            self.flying = False 
            self.current_phase = self.PHASE_TAXI
            self.phase_start_time = time.time()
            
            # Cut throttle, apply brakes
            self.xpc.sendCTRL([0.0, 0.0, 0.0, 0.0, 1.0])
            self.xpc.sendDREF(self.set_drefs_dict["parkbrake"], 1.0)
    
    def execute_taxi(self):
        """Execute post-landing taxi phase."""
        # Simply keep throttle at zero and brakes applied
        self.xpc.sendCTRL([0.0, 0.0, 0.0, 0.0, 1.0])
        self.xpc.sendDREF("sim/flightmodel/controls/parkbrake", 1.0)
        
        # Print final status
        if int(time.time() - self.phase_start_time) == 3:
            total_flight_time = time.time() - self.flight_start_time
            print(f"Flight completed in {total_flight_time:.1f} seconds.")

if __name__ == "__main__":
    print("Starting X-Plane 11 Flight Automation with Data Recording")
    print("Using NASA's XPlaneConnect client")
    
    # Create and run the automated flight
    flight = AutomatedFlight()
    while True:
        flight.run_flight()
    
