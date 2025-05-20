import xpc
from datetime import datetime
import time
import csv

class FlightParameters:
    def __init__(self):
        # Flight parameters
        self.flight_time = 300.0 # seconds
        self.target_altitude = 10000.0 # feet
        self.cruise_speed = 200.0 # knots
        self.climb_speed = 4000.0 # feet per minute
        self.descent_speed = -800.0 # feet per minute
        self.landing_speed = -400.0 # feet per minute
        self.phase_start_time = 0.0
        self.flight_start_time = 0.0
        self.last_record_time = 0.0

        # drefs to get
        self.get_drefs_dict = {
            "airspeed": "sim/flightmodel/position/indicated_airspeed",
            "altitude": "sim/flightmodel/position/elevation",
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
            "failure_engine_1": "sim/operation/failures/rel_engfai0",
            "failure_engine_2": "sim/operation/failures/rel_engfai1"
        }
        self.get_drefs_lookup = {v: k for k, v in self.get_drefs_dict.items()}
        self.get_drefs = self.get_drefs_dict.values()

        # drefs to set
        self.set_drefs_dict = {
            "parkbrake": "sim/flightmodel/controls/parkbrake",
            "flaprqst": "sim/flightmodel/controls/flaprqst",
            "gear_handle_status": "sim/cockpit/switches/gear_handle_status",
            "ENGN_mixt": "sim/flightmodel/engine/ENGN_mixt",
            "autopilot_mode": "sim/cockpit/autopilot/autopilot_mode",
            "autopilot_altitude": "sim/cockpit/autopilot/altitude",
            "autopilot_altitude_hold_on": "sim/cockpit/autopilot/altitude_hold_armed",
        }
        self.set_drefs_lookup = {v: k for k, v in self.set_drefs_dict.items()}
        self.set_drefs = self.set_drefs_dict.values()

        # preflight parameters
        # self.runway_heading = 106.0445785522461270 # degrees
        self.runway_heading = 0.0
        POSI = [32.73706817626953, -117.20414733886719, 0, 0, 0, self.runway_heading, 1]



    

class AutomatedFlight:
    def __init__(self, flight_params: FlightParameters = None, xpHost='127.0.0.1', xpPort=49009, timeout=1000.0, output_filename : str = "flight_data.csv", recording_interval : float = 0.5) :
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

        self.current_phase = self.PHASE_PREFLIGHT

        # Initialize flight parameters
        if flight_params is None:
            flight_params = FlightParameters()

        # Automatically assign all fields from flight_params
        for key, value in vars(flight_params).items():
            print(key, value)
            setattr(self, key, value)

        self.recording_interval = recording_interval
        self.xpc = xpc.XPlaneConnect()
        self.csv_filename = f"xplane_flight_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.init_csv()

    def init_csv(self):
        """
        Args:
            filename: The name of the CSV file to save data
        """
        headers = self.get_drefs_dict.keys()
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
                    value = self.xpc.getDREF(dref)
                    values.append(value)
                except Exception as e:
                    print(f"Error getting DREF {dref}: {e}")
                    values.append(None)
            
            # Prepare row with timestamp and phase
            timestamp = current_time - self.flight_start_time
            row = [timestamp, self.current_phase] + values
            
            # Print the row for debugging
            print(f"\nRecording flight data: time: {timestamp:.2f}")
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
            # Test connection
            try:
                altitude = self.xpc.getDREF("sim/flightmodel/position/elevation")[0]
                print(f"Connected to X-Plane. Current altitude: {altitude:.2f} feet")
            except:
                print("Could not connect to X-Plane. Make sure X-Plane is running with the XPlaneConnect plugin.")
                return
            
            # Start flight sequence
            self.flight_start_time = time.time()
            self.phase_start_time = self.flight_start_time
            self.last_record_time = self.flight_start_time
            
            print("Starting automated 10-minute flight...")
            print(f"Recording flight data every {self.recording_interval} seconds to {self.csv_filename}")
        
            cur_longitude = self.xpc.getDREF("sim/flightmodel/position/longitude")[0]
            cur_latitude = self.xpc.getDREF("sim/flightmodel/position/latitude")[0]
            cur_heading = self.xpc.getDREF("sim/flightmodel/position/beta")[0]
            print(f"Preflight position: {cur_latitude}, {cur_longitude}, {cur_heading}")

            while True:
                elapsed_time = time.time() - self.flight_start_time
               
                # Record flight data
                self.record_flight_data()
               
                # Check if flight time has exceeded the limit
                if elapsed_time >= self.flight_time:
                    print(f"Flight time of {self.flight_time} seconds reached.")
                    if self.current_phase < self.PHASE_DESCENT:
                        print("Starting descent for landing...")
                        self.current_phase = self.PHASE_DESCENT
                        self.phase_start_time = time.time 
               
                # Execute current flight phase
                self.execute_flight_phase()
               
                # Small sleep to prevent overwhelming X-Plane with commands
                time.sleep(0.1)
               
                # Check if flight is complete
                if self.current_phase == self.PHASE_TAXI and (time.time() - self.phase_start_time) > 5:
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
        """Set up the aircraft for takeoff."""
        print("Executing preflight checks...")
        
        # Set aircraft position on runway using POSI command
        # Parameters: [lat, lon, alt, pitch, roll, heading, gear]
        #print("Setting aircraft position...")
        #self.xpc.sendPOSI([32.73706817626953, -117.20414733886719, 0.0, 0.0, 0.0, 10.0, 1.0])
        
        # Set parking brakes
        print("Setting parking brakes to 1.0 ...")
        self.xpc.sendDREF(self.set_drefs_dict["parkbrake"], float(1.0))
        
        # Set flaps for takeoff
        print("Setting flaps for takeoff...")
        self.xpc.sendDREF(self.set_drefs_dict["flaprqst"], float(0.3))  # 30% flaps
        
        # Make sure gear is down
        print("Setting gear down...")
        self.xpc.sendDREF(self.set_drefs_dict["gear_handle_status"], float(1.0))
        
        # Set engine mixture to full rich
        print("Setting engine mixture to full rich...")
        self.xpc.sendDREF(self.set_drefs_dict["ENGN_mixt"], 1.0)
        
        # Advance to takeoff phase
        self.current_phase = self.PHASE_TAKEOFF
        self.phase_start_time = time.time()
        print("Preflight complete. Ready for takeoff.")
    
    def execute_takeoff(self):
        """Execute the takeoff phase."""
        # Get current values
        altitude = self.xpc.getDREF(self.get_drefs_dict["altitude"])[0]
        speed = self.xpc.getDREF(self.get_drefs_dict["airspeed"])[0]
        heading = self.xpc.getDREF(self.get_drefs_dict["heading"])[0]  # Using proper heading dataref
        roll = self.xpc.getDREF(self.get_drefs_dict["roll"])[0]  # Current roll angle
    
        phase_time = time.time() - self.phase_start_time
        
        if phase_time < 2:
            # Release parking brake
            self.xpc.sendDREF(self.set_drefs_dict["parkbrake"], float(0.0))
            
            # Set throttle to full using CTRL command
            # Parameters: [elevation, roll, rudder, throttle, gear]
            self.xpc.sendCTRL([0.0, 0.0, 0.0, 1.0, 1.0])
            
        elif speed > 80 and altitude < 50:
            # Rotate once we reach rotation speed
            self.xpc.sendCTRL([0.0, 0.0, 0.0, 1.0, 1.0])  # Pull back on the elevator
            
        elif altitude > 100:
            # Transition to climb phase
            print(f"Takeoff complete at altitude: {altitude:.2f} feet")
            self.current_phase = self.PHASE_CLIMB
            self.phase_start_time = time.time()
            
        # Add a small amount of rudder to keep straight during takeoff run
        if speed < 80:
            # Apply small rudder correction if needed to maintain runway heading
            heading = self.xpc.getDREF(self.get_drefs_dict["heading"])[0]
            heading_error = (self.runway_heading - heading) % 360
            if heading_error > 180:
                heading_error -= 360
                
            rudder_correction = max(min(heading_error * 0.01, 0.1), -0.1)
            roll_correction = -roll * 0.05
                    
            print(f"Heading error: {heading_error:.2f}°, Roll: {roll:.2f}°, Aileron: {roll_correction:.3f}, Rudder: {rudder_correction:.3f}")

            self.xpc.sendCTRL([0.0, roll_correction, rudder_correction, 1.0, 1.0])
    
    def execute_climb(self):
        """Execute the climb phase."""
        # Get current values
        altitude = self.xpc.getDREF(self.get_drefs_dict["altitude"])[0]
        speed = self.xpc.getDREF(self.get_drefs_dict["airspeed"])[0]
        vertical_speed = self.xpc.getDREF(self.get_drefs_dict["vertical_speed"])[0]
        heading = self.xpc.getDREF(self.get_drefs_dict["heading"])[0]
        roll = self.xpc.getDREF(self.get_drefs_dict["roll"])[0]
    
        
        # Gradually retract flaps as speed increases
        if speed > 100:
            self.xpc.sendDREF(self.set_drefs_dict["flaprqst"], 0.0)
        
        # Target pitch to maintain climb speed
        target_vertical_speed = self.climb_speed
        vertical_speed_error = target_vertical_speed - vertical_speed
        
        # Simple proportional control for pitch
        pitch_adjustment = vertical_speed_error * 0.0005
        pitch_command = max(min(0.2 + pitch_adjustment, 0.4), 0.0)  # Limit elevator command
        
        # Calculate heading error
        heading_error = (self.runway_heading - heading) % 360
        if heading_error > 180:
            heading_error -= 360
        
        # Calculate roll correction
        roll_correction = -roll * 0.05  # Counters current roll

        # Calculate rudder correction
        rudder_correction = max(min(heading_error * 0.005, 0.15), -0.15)
        
        # Send control commands with corrections
        self.xpc.sendCTRL([pitch_command, roll_correction, rudder_correction, 1.0, 1.0])
        
        # Transition to cruise when reaching cruise altitude
        if altitude >= self.target_altitude:
            print(f"Reached cruise altitude: {altitude:.2f} feet")
            self.current_phase = self.PHASE_CRUISE
            self.phase_start_time = time.time()
            
            # Engage altitude hold autopilot
            self.xpc.sendDREF(self.set_drefs_dict["autopilot_altitude"], self.target_altitude)
            self.xpc.sendDREF(self.set_drefs_dict["autopilot_altitude_hold_on"], 1)
            
            # Set cruise throttle
            self.xpc.sendCTRL([0.0, 0.0, 0.0, 0.8, 1.0])  # Reduce throttle for cruise
    
    def execute_cruise(self):
        """Execute the cruise phase."""
        # Get current values
        altitude = self.xpc.getDREF(self.get_drefs_dict["altitude"])[0]
        speed = self.xpc.getDREF(self.get_drefs_dict["airspeed"])[0]
        heading = self.xpc.getDREF(self.get_drefs_dict["heading"])[0]
        roll = self.xpc.getDREF(self.get_drefs_dict["roll"])[0]
        
        # Calculate roll correction
        roll_correction = -roll * 0.05  # Counters current roll
        
        # Calculate time spent in cruise
        cruise_time = time.time() - self.phase_start_time
        total_elapsed = time.time() - self.flight_start_time
        
        # Maintain heading using simple proportional control
        heading_error = (self.runway_heading - heading) % 360
        if heading_error > 180:
            heading_error -= 360

        # Calculate rudder correction
        rudder_correction = max(min(heading_error * 0.005, 0.15), -0.15)

        # Maintain level flight with slight pitch adjustment for altitude
        altitude_error = self.target_altitude - altitude
        pitch_adjustment = altitude_error * 0.0001
        pitch_command = max(min(pitch_adjustment, 0.1), -0.1)

        # Adjust throttle to maintain cruise speed
        throttle = 0.8
        if speed < self.cruise_speed - 5:
            throttle = 0.85
        elif speed > self.cruise_speed + 5:
            throttle = 0.75

        # Send control commands
        self.xpc.sendCTRL([pitch_command, roll_correction, rudder_correction, throttle, 1.0])
        
        # Check if it's time to start descent
        remaining_time = self.flight_time - total_elapsed
        estimated_descent_time = (self.target_altitude / abs(self.descent_speed)) * 60  # in seconds
        
        if remaining_time <= estimated_descent_time + 60:  # Add 60 seconds buffer
            print("Beginning descent phase...")
            self.current_phase = self.PHASE_DESCENT
            self.phase_start_time = time.time()
            
            # Disengage autopilot
            self.xpc.sendDREF(self.set_drefs_dict["autopilot_altitude_hold_on"], 0)
    
    def execute_descent(self):
        """Execute the descent phase."""
        # Get current values
        altitude = self.xpc.getDREF(self.get_drefs_dict["altitude"])[0]
        speed = self.xpc.getDREF(self.get_drefs_dict["airspeed"])[0]
        vertical_speed = self.xpc.getDREF(self.get_drefs_dict["vertical_speed"])[0]
        heading = self.xpc.getDREF("sim/flightmodel/position/psi")[0]
        roll = self.xpc.getDREF(self.get_drefs_dict["roll"])[0]
        
        # Calculate roll correction
        roll_correction = -roll * 0.05  # Counters current roll
        
        # Maintain heading using simple proportional control
        heading_error = (self.runway_heading - heading) % 360
        if heading_error > 180:
            heading_error -= 360

        # Calculate rudder correction
        rudder_correction = max(min(heading_error * 0.005, 0.15), -0.15)

        # Target pitch to maintain descent speed
        target_vertical_speed = self.descent_speed
        vertical_speed_error = target_vertical_speed - vertical_speed
        
        # Simple proportional control for pitch
        pitch_adjustment = vertical_speed_error * 0.0003
        pitch_command = max(min(-0.1 + pitch_adjustment, 0.1), -0.3)  # Limit elevator command
        
        # Reduce throttle for descent
        self.xpc.sendCTRL([pitch_command, roll_correction, rudder_correction, 0.3, 1.0])
        
        # Add flaps below certain speed and altitude thresholds
        if altitude < 1000 and speed < 140:
            self.xpc.sendDREF(self.set_drefs_dict["flaprqst"], 0.5)  # 50% flaps
            
        # Transition to landing phase at appropriate altitude
        if altitude < 500:
            print("Transitioning to landing phase...")
            self.current_phase = self.PHASE_LANDING
            self.phase_start_time = time.time()
            
            # Ensure gear is down for landing
            self.xpc.sendDREF(self.set_drefs_dict["gear_handle_status"], 1.0)
            self.xpc.sendCTRL([pitch_command, 0.0, 0.0, 0.3, 1.0])  # Set gear state in CTRL as well
    
    def execute_landing(self):
        """Execute the landing phase."""
        # Get current values
        altitude = self.xpc.getDREF(self.get_drefs_dict["altitude"])[0]
        speed = self.xpc.getDREF(self.get_drefs_dict["airspeed"])[0]
        vertical_speed = self.xpc.getDREF(self.get_drefs_dict["vertical_speed"])[0]
        
        # Full flaps for landing
        self.xpc.sendDREF(self.set_drefs_dict["flaprqst"], 1.0)
        
        # Target a gentle descent speed for landing
        vertical_speed_error = self.landing_speed - vertical_speed
        
        # More sensitive control for landing
        pitch_adjustment = vertical_speed_error * 0.0005
        pitch_command = max(min(0.1 + pitch_adjustment, 0.3), -0.2)
        
        # Reduce throttle as we get closer to the ground
        throttle = max(0.2 - (500 - altitude) / 2500, 0.05) if altitude < 300 else 0.2
        
        # Send control commands
        self.xpc.sendCTRL([pitch_command, 0.0, 0.0, throttle, 1.0])
        
        # Check for touchdown
        if altitude < 5 and vertical_speed > -100:
            print("Touchdown!")
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
    flight.run_flight()
    
    print(f"Flight data recorded to {flight.csv_filename}")
    print("You can analyze this data using data analysis tools like pandas, matplotlib, etc.")