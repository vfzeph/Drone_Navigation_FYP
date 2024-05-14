import airsim
import tkinter as tk
from threading import Thread
import time

class DroneTelemetryGUI:
    def __init__(self, airsim_client):
        self.client = airsim_client
        self.root = tk.Tk()
        self.root.title("Drone Telemetry")
        self.telemetry_label = tk.Label(self.root, text="Initializing...")
        self.telemetry_label.pack()

        # Manual Control Buttons
        self.manual_up_button = tk.Button(self.root, text="Up", command=self.manual_up)
        self.manual_up_button.pack(side=tk.LEFT, padx=5)
        self.manual_down_button = tk.Button(self.root, text="Down", command=self.manual_down)
        self.manual_down_button.pack(side=tk.LEFT, padx=5)
        self.manual_forward_button = tk.Button(self.root, text="Forward", command=self.manual_forward)
        self.manual_forward_button.pack(side=tk.LEFT, padx=5)
        self.manual_backward_button = tk.Button(self.root, text="Backward", command=self.manual_backward)
        self.manual_backward_button.pack(side=tk.LEFT, padx=5)

        # Start updating telemetry
        self.update_telemetry()

    def manual_up(self):
        altitude_increment = 1  # Adjust as needed
        current_altitude = self.client.getMultirotorState().kinematics_estimated.position.z_val
        new_altitude = current_altitude + altitude_increment
        self.client.moveToZAsync(new_altitude, 1).join()

    def manual_down(self):
        altitude_decrement = 1  # Adjust as needed
        current_altitude = self.client.getMultirotorState().kinematics_estimated.position.z_val
        new_altitude = current_altitude - altitude_decrement
        self.client.moveToZAsync(new_altitude, 1).join()

    def manual_forward(self):
        velocity = 2  # Adjust as needed
        self.client.moveByVelocityAsync(velocity, 0, 0, 1).join()

    def manual_backward(self):
        velocity = -2  # Adjust as needed
        self.client.moveByVelocityAsync(velocity, 0, 0, 1).join()

    def update_telemetry(self):
        while True:
            pos = self.client.getMultirotorState().kinematics_estimated.position
            text = f"Position: x={pos.x_val:.2f}, y={pos.y_val:.2f}, z={pos.z_val:.2f}"
            self.telemetry_label.config(text=text)
            time.sleep(0.1)  # Update rate in seconds

    def run(self):
        Thread(target=self.update_telemetry).start()
        self.root.mainloop()

class DroneController:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

    def basic_movement(self):
        print("Taking off")
        self.client.takeoffAsync().join()
        print("Moving up")
        self.client.moveToZAsync(-10, 5).join()  # Negative Z is upwards
        print("Moving forward")
        self.client.moveByVelocityAsync(5, 0, 0, 5).join()  # X direction
        print("Landing")
        self.client.landAsync().join()

if __name__ == "__main__":
    drone_controller = DroneController()
    gui = DroneTelemetryGUI(drone_controller.client)
    Thread(target=gui.run).start()
    drone_controller.basic_movement()
