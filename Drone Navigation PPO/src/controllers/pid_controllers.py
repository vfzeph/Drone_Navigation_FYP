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
