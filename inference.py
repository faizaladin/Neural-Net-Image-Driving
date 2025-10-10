import sys
import os
import glob
import time
import queue
import numpy as np
from PIL import Image
import cv2
import torch
import subprocess

# --- Find and Add CARLA .egg to Python Path ---
try:
    CARLA_ROOT = "/home/faizaladin/Desktop/carla" # Update this path if needed
    dist_path = os.path.join(CARLA_ROOT, 'PythonAPI/carla/dist')
    egg_files = glob.glob(f'{dist_path}/carla-*-py{sys.version_info.major}.{sys.version_info.minor}-*.egg')
    if not egg_files: raise IndexError
    sys.path.append(egg_files[0])
except IndexError:
    print("❌ Could not find a compatible CARLA .egg file.")
    sys.exit()

import carla
from model import Driving # Assumes your model class is in model.py

# --- Configuration ---
HOST = 'localhost'
PORT = 2000
MODEL_PATH = 'town3.pth'
IMAGE_WIDTH = 400
IMAGE_HEIGHT = 300
STEERING_SCALING_FACTOR = 100.0
DEVICE = torch.device('cpu')
FIXED_DELTA_SECONDS = 0.1
CARLA_EXECUTABLE = os.path.join(CARLA_ROOT, "CarlaUE4.sh")

def restart_simulator():
    """Kills any running CARLA processes and starts a new one in the background."""
    print("Killing existing Carla processes...")
    os.system("pkill -f CarlaUE4-Linux-Shipping")
    time.sleep(5)  # Allow time for system ports to be released

    print("Starting new Carla simulator instance...")
    env = os.environ.copy()
    # This environment variable might be specific to your system for Vulkan rendering
    env["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/nvidia_icd.json"
    
    # Run the simulator in the background and suppress its output
    subprocess.Popen(
        [CARLA_EXECUTABLE, "-RenderOffScreen"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT
    )
    print("Waiting 20 seconds for the simulator to initialize...")
    time.sleep(20)

def preprocess_image(carla_image):
    """
    Preprocesses a CARLA image to the format expected by the model.
    """
    array = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
    array = array.reshape((carla_image.height, carla_image.width, 4))
    array = array[:, :, :3]
    img = Image.fromarray(array)
    img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.LANCZOS)
    img_np = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
    return img_tensor

def main():
    # Initialize all variables to None to prevent UnboundLocalError on cleanup
    client = None
    world = None
    vehicle = None
    camera = None
    original_settings = None

    try:
        # --- 0. Start the Simulator ---
        restart_simulator()

        # --- 1. Load Model ---
        print(f"Loading model from {MODEL_PATH} on device {DEVICE}...")
        model = Driving().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print("✅ Model loaded successfully.")

        # --- 2. Connect to CARLA with Retries ---
        # The retry loop is still useful in case the simulator takes longer than 20s
        for i in range(10): 
            try:
                print(f"Attempting to connect to CARLA (Attempt {i+1}/10)...")
                client = carla.Client(HOST, PORT)
                client.set_timeout(10.0)
                print("Loading Town02...")
                world = client.load_world('Town03')
                print("✅ CARLA connection successful and Town02 loaded.")
                break # Exit the loop on successful connection
            except RuntimeError as e:
                print(f"Connection failed: {e}. Retrying in 5 seconds...")
                time.sleep(5)
        
        if not world:
            raise RuntimeError("❌ Could not connect to CARLA after multiple attempts.")

        # --- 3. Set Synchronous Mode ---
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        world.apply_settings(settings)
        print(f"✅ CARLA set to synchronous mode with dt={FIXED_DELTA_SECONDS}s (10 FPS).")

        weather = carla.WeatherParameters(
            cloudiness=60.0,
            precipitation=30.0,
            precipitation_deposits=20.0,
            wind_intensity=10.0,
            sun_azimuth_angle=0.0,
            sun_altitude_angle=70.0,
            fog_density=10.0,
            fog_distance=100.0,
            wetness=40.0
        )
        #world.set_weather(carla.WeatherParameters.SoftRainNoon)
        
        # --- 4. Spawn Vehicle and Camera ---
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        # Use a later spawn point (e.g., index 10) to start further down the road
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = spawn_points[40] if len(spawn_points) > 10 else spawn_points[-1]
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(IMAGE_WIDTH))
        camera_bp.set_attribute('image_size_y', str(IMAGE_HEIGHT))
        camera_transform = carla.Transform(carla.Location(x=2.0, z=1.4))  # Match collect_data.py
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

        image_queue = queue.Queue()
        camera.listen(image_queue.put)
        
        world.tick() # Initial tick to get the first frame
        print("🚀 Starting inference loop. Press Ctrl+C to exit.")

        # --- 5. Real-time Inference and Control Loop ---
        target_speed = 6.0  # m/s (about 28.8 km/h)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('carla_inference_recording.mp4', fourcc, 10, (IMAGE_WIDTH, IMAGE_HEIGHT))
        try:
            while True:
                world.tick() # Advance the simulation by one step
                image = image_queue.get()
                image_tensor = preprocess_image(image)
                with torch.no_grad():
                    prediction = model(image_tensor)
                predicted_steer = prediction.item()

                # Get current speed
                velocity = vehicle.get_velocity()
                speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                speed_error = target_speed - speed
                # Simple proportional controller for throttle
                throttle = 0.4 + 0.1 * speed_error
                throttle = max(0.0, min(1.0, throttle))

                control = carla.VehicleControl(throttle=throttle, steer=predicted_steer)
                vehicle.apply_control(control)

                # Visualization using OpenCV
                view = np.frombuffer(image.raw_data, dtype=np.uint8)
                view = view.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 4))
                view = view[:, :, :3]
                view = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
                cv2.putText(
                    view, f"Steer: {predicted_steer:.3f}  Speed: {speed:.2f} m/s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                )
                out.write(view)
                cv2.imshow("CARLA Autonomous Drive", view)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            out.release()

    finally:
        # --- 6. Cleanup ---
        print("\nCleaning up actors and restoring settings...")
        if world is not None and original_settings is not None:
            world.apply_settings(original_settings)
        
        # Safely destroy actors if they were created
        if camera is not None:
            camera.destroy()
        if vehicle is not None:
            vehicle.destroy()
        
        cv2.destroyAllWindows()
        print("Cleanup complete.")

if __name__ == "__main__":
    main()

