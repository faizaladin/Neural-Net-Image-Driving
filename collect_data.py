import sys
import os
import glob
import time
import subprocess
import queue
import numpy as np
from PIL import Image

# --- Find and Add CARLA .egg to Python Path ---
# This block MUST run before 'import carla'
try:
    # This should be the root directory of your CARLA installation
    CARLA_ROOT = "/home/faizaladin/Desktop/carla"
    
    # Find the .egg file that matches the running Python version (e.g., 3.7)
    dist_path = os.path.join(CARLA_ROOT, 'PythonAPI/carla/dist')
    egg_files = glob.glob(f'{dist_path}/carla-*-py{sys.version_info.major}.{sys.version_info.minor}-*.egg')
    
    if not egg_files:
        raise IndexError
        
    # Add the found .egg file to the Python path
    sys.path.append(egg_files[0])

except IndexError:
    print(f"❌ Could not find a CARLA .egg file for Python {sys.version_info.major}.{sys.version_info.minor}.")
    print(f"   Please verify that CARLA_ROOT is set correctly to '{CARLA_ROOT}'")
    print("   and that you have a compiled .egg file for your Python version.")
    sys.exit()

# Now that the path is correctly set, this import will succeed.
import carla

# --- Configuration ---
CARLA_EXECUTABLE = os.path.join(CARLA_ROOT, "CarlaUE4.sh")
HOST = 'localhost'
PORT = 2000
TM_PORT = 8012
# Original sensor resolution
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600
# New dimensions for the saved, resized images
RESIZED_IMAGE_WIDTH = 400
RESIZED_IMAGE_HEIGHT = 300
MAPS = ["Town04"]  # Set to run only on Town03 as requested
FIXED_DELTA_SECONDS = 0.1 # Corresponds to 10 FPS

def restart_simulator():
    """Kills any running CARLA processes and starts a new one in the background."""
    print("Killing existing Carla processes...")
    os.system("pkill -f CarlaUE4-Linux-Shipping")
    time.sleep(5)  # Allow time for system ports to be released

    print("Starting new Carla simulator instance...")
    env = os.environ.copy()
    env["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/nvidia_icd.json"
    
    # Run the simulator in the background and suppress its output for a cleaner log
    subprocess.Popen(
        [CARLA_EXECUTABLE, "-RenderOffScreen"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT
    )
    print("Waiting 20 seconds for the simulator to initialize...")
    time.sleep(20)

def collect_data_on_map(map_name, output_dir):
    """Connects to CARLA, runs the simulation on a given map, and collects data."""
    client = None
    world = None
    vehicle = None
    camera = None
    original_settings = None
    
    try:
        # --- 1. Connect to Simulator ---
        for _ in range(10): # Retry connection up to 10 times
            try:
                client = carla.Client(HOST, PORT)
                client.set_timeout(10.0)
                world = client.load_world(map_name)
                print(f"✅ Successfully connected to CARLA and loaded {map_name}")
                break
            except RuntimeError as e:
                print(f"Connection failed: {e}. Retrying in 2 seconds...")
                time.sleep(2)
        
        if not world:
            raise RuntimeError("Could not connect to CARLA simulator after multiple attempts.")
        
        world.set_weather(carla.WeatherParameters.Default)

        # --- 2. Configure World and Traffic Manager for Synchronous Mode ---
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        world.apply_settings(settings)

        tm = client.get_trafficmanager(TM_PORT)
        tm.set_synchronous_mode(True)
        tm.set_random_device_seed(42) # For reproducible autopilot behavior

        # --- 3. Spawn Actors (Vehicle and Camera) ---
        blueprint_library = world.get_blueprint_library()
        
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        spawn_point = world.get_map().get_spawn_points()[0]
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        
        vehicle.set_autopilot(True, TM_PORT)
        tm.auto_lane_change(vehicle, False)
        tm.ignore_lights_percentage(vehicle, 100)
        #tm.ignore_stop_signs(vehicle, True)

        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(IMAGE_WIDTH))
        camera_bp.set_attribute('image_size_y', str(IMAGE_HEIGHT))
        camera_bp.set_attribute('fov', '90')
        camera_transform = carla.Transform(carla.Location(x=2.0, z=1.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

        # --- 4. Data Collection Loop ---
        os.makedirs(output_dir, exist_ok=True)
        data_labels = []
        
        image_queue = queue.Queue()
        camera.listen(image_queue.put)

        start_location = spawn_point.location
        max_frames = 20000  # Safety limit to prevent infinite loops

        print(f"🚀 Starting data collection loop for one lap...")
        world.tick() # Initial tick

        for frame_num in range(max_frames):
            world.tick()
            
            image = image_queue.get() # Wait for the image from this frame
            control = vehicle.get_control() # Get control command from the same frame
            
            # Start recording after a short "warm-up" period
            if frame_num > 100:
                # --- IMAGE RESIZING LOGIC ---
                # Convert raw sensor data to a NumPy array
                array = np.frombuffer(image.raw_data, dtype=np.uint8)
                array = array.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 4))
                
                # Keep only the RGB channels (discard the Alpha channel)
                array = array[:, :, :3]
                
                # Create a Pillow Image object
                img = Image.fromarray(array)
                
                # Resize the image using a high-quality downsampling filter
                img = img.resize((RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT), Image.LANCZOS)
                
                # Save the resized image to disk
                img.save(os.path.join(output_dir, f"{image.frame}.png"))

                # --- END OF RESIZING LOGIC ---

                velocity = vehicle.get_velocity()
                speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])
                location = vehicle.get_location()

                data_labels.append((image.frame, control.steer, speed, location.x, location.y, location.z))

                # Check if the car has returned near the starting point
                if frame_num > 200 and location.distance(start_location) < 5.0:
                    print("🏁 Completed one lap.")
                    break
        
    finally:
        # --- 5. Cleanup ---
        # This block ensures actors are destroyed even if an error occurs
        print("Cleaning up actors and resetting world settings...")
        if world is not None and original_settings is not None:
            world.apply_settings(original_settings) # Revert to async mode
        if camera is not None:
            camera.stop()
            camera.destroy()
        if vehicle is not None:
            vehicle.destroy()
        print("Cleanup complete.")

    # --- 6. Save Labels to CSV ---
    if data_labels:
        with open(os.path.join(output_dir, "labels.csv"), "w") as f:
            f.write("frame,steer,speed,x,y,z\n")
            for row in data_labels:
                f.write(f"{row[0]},{row[1]:.6f},{row[2]:.6f},{row[3]:.6f},{row[4]:.6f},{row[5]:.6f}\n")
        print(f"💾 Saved {len(data_labels)} data points to '{output_dir}/labels.csv'")

def main():
    for i, map_name in enumerate(MAPS, 1):
        print(f"\n{'='*50}\nProcessing Map {i}/{len(MAPS)}: {map_name}\n{'='*50}")
        
        restart_simulator()
        
        output_dir = f"town_{i:02d}_{map_name}"
        collect_data_on_map(map_name, output_dir)
        
    print("\n🎉 Data collection complete. 🎉")

if __name__ == "__main__":
    main()

