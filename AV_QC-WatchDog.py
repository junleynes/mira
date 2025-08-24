import os
import time
import subprocess
import logging
import requests
import signal
from datetime import datetime
from multiprocessing import Process, Lock
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import re

# Configuration Variables
LOG_LEVEL = logging.INFO
STABILITY_THRESHOLD = 5  # Time (seconds) for file stability check
MAX_WAIT_TIME = 600  # Maximum wait time before processing
NETWORK_PATH = r"\\10.0.1.130\PMC_MAMS_PUBLISHING"
USERNAME = r"postsns\stanza"  # Replace with your network username
PASSWORD = r"gma7mamS"  # Replace with your network password
FOLDER_MAPPING = {
    r"\\10.0.1.130\PMC_MAMS_PUBLISHING\QC": (r"D:\STANZA_SHARE\POSTMAMS\scripts\FRAMERATE-CHECKER\av-qc-tool.py", True),
}

MEDIA_EXTENSIONS = {
    '.mp4', '.mov', '.avi', '.mkv', '.mxf', '.mpg', '.mpeg', '.wmv', '.flv',
    '.mp3', '.wav', '.aiff', '.aif', '.flac', '.m4a', '.wma', '.aac'
}

# Configure logging
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")

def clean_filename(filename):
    """Cleans the filename by removing leading/trailing spaces and replacing internal spaces with underscores."""
    # Strip leading and trailing spaces from the entire filename
    filename = filename.strip()

    # Split the filename into base name and extension
    base_name, ext = os.path.splitext(filename)

    # Strip any leading or trailing spaces from the base name (before the extension)
    base_name = base_name.strip()

    # Replace spaces with underscores inside the base name, but not before the extension
    base_name = re.sub(r' (?=\S)', '_', base_name)  # Replace internal spaces with underscores

    # Rebuild the filename with the cleaned base name and extension
    filename = base_name + ext

    # Define invalid characters (you can add or modify this list as needed)
    invalid_chars = r'<>:"/\|?*-'

    # Remove any invalid characters using a regular expression
    filename = re.sub(r'[' + re.escape(invalid_chars) + r']', '', filename)

    return filename

class FolderWatchHandler(FileSystemEventHandler):
    def __init__(self, script_path, is_python):
        self.script_path = script_path
        self.is_python = is_python
        self.lock = Lock()

    def wait_for_stable_file(self, file_path):
        initial_size = -1
        unchanged_time = 0
        wait_start_time = time.time()
        wait_interval = 1

        while unchanged_time < STABILITY_THRESHOLD:
            try:
                current_size = os.path.getsize(file_path)
                if current_size == initial_size:
                    unchanged_time += wait_interval
                    wait_interval = min(wait_interval * 2, STABILITY_THRESHOLD)
                else:
                    unchanged_time = 0
                    wait_interval = 1
                initial_size = current_size
            except IOError:
                logging.info(f"File {file_path} is still being written to or locked.")
                unchanged_time = 0
            time.sleep(wait_interval)
            if time.time() - wait_start_time > MAX_WAIT_TIME:
                logging.warning(f"File {file_path} did not stabilize within {MAX_WAIT_TIME} seconds.")
                return False
        return True

    def move_to_folder(self, file_path, folder_name):
        source_dir = os.path.dirname(file_path)
        base_dir = os.path.dirname(source_dir)
        target_dir = os.path.join(base_dir, folder_name)

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        destination_path = os.path.join(target_dir, os.path.basename(file_path))

        if os.path.exists(destination_path):
            filename, ext = os.path.splitext(os.path.basename(file_path))
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            destination_path = os.path.join(target_dir, f"{filename}_{timestamp}{ext}")

        with self.lock:
            try:
                os.rename(file_path, destination_path)
                logging.info(f"Moved file to: {destination_path}")
            except Exception as e:
                logging.error(f"Failed to move file {file_path} to {destination_path}: {e}")

    def run_script(self, file_arg):
        filename = os.path.basename(file_arg)
        try:
            logging.info("Processing.")
            command = ["python", self.script_path, file_arg] if self.is_python else ["powershell", "-File", self.script_path, file_arg]
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            #subprocess.run(command, check=True)
            logging.info(f"Script executed successfully for: {file_arg}")
            self.move_to_folder(file_arg, "PROCESSED")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error running script for {file_arg}: {e}")
            self.move_to_folder(file_arg, "FAILED")
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            self.move_to_folder(file_arg, "FAILED")



    def on_created(self, event):
        if event.is_directory:
            return

        file_path = event.src_path
        filename = os.path.basename(file_path)
        file_ext = os.path.splitext(filename)[1].lower()  # Get file extension in lowercase
        
        # Skip if not a media file
        if file_ext not in MEDIA_EXTENSIONS:
            logging.info(f"Skipping non-media file: {file_path}")
            return

        logging.info(f"New media file detected: {file_path}")

        # Rest of the existing on_created method remains the same...
        if self.wait_for_stable_file(file_path):
            # Clean the filename after it's stable
            cleaned_filename = clean_filename(filename)
            cleaned_file_path = os.path.join(os.path.dirname(file_path), cleaned_filename)

            if cleaned_file_path != file_path:
                try:
                    # Rename the file immediately after it stabilizes
                    os.rename(file_path, cleaned_file_path)
                    logging.info(f"Renamed file to: {cleaned_file_path}")
                    file_path = cleaned_file_path  # Update the file_path to the new one
                except Exception as e:
                    logging.error(f"Failed to rename file {file_path} to {cleaned_file_path}: {e}")
                    return  # Exit the method if renaming failed

            try:
                logging.info("Transfer is ongoing.")
            except Exception as e:
                logging.error(f"Failed to send status update: {e}")

            # Proceed with script execution
            process = Process(target=self.run_script, args=(file_path,))
            process.start()

def purge_old_files():
    expiry_time = 72 * 3600  # 72 hours in seconds
    base_dirs = [r"\\10.0.1.130\PMC_MAMS_PUBLISHING\WATCH\PROCESSED", r"\\10.0.1.130\PMC_MAMS_PUBLISHING\WATCH\FAILED"]

    for folder in base_dirs:
        if os.path.exists(folder):
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                if os.path.isfile(file_path):
                    file_age = time.time() - os.path.getctime(file_path)
                    if file_age > expiry_time:
                        try:
                            os.remove(file_path)
                            logging.info(f"Deleted old file: {file_path}")
                        except Exception as e:
                            logging.error(f"Failed to delete {file_path}: {e}")

def is_network_drive_mounted(network_path):
    try:
        output = subprocess.run("net use", shell=True, check=True, capture_output=True, text=True)
        return network_path.lower() in output.stdout.lower()
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to check network drives: {e}")
        return False

def map_network_drive():
    if is_network_drive_mounted(NETWORK_PATH):
        logging.info(f"Network drive {NETWORK_PATH} is already mounted. Skipping mapping.")
        return
    command = f'net use {NETWORK_PATH} /user:{USERNAME} {PASSWORD}'
    try:
        subprocess.run(command, shell=True, check=True)
        logging.info(f"Successfully mapped network drive: {NETWORK_PATH}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to map network drive: {e}")
        raise

def start_folder_watch():
    observer = Observer()
    for folder, (script_path, is_python) in FOLDER_MAPPING.items():
        if not os.path.exists(script_path):
            logging.error(f"Script not found: {script_path}")
            continue
        event_handler = FolderWatchHandler(script_path, is_python)
        observer.schedule(event_handler, folder, recursive=False)
        logging.info(f"Watching folder: {folder}")
    observer.start()
    logging.info("Folder watch started.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logging.info("Stopping folder watch...")
    observer.join()

if __name__ == "__main__":
    # Start Flask server
    server_process = subprocess.Popen(["python", "server.py"])
    try:
        map_network_drive()
        #purge_old_files()
        start_folder_watch()
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        # Stop Flask server
        os.kill(server_process.pid, signal.SIGTERM)
        print("Flask server stopped.")