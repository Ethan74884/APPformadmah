import os
import cv2
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.filechooser import FileChooserListView
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image as PILImage
from torch.utils.data import Dataset
import heapq


class ParkingSpotDataset(Dataset):
    def __init__(self, image, parking_spots, transform=None):
        self.image = image
        self.parking_spots = parking_spots
        self.transform = transform

    def __len__(self):
        return len(self.parking_spots)
        
    def __getitem__(self, idx):
        spot = self.parking_spots[idx]
        x, y, h, w, spot_id = spot['x'], spot['y'], spot['h'], spot['w'], spot['id']
        spot_image = self.image[y:y+h, x:x+w]
        # Convert the NumPy array to a PIL Image
        spot_image_pil = PILImage.fromarray(cv2.cvtColor(spot_image, cv2.COLOR_BGR2RGB))
        if self.transform:
            spot_image_pil = self.transform(spot_image_pil)
        return spot_image_pil, spot_id


def process_parking_spots(dataset, model):
    results = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            image, spot_id = dataset[i]
            image = image.unsqueeze(0).to(device)
            output = model(image)
            _, predicted = torch.max(output, 1)
            prediction = predicted.item()
            results.append({'id': spot_id, 'prediction': prediction})
    results = pd.DataFrame(results)
    print("Predictions:", results)
    return results


def multi_scale_template_matching(img, template, scale_range=(0.5, 3.0), scale_steps=40):
    """Search for the template at different scales"""
    best_score = -1
    best_loc = None
    best_scale = None
    best_template_shape = None
    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) > 2 else img
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) > 2 else template

    # Get template dimensions
    h, w = template.shape[:2]
    print("Template shape:", template.shape)
    for scale in np.linspace(scale_range[0], scale_range[1], scale_steps):
        # Resize the template
        resized_template = cv2.resize(template_gray, (int(w * scale), int(h * scale)))

        # Skip if template is larger than image
        if resized_template.shape[0] > img_gray.shape[0] or resized_template.shape[1] > img_gray.shape[1]:
            continue

        # Apply template matching
        result = cv2.matchTemplate(img_gray, resized_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best_score:
            best_score = max_val
            best_loc = max_loc
            best_template_shape = resized_template.shape[:2]
            best_scale = scale

    if best_loc is None:
        return None  # No matches found

    # Extract dimensions of the best matched template
    h, w = best_template_shape

    # Calculate the bottom-right corner of the matched region
    top_left = best_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # Crop the matched region
    cropped_img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    # Resize to match the original template size
    cropped_img = cv2.resize(cropped_img, (template.shape[1], template.shape[0]))

    return cropped_img


def transform_bboxes_to_grid(image, parking_spots, result, yolo_results, grid_size):
    """
    Transforms bounding boxes to a 2D grid representation.
    Args:
        image: The image as a numpy array.
        parking_spots: A list of dictionaries with parking spot information.
        result: DataFrame with prediction results.
        yolo_results: Results from YOLO object detection.
        grid_size: The size of the grid (e.g., 10x10).
    Returns:
        A 2D numpy array representing the grid.
    """
    image_height, image_width, _ = image.shape
    grid = np.zeros(grid_size, dtype=int)
    cell_height = image_height / grid_size[0]
    cell_width = image_width / grid_size[1]
    
    for spot in parking_spots:
        x, y, h, w, spot_id = spot['x'], spot['y'], spot['h'], spot['w'], spot['id']
        # Calculate the grid indices for the bounding box
        start_row = int(y // cell_height)
        end_row = int((y + h) // cell_height)
        start_col = int(x // cell_width)
        end_col = int((x + w) // cell_width)
        # Clip indices to be within the grid bounds
        start_row = max(0, min(start_row, grid_size[0] - 1))
        end_row = max(0, min(end_row, grid_size[0] - 1))
        start_col = max(0, min(start_col, grid_size[1] - 1))
        end_col = max(0, min(end_col, grid_size[1] - 1))
        # Assign the parking spot ID to the corresponding grid cells
        for i in range(start_row, end_row + 1):
            for j in range(start_col, end_col + 1):
                grid[i, j] = result.loc[result['id'] == spot_id, 'prediction'].iloc[0] + 1
    
    # Process YOLO results
    for box in yolo_results[0].boxes:  # Iterate through detected boxes in the first result
        x1, y1, x2, y2, conf, cls = box.xyxy[0].tolist() + [box.conf.item(), box.cls.item()]
        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
        # Calculate grid indices and clip similar to parking_spots processing
        start_row = int(y // cell_height)
        end_row = int((y + h) // cell_height)
        start_col = int(x // cell_width)
        end_col = int((x + w) // cell_width)
        start_row = max(0, min(start_row, grid_size[0] - 1))
        end_row = max(0, min(end_row, grid_size[0] - 1))
        start_col = max(0, min(start_col, grid_size[1] - 1))
        end_col = max(0, min(end_col, grid_size[1] - 1))
        # Assign YOLO class label to grid cells
        for i in range(start_row, end_row + 1):
            for j in range(start_col, end_col + 1):
                grid[i, j] = 1  # Mark as obstacle
    
    print("Grid values:", np.unique(grid))
    return grid


def heuristic(a, b):
    """Calculates the Manhattan distance with a slight downward bias."""
    manhattan = abs(a[0] - b[0]) + abs(a[1] - b[1])
    # Small penalty for upward position (smaller row values)
    # This breaks ties in favor of downward movement
    down_bias = -0.001 * a[0]  # Slightly favor larger row values
    return manhattan + down_bias


def get_neighbors(grid, node):
    """Returns a list of valid neighbors for a given node on the grid."""
    rows, cols = grid.shape
    row, col = node
    neighbors = []
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < rows and 0 <= new_col < cols and grid[new_row, new_col] != 1:
            neighbors.append((new_row, new_col))
    return neighbors


def reconstruct_path(came_from, current):
    """Reconstructs the path from the came_from dictionary."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def find_path_astar(grid, start, end):
    """
    Finds the shortest path from start to end on a grid using the A* search algorithm.
    """
    print("Start position:", start)
    print("End position:", end)
    print("Grid values around start:", grid[max(0, start[0]-1):start[0]+2, max(0, start[1]-1):start[1]+2])
    print("Grid values around end:", grid[max(0, end[0]-1):end[0]+2, max(0, end[1]-1):end[1]+2])

    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}
    
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == end:
            return reconstruct_path(came_from, current)
        
        for neighbor in get_neighbors(grid, current):
            tentative_g_score = g_score[current] + 1  # Assuming uniform cost
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                if neighbor not in open_set:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return None


def display_path_on_grid(grid, path):
    """
    Displays the grid with the path highlighted.

    Args:
        grid: A 2D numpy array representing the grid.
        path: A list of tuples representing the path.
    """
    plt.imshow(grid, cmap='viridis', interpolation='nearest')
    plt.colorbar()

    if path:
        path_x, path_y = zip(*path)  # Unpack path tuples
        plt.plot(path_y, path_x, marker='o', color='red', linestyle='-') # Plot path

    plt.show()


def path_to_instructions(path, pixel_to_meter=0.04):
    """
    Transforms a path (list of coordinates) into driving instructions,
    batching consecutive "Drive straight" instructions.
    """
    instructions = ""
    if not path:
        return "No path found."
        
    current_direction = 0  # 0: North, 1: East, 2: South, 3: West
    current_position = path[0]
    total_distance = 0  # Accumulate distance for straight segments
    
    for i in range(1, len(path)):
        next_position = path[i]
        delta_row = next_position[0] - current_position[0]
        delta_col = next_position[1] - current_position[1]
        
        new_direction = 0
        if delta_col > 0:
            new_direction = 1  # East
        elif delta_col < 0:
            new_direction = 3  # West
        elif delta_row > 0:
            new_direction = 2  # South
        elif delta_row < 0:
            new_direction = 0  # North
            
        if new_direction != current_direction:
            # If direction changes, add accumulated straight distance
            if total_distance > 0:
                instructions += f"Drive straight for {total_distance * pixel_to_meter:.2f} meters. "
                total_distance = 0  # Reset for the next straight segment
            
            if (current_direction - new_direction) % 4 == 1:
                instructions += "Turn left. "
            elif (current_direction - new_direction) % 4 == 3:
                instructions += "Turn right. "
                
            current_direction = new_direction
            
        total_distance += ((delta_row**2) + (delta_col**2))**0.5
        current_position = next_position
        
    # Add any remaining straight distance at the end
    if total_distance > 0:
        instructions += f"Drive straight for {total_distance * pixel_to_meter:.2f} meters. "
        
    return instructions


def run_all_processes(img, template, model2, model):
    """Main function to run all parking analysis processes"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),          # Convert to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet mean and std
                            std=[0.229, 0.224, 0.225])])

    # Hardcoded parking spots (would be better to detect these automatically)
    parking_spots = []
    parking_spots.append({'x': 110, 'y': 45, 'h': 30, 'w': 80, 'id': 1})
    parking_spots.append({'x': 95, 'y': 80, 'h': 30, 'w': 80, 'id': 2})
    parking_spots.append({'x': 85, 'y': 115, 'h': 35, 'w': 85, 'id': 3})
    parking_spots.append({'x': 65, 'y': 145, 'h': 35, 'w': 85, 'id': 4})
    parking_spots.append({'x': 50, 'y': 185, 'h': 35, 'w': 85, 'id': 5})
    parking_spots.append({'x': 35, 'y': 220, 'h': 35, 'w': 85, 'id': 6})

    # Call functions in the desired order
    cropped_img = multi_scale_template_matching(img, template)
    dataset = ParkingSpotDataset(cropped_img, parking_spots, transform=transform)
    results_df = process_parking_spots(dataset, model)
    yolo_results = model2.predict(cropped_img)

    grid_size = (500, 500)  # Example grid size
    grid = transform_bboxes_to_grid(cropped_img, parking_spots, results_df, yolo_results, grid_size)

    start = (0, 330)
    # Find all grid cells with value 2 (valid parking spots)
    parking_spots = np.argwhere(grid == 2)

    # Check if there are no valid parking spots
    if parking_spots.size == 0:
        print("No valid parking spots found.")
        return "No valid parking spots found."

    # Find the closest parking spot to the start position
    target_coordinates = min(parking_spots,key=lambda spot: abs(start[0] - spot[0]) + abs(start[1] - spot[1]))

    # Ensure target_coordinates is valid
    if target_coordinates is not None:
        end = tuple(target_coordinates)  # Convert to tuple
        path = find_path_astar(grid, start, end)
        #display_path_on_grid(grid, path)
        if path:
            instructions = path_to_instructions(path, 0.04)
            return instructions
        else:
            print("No path could be found to the target.")
            return "No path could be found to the target."
    else:
        print("No valid target coordinates found.")
        return "No valid target coordinates found."

        if target_coordinates[0].size > 0:
            end = (target_coordinates[0][0], target_coordinates[1][0])
            path = find_path_astar(grid, start, end)
            display_path_on_grid(grid, path)
            if path:
                instructions = path_to_instructions(path, 0.04)
                return instructions
    
        return None


class ParkingApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')
        
        # Video source options
        self.source_layout = BoxLayout(orientation='horizontal', size_hint_y=0.1)
        self.webcam_button = Button(text="Use Webcam")
        self.webcam_button.bind(on_press=self.start_webcam)
        self.source_layout.add_widget(self.webcam_button)
        
        self.video_button = Button(text="Load Video File")
        self.video_button.bind(on_press=self.load_video)
        self.source_layout.add_widget(self.video_button)
        self.layout.add_widget(self.source_layout)
        
        # Video display
        self.video_widget = Image(size_hint_y=0.6)
        self.layout.add_widget(self.video_widget)
        
        # Control buttons
        self.controls_layout = BoxLayout(orientation='horizontal', size_hint_y=0.1)
        
        self.play_pause_button = Button(text="Play", disabled=True)
        self.play_pause_button.bind(on_press=self.toggle_playback)
        self.controls_layout.add_widget(self.play_pause_button)
        
        self.capture_button = Button(text="Capture Frame", disabled=True)
        self.capture_button.bind(on_press=self.capture_frame)
        self.controls_layout.add_widget(self.capture_button)
        
        self.layout.add_widget(self.controls_layout)
        
        # Add an image widget to display the captured frame
        self.image_widget = Image(size_hint_y=0.3)
        self.layout.add_widget(self.image_widget)
        
        # Add a button to run all processes
        self.run_button = Button(text="Run Process", size_hint_y=0.1, disabled=True)
        self.run_button.bind(on_press=self.run_process)
        self.layout.add_widget(self.run_button)
        
        # Add a label to display instructions
        self.instructions_label = Label(text="Select video source to begin", size_hint_y=0.1)
        self.layout.add_widget(self.instructions_label)
        
        # Set up file chooser (hidden initially)
        self.file_chooser = FileChooserListView(
            path=os.path.expanduser('~'),
            filters=['*.mp4', '*.avi', '*.mkv', '*.mov'],
            size_hint_y=0.4
        )
        self.file_chooser.bind(on_submit=self.on_video_selected)
        
        # Initialize video-related variables
        self.video_source = None
        self.is_playing = False
        self.clock_event = None
        
        return self.layout
    
    def get_app_directory(self):
        # All files are contained within the APPformadmah directory
        return os.path.dirname(os.path.abspath(__file__))
    
    def load_video(self, instance):
        if hasattr(self, 'file_chooser'):
            if self.file_chooser in self.layout.children:
                self.layout.remove_widget(self.file_chooser)
            else:
                self.layout.add_widget(self.file_chooser)
    
    def on_video_selected(self, instance, selection, *args):
        if selection:
            try:
                video_path = selection[0]
                self.setup_video_source(cv2.VideoCapture(video_path))
                self.instructions_label.text = f"Video loaded: {os.path.basename(video_path)}"
                if self.file_chooser in self.layout.children:
                    self.layout.remove_widget(self.file_chooser)
            except Exception as e:
                self.instructions_label.text = f"Error loading video: {str(e)}"
    
    def start_webcam(self, instance):
        try:
            self.setup_video_source(cv2.VideoCapture(0))
            self.instructions_label.text = "Webcam activated"
        except Exception as e:
            self.instructions_label.text = f"Error accessing webcam: {str(e)}"
    
    def setup_video_source(self, video_capture):
        # Stop any existing video playback
        if self.clock_event:
            self.clock_event.cancel()
        
        # Set up new video source
        self.video_source = video_capture
        if not self.video_source.isOpened():
            self.instructions_label.text = "Failed to open video source"
            return
        
        # Enable control buttons
        self.play_pause_button.disabled = False
        self.capture_button.disabled = False
        
        # Start video playback
        self.is_playing = True
        self.play_pause_button.text = "Pause"
        self.update_video_frame()
    
    def update_video_frame(self, dt=None):
        if self.video_source and self.is_playing:
            ret, frame = self.video_source.read()
            if ret:
                # Rotate the frame 90 degrees clockwise for tall videos
                if frame.shape[0] < frame.shape[1]:  # Check if the video is taller than it is wide
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                # Convert the frame to texture for display
                buf = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                buf = cv2.flip(buf, 0)  # Flip the frame vertically
                texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
                texture.blit_buffer(buf.flatten(), colorfmt='rgb', bufferfmt='ubyte')
                self.video_widget.texture = texture

                # Save current frame for processing
                self.current_frame = frame

                # Schedule next frame update (aiming for real-time playback)
                self.clock_event = Clock.schedule_once(self.update_video_frame, 1/30)  # ~30 FPS
            else:
                # Handle end of video or error
                self.video_source.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop back to start
                self.clock_event = Clock.schedule_once(self.update_video_frame, 1/30)

                
    def toggle_playback(self, instance):
        if self.video_source:
            self.is_playing = not self.is_playing
            if self.is_playing:
                instance.text = "Pause"
                self.update_video_frame()
            else:
                instance.text = "Play"
                if self.clock_event:
                    self.clock_event.cancel()
    
    def capture_frame(self, instance):
        try:
            if not hasattr(self, 'current_frame'):
                self.instructions_label.text = "No video frame available"
                return
            
            # Store the current frame directly in memory without saving to disk
            self.captured_frame = self.current_frame.copy()
            # Display captured frame
            buf = cv2.cvtColor(self.captured_frame, cv2.COLOR_BGR2RGB)
            buf = cv2.flip(buf, 0)  # Flip the frame vertically
            texture = Texture.create(size=(buf.shape[1], buf.shape[0]), colorfmt='rgb')
            texture.blit_buffer(buf.flatten(), colorfmt='rgb', bufferfmt='ubyte')
            self.image_widget.texture = texture
            self.instructions_label.text = "Frame captured. Press 'Run Process' to analyze."
            self.run_button.disabled = False
        except Exception as e:
            self.instructions_label.text = f"Error capturing frame: {str(e)}"
    
    def run_process(self, instance):
        try:
            if not hasattr(self, 'captured_frame'):
                self.instructions_label.text = "Please capture a frame first"
                return
            
            app_dir = self.get_app_directory()
            
            # Load template image from APPformadmah directory
            template_path = os.path.join(app_dir, 'template.jpg')
            if not os.path.exists(template_path):
                self.instructions_label.text = "Template image not found"
                return
            
            template = cv2.imread(template_path)
            if template is None:
                self.instructions_label.text = "Error loading template image"
                return
            
            # Load YOLO model directly from APPformadmah
            model_path = os.path.join(app_dir, 'yolov8m.pt')
            if not os.path.exists(model_path):
                self.instructions_label.text = "YOLO model not found"
                return
            
            model2 = YOLO(model_path)
            
            # Load parking classification model
            model = models.mobilenet_v2()
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, 2)
            
            model_state_path = os.path.join(app_dir, 'modelfinal.pth')
            if not os.path.exists(model_state_path):
                self.instructions_label.text = "Classification model not found"
                return
            
            # Load model weights to CPU if CUDA is not available
            if torch.cuda.is_available():
                model.load_state_dict(torch.load(model_state_path))
                model = model.cuda()
            else:
                model.load_state_dict(torch.load(model_state_path, map_location=torch.device('cpu')))
            model.eval()
            
            # Process the captured frame
            self.instructions_label.text = "Processing frame..."
            instructions = run_all_processes(self.captured_frame, template, model2, model)
            if instructions:
                self.instructions_label.text = instructions
            else:
                self.instructions_label.text = "No available parking spots found"
        
        except Exception as e:
            self.instructions_label.text = f"Error processing frame: {str(e)}"
    
    def on_stop(self):
        # Clean up resources when app closes
        if self.clock_event:
            self.clock_event.cancel()
        if hasattr(self, 'video_source') and self.video_source:
            self.video_source.release()

if __name__ == "__main__":
    ParkingApp().run()