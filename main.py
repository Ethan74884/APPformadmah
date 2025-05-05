from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image as KivyImage
from kivy.uix.camera import Camera
from PIL import Image as PILImage
from kivy.uix.filechooser import FileChooserListView
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models,transforms, datasets
import torch.optim as optim
import cv2
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO
import heapq
from kivy.clock import Clock
from kivy.utils import platform
import time


class ParkingApp(App):
    def build(self):
        # Request Android permissions if on Android
        if platform == 'android':
            request_permissions([
                Permission.CAMERA,
                Permission.WRITE_EXTERNAL_STORAGE,
                Permission.READ_EXTERNAL_STORAGE
            ])
           
        self.layout = BoxLayout(orientation='vertical')
       
        # Try to initialize camera
        try:
            self.camera = Camera(play=False)
            self.has_camera = True
            self.layout.add_widget(self.camera)
            self.capture_button = Button(text="Capture Image", size_hint_y=0.1)
            self.capture_button.bind(on_press=self.capture_image)
            self.layout.add_widget(self.capture_button)
            # Start the camera
            self.camera.play = True
        except Exception as e:
            print(f"Camera error: {str(e)}")
            self.has_camera = False
            # Add a button to load an image instead
            self.load_button = Button(text="Load Image", size_hint_y=0.1)
            self.load_button.bind(on_press=self.load_image)
            self.layout.add_widget(self.load_button)
           
            # Set appropriate path for Android or desktop
            if platform == 'android':
                user_path = primary_external_storage_path()
            else:
                user_path = os.path.expanduser('~')
               
            # Add file chooser (hidden initially)
            self.file_chooser = FileChooserListView(
                path=user_path,
                filters=['*.jpg', '*.png', '*.jpeg'],
                size_hint_y=0.4,
                rootpath=user_path
            )
            self.file_chooser.bind(on_submit=self.on_file_selected)
       
        # Add an image widget to display the captured/loaded image
        self.image_widget = KivyImage(size_hint_y=0.4)
        self.layout.add_widget(self.image_widget)
       
        # Add a button to run all processes
        self.run_button = Button(text="Run Process", size_hint_y=0.1)
        self.run_button.bind(on_press=self.run_process)
        self.layout.add_widget(self.run_button)
       
        # Add a label to display instructions
        initial_text = "Press 'Capture Image' to take a photo" if self.has_camera else "Press 'Load Image' to select an image"
        self.instructions_label = Label(text=initial_text, size_hint_y=0.1)
        self.layout.add_widget(self.instructions_label)
       
        return self.layout
   
    def get_app_directory(self):
        if platform == 'android':
            return os.path.join(primary_external_storage_path(), 'ParkingApp')
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ParkingApp')
   
    def load_image(self, instance):
        if hasattr(self, 'file_chooser'):
            if self.file_chooser in self.layout.children:
                self.layout.remove_widget(self.file_chooser)
            else:
                self.layout.add_widget(self.file_chooser)


    def on_file_selected(self, instance, selection, *args):
        if selection:
            try:
                self.image_path = selection[0]
                self.image_widget.source = self.image_path
                self.instructions_label.text = "Image loaded. Press 'Run Process' to analyze."
                if self.file_chooser in self.layout.children:
                    self.layout.remove_widget(self.file_chooser)
            except Exception as e:
                self.instructions_label.text = f"Error loading image: {str(e)}"
   
    def capture_image(self, instance):
        try:
            # Create app directory if it doesn't exist
            app_dir = self.get_app_directory()
            os.makedirs(os.path.join(app_dir, 'captures'), exist_ok=True)
           
            # Create a filename with timestamp
            filename = os.path.join(app_dir, 'captures', f"capture_{int(time.time())}.jpg")
            self.camera.export_to_png(filename)
            self.image_path = filename
            self.image_widget.source = filename
            self.instructions_label.text = "Image captured. Press 'Run Process' to analyze."
        except Exception as e:
            self.instructions_label.text = f"Error capturing image: {str(e)}"
   
    def run_process(self, instance):
        try:
            if not hasattr(self, 'image_path'):
                self.instructions_label.text = "Please capture or load an image first"
                return
           
            app_dir = self.get_app_directory()
           
            # Load template image from app resources
            template_path = os.path.join(app_dir, 'resources', 'template.jpg')
            if not os.path.exists(template_path):
                self.instructions_label.text = "Template image not found"
                return
               
            template = cv2.imdecode(np.fromfile(template_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if template is None:
                self.instructions_label.text = "Error loading template image"
                return
   
            # Load YOLO model
            model_path = os.path.join(app_dir, 'models', 'yolov8m.pt')
            if not os.path.exists(model_path):
                self.instructions_label.text = "YOLO model not found"
                return
               
            model2 = YOLO(model_path)
   
            # Load parking classification model
            model = models.mobilenet_v2()
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, 2)
           
            model_state_path = os.path.join(app_dir, 'models', 'modelfinal.pth')
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
           
            # Load and process the image
            img = cv2.imdecode(np.fromfile(self.image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                self.instructions_label.text = "Error loading image"
                return
           
            # Call run_all_processes and update UI with results
            self.instructions_label.text = "Processing image..."
            instructions = run_all_processes(img, template, model2, model)
            if instructions:
                self.instructions_label.text = instructions
            else:
                self.instructions_label.text = "No available parking spots found"


        except Exception as e:
            self.instructions_label.text = f"Error processing image: {str(e)}"


   
   
def path_to_instructions(path, pixel_to_meter=0.02):
    """
    Transforms a path (list of coordinates) into driving instructions,
    batching consecutive "Drive straight" instructions.
    Args:
        path: A list of (row, col) tuples representing the path.
        pixel_to_meter: Conversion factor from pixels to meters.
    Returns:
        A string containing driving instructions.
    """
    instructions = ""
    if not path:
        return "No path found."
    current_direction = 0 # 0: North, 1: East, 2: South, 3: West
    current_position = path[0]
    total_distance = 0  # Accumulate distance for straight segments
    for i in range(1,len(path)):
        next_position = path[i]
        delta_row = next_position[0] - current_position[0]
        delta_col = next_position[1] - current_position[1]
        new_direction = 0
        if delta_col > 0:
            new_direction = 1  # East
        elif delta_col < 0:
            new_direction = 3  # West
        elif delta_row > 0:
            new_direction = 2 # South
        elif delta_row < 0:
            new_direction = 0 # North
        if new_direction != current_direction:
            # If direction changes, add accumulated straight distance
            if total_distance > 0:
                instructions += f"Drive straight for {total_distance * pixel_to_meter:.2f} meters. "
                total_distance = 0  # Reset for the next straight segment
            if (current_direction - new_direction) % 4 == 1:
                instructions += "Turn right. "
            elif (current_direction - new_direction) % 4 == 3:
                instructions += "Turn left. "
            current_direction = new_direction
        total_distance += ((delta_row**2) + (delta_col**2))**0.5
        current_position = next_position
    # Add any remaining straight distance at the end
    if total_distance > 0:
        instructions += f"Drive straight for {total_distance * pixel_to_meter:.2f} meters. "
    return instructions


def find_path_astar(grid, start, end):
    """
    Finds the shortest path from start to end on a grid using the A* search algorithm.
    Args:
        grid: A 2D numpy array representing the grid.
        start: A tuple (row, col) representing the starting position.
        end: A tuple (row, col) representing the ending position.
    Returns:
        A list of tuples representing the path from start to end, or None if no path is found.
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
def heuristic(a, b):
    """Calculates the Manhattan distance between two points."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
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


def transform_bboxes_to_grid(image, parking_spots, result, yolo_results, grid_size):
    """
    Transforms bounding boxes to a 2D grid representation.
    Args:
        image_path: Path to the image.
        parking_spots: A list of dictionaries, where each dictionary represents a bounding box
                       with 'x', 'y', 'h', 'w', and 'id' keys.
        grid_size: The size of the grid (e.g., 10x10).
    Returns:
        A 2D numpy array representing the grid, with each cell containing the ID of the
        parking spot if it intersects the cell, or 0 otherwise.
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
        # Assign YOLO class label to grid cells (adjust as needed)
        for i in range(start_row, end_row + 1):
            for j in range(start_col, end_col + 1):
                grid[i, j] = 1  # Add 2 to distinguish from parking spot IDs and 0
    print("Grid values:", np.unique(grid))
    return grid


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


class ParkingSpotDataset(Dataset):
    def __init__(self, image_path, parking_spots, transform=None):
        self.image_path = image_path
        self.parking_spots = parking_spots
        self.transform = transform
        # Check if image_path is a path or an image
        if isinstance(image_path, str) and os.path.exists(image_path):
            self.image = cv2.imread(image_path)
        else:
            # Assume image_path is already an image (e.g., NumPy array)
            self.image = image_path
    def __len__(self):
        return len(self.parking_spots)
    def __getitem__(self, idx):
        spot = self.parking_spots[idx]
        x, y, h, w, spot_id = spot['x'], spot['y'], spot['h'], spot['w'], spot['id']
        spot_image = self.image[y:y+h, x:x+w]
        # Convert the NumPy array to a PIL Image
        spot_image_pil = Image.fromarray(cv2.cvtColor(spot_image, cv2.COLOR_BGR2RGB))
        if self.transform:
            spot_image_pil = self.transform(spot_image_pil)
        return spot_image_pil, spot_id

   
def Perform_template_matching(img, template):
    # Convert the images to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    # Find the location of the best match
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc

    # Get the width and height of the template
    h, w = template.shape[:2]

    # Calculate the bottom-right corner of the matched region
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # Crop the matched region
    cropped_img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    cropped_img = cv2.resize(cropped_img, (template.shape[1], template.shape[0]))

    return cropped_img


def display_grid_as_image(grid):
    plt.imshow(grid, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.show()


def display_path_on_grid(grid, path):


    plt.imshow(grid, cmap='viridis', interpolation='nearest')
    plt.colorbar()


    if path:
        path_x, path_y = zip(*path)  # Unpack path tuples
        plt.plot(path_y, path_x, marker='o', color='red', linestyle='-') # Plot path


    plt.show()


#prompt: call all pre-post procceses in one
def run_all_processes(img, template, model2, model):


  transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),          # Convert to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet mean and std
                         std=[0.229, 0.224, 0.225])])




  parking_spots = []
  parking_spots.append({'x': 110, 'y': 45, 'h': 30, 'w': 80, 'id': 1})
  parking_spots.append({'x': 95, 'y': 80, 'h': 30, 'w': 80, 'id': 2})
  parking_spots.append({'x': 85, 'y': 115, 'h': 35, 'w': 85, 'id': 3})
  parking_spots.append({'x': 65, 'y': 145, 'h': 35, 'w': 85, 'id': 4})
  parking_spots.append({'x': 50, 'y': 185, 'h': 35, 'w': 85, 'id': 5})
  parking_spots.append({'x': 35, 'y': 220, 'h': 35, 'w': 85, 'id': 6})



  # Call functions in the desired order
  cropped_img = Perform_template_matching(img, template)
  dataset = ParkingSpotDataset(cropped_img, parking_spots, transform=transform)
  results_df = process_parking_spots(dataset, model)
  yolo_results = model2.predict(cropped_img)


  grid_size = (500, 500)  # Example grid size
  grid = transform_bboxes_to_grid(cropped_img, parking_spots, results_df, yolo_results, grid_size)
  display_grid_as_image(grid)


  start = (300, 0)
  target_coordinates = np.where(grid == 2)
  if target_coordinates[0].size > 0:
    end = (target_coordinates[0][0], target_coordinates[1][0])
    path = find_path_astar(grid, start, end)
    if path:
      display_path_on_grid(grid, path)
      instructions = path_to_instructions(path, 0.04)
      return(instructions)


# Call the function to execute all processes


if __name__ == "__main__":
   ParkingApp().run()



