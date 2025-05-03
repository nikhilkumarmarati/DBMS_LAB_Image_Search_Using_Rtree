import torch
import torchvision.transforms as transforms
import tkinter as tk
from tkinter import filedialog, ttk, font
from PIL import Image, ImageTk
import numpy as np
import cv2
import rtree
import random
import time

# Load the trained PyTorch model
model = torch.jit.load("dinov2_pca_25d.pt")
model.eval()

# Image preprocessing transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize R-tree object
rt = rtree.Rtree()
rt.load("rtree.bin")

# STL-10 Binary dataset paths
train_file_path = "stl10_binary/train_X.bin"
test_file_path = "stl10_binary/test_X.bin"

# Total images in each dataset
NUM_TRAIN_IMAGES = 5000
NUM_TEST_IMAGES = 8000

class ImageSearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Search Engine - R-Tree Search")
        self.root.geometry("1200x750")
        
        # Color palette
        self.colors = {
            "primary": "#2c3e50",       # Dark blue/slate for headers
            "secondary": "#3498db",      # Blue for highlights and accents
            "accent": "#e74c3c",         # Red for important elements
            "light_bg": "#ecf0f1",       # Light gray for backgrounds
            "dark_bg": "#34495e",        # Darker blue for contrast areas
            "text_light": "#ffffff",     # White text
            "text_dark": "#2c3e50",      # Dark text
            "success": "#2ecc71",        # Green for success indicators
            "neutral": "#95a5a6"         # Neutral gray
        }
        
        # Configure the root window background
        self.root.configure(bg=self.colors["light_bg"])
        
        # Setup custom fonts
        self.setup_fonts()
        
        # Build the UI
        self.setup_ui()
        
    def setup_fonts(self):
        # Define custom fonts
        self.fonts = {
            "title": font.Font(family="Helvetica", size=18, weight="bold"),
            "header": font.Font(family="Helvetica", size=14, weight="bold"),
            "subheader": font.Font(family="Helvetica", size=12, weight="bold"),
            "body": font.Font(family="Helvetica", size=10),
            "small": font.Font(family="Helvetica", size=9),
            "button": font.Font(family="Helvetica", size=11, weight="bold")
        }
    
    def setup_ui(self):
        # Main container
        main_frame = tk.Frame(self.root, bg=self.colors["light_bg"], padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Application header
        header_frame = tk.Frame(main_frame, bg=self.colors["light_bg"])
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        # App title with a splash of color
        title_label = tk.Label(header_frame, 
                              text="Rtree KNN Image Search", 
                              font=self.fonts["title"],
                              fg=self.colors["primary"],
                              bg=self.colors["light_bg"])
        title_label.pack(side=tk.LEFT)
        
        # Subtitle with description
        subtitle_frame = tk.Frame(main_frame, bg=self.colors["light_bg"])
        subtitle_frame.pack(fill=tk.X, pady=(0, 15))
        
        subtitle = tk.Label(subtitle_frame, 
                           text="Find visually similar images using R-Tree and deep learning features", 
                           font=self.fonts["body"],
                           fg=self.colors["text_dark"],
                           bg=self.colors["light_bg"])
        subtitle.pack(anchor=tk.W)
        
        # Create a horizontal separator
        separator = ttk.Separator(main_frame, orient='horizontal')
        separator.pack(fill=tk.X, pady=10)
        
        # Action buttons with custom styling
        btn_frame = tk.Frame(main_frame, bg=self.colors["light_bg"])
        btn_frame.pack(fill=tk.X, pady=10)
        
        # Custom buttons
        upload_btn = tk.Button(btn_frame, 
                              text="Upload Image", 
                              font=self.fonts["button"],
                              bg=self.colors["secondary"],
                              fg=self.colors["text_light"],
                              padx=15, pady=8,
                              relief=tk.RAISED,
                              command=self.select_image)
        upload_btn.pack(side=tk.LEFT, padx=5)
        
        random_btn = tk.Button(btn_frame, 
                              text="Random Test Image", 
                              font=self.fonts["button"],
                              bg=self.colors["accent"],
                              fg=self.colors["text_light"],
                              padx=15, pady=8,
                              relief=tk.RAISED,
                              command=self.load_random_test_image)
        random_btn.pack(side=tk.LEFT, padx=5)
        
        # Status indicator with custom style
        self.status_frame = tk.Frame(main_frame, bg=self.colors["light_bg"])
        self.status_frame.pack(fill=tk.X, pady=10)
        
        self.status_label = tk.Label(self.status_frame, 
                                    text="Ready to search - Select an image to begin", 
                                    font=self.fonts["subheader"],
                                    fg=self.colors["secondary"],
                                    bg=self.colors["light_bg"])
        self.status_label.pack(side=tk.LEFT)
        
        # Create frames for images with better visual styling
        self.image_display_frame = tk.Frame(main_frame, bg=self.colors["light_bg"])
        self.image_display_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Configure grid weights for proper resizing
        self.image_display_frame.grid_columnconfigure(0, weight=1)  # Query image column
        self.image_display_frame.grid_columnconfigure(1, weight=3)  # Results column
        
        # Query image section with border and better styling
        query_frame = tk.LabelFrame(self.image_display_frame, 
                                   text="Query Image", 
                                   font=self.fonts["subheader"],
                                   fg=self.colors["primary"],
                                   bg=self.colors["light_bg"],
                                   padx=10, pady=10)
        query_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # Canvas with border and background color
        self.query_canvas = tk.Canvas(query_frame, 
                                     width=250, height=250, 
                                     bg=self.colors["light_bg"], 
                                     highlightbackground=self.colors["secondary"],
                                     highlightthickness=2)
        self.query_canvas.pack(padx=10, pady=10)
        
        # Add a placeholder text on the canvas
        self.query_canvas.create_text(125, 125, 
                                     text="Your image will appear here", 
                                     fill=self.colors["neutral"],
                                     font=self.fonts["subheader"])
        
        # Results section with improved styling - MODIFIED for horizontal layout
        results_frame = tk.LabelFrame(self.image_display_frame, 
                                     text="Nearest Neighbors", 
                                     font=self.fonts["subheader"],
                                     fg=self.colors["primary"],
                                     bg=self.colors["light_bg"],
                                     padx=10, pady=10)
        results_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        # Create a frame for results with grid layout
        self.results_frame = tk.Frame(results_frame, bg=self.colors["light_bg"])
        self.results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Configure grid for the results (3 columns)
        for i in range(5):  # Configure columns for results
            self.results_frame.columnconfigure(i, weight=1, uniform="col")
        
        # Metrics section with custom styling
        self.metrics_frame = tk.LabelFrame(main_frame, 
                                         text="Search Metrics", 
                                         font=self.fonts["subheader"],
                                         fg=self.colors["primary"],
                                         bg=self.colors["light_bg"],
                                         padx=10, pady=10)
        self.metrics_frame.pack(fill=tk.X, pady=10)
        
        # Time display with icon-like indicator
        time_container = tk.Frame(self.metrics_frame, bg=self.colors["light_bg"])
        time_container.pack(fill=tk.X, expand=True)
        
        time_icon = tk.Label(time_container, 
                            text="⏱", 
                            font=self.fonts["header"],
                            fg=self.colors["text_dark"],
                            bg=self.colors["light_bg"])
        time_icon.pack(side=tk.LEFT, padx=(0, 5))
        
        self.time_label = tk.Label(time_container, 
                                  text="Search time: Waiting for search...", 
                                  font=self.fonts["body"],
                                  fg=self.colors["accent"],
                                  bg=self.colors["light_bg"])
        self.time_label.pack(side=tk.LEFT, padx=5)
        
        # Footer with application info
        footer_frame = tk.Frame(main_frame, bg=self.colors["light_bg"])
        footer_frame.pack(fill=tk.X, pady=(20, 0))
        
        footer_text = tk.Label(footer_frame, 
                              text="Visual Search Engine powered by PyTorch, R-Tree, and STL-10 Dataset", 
                              font=self.fonts["body"],
                              fg=self.colors["text_dark"],
                              bg=self.colors["light_bg"])
        footer_text.pack(side=tk.LEFT)
        
    def select_image(self):
        """Open file dialog, process the image, and render nearest images."""
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image files", ".jpg *.jpeg *.png"), ("All Files", ".*")]
        )
        if not file_path:
            return

        self.status_label.config(text="Processing image... Please wait")
        self.root.update()

        # Display the query image
        query_img = Image.open(file_path).convert("RGB")
        self.display_query_image(query_img)

        # Extract features & find nearest images
        start_time = time.time()
        query_point = self.extract_features(file_path)
        self.show_nearest_images(query_point)
        
        elapsed = time.time() - start_time
        self.time_label.config(text=f"Search time: {elapsed:.3f} seconds")
        self.status_label.config(text="Search completed successfully")

    def load_random_test_image(self):
        """Load a random image from test_X.bin and find its nearest neighbors."""
        self.status_label.config(text="Loading random test image... Please wait")
        self.root.update()
        
        image_id = random.randint(0, NUM_TEST_IMAGES - 1)  # Random test image
        
        start_time = time.time()
        query_point, test_image = self.extract_features_from_binary(image_id)
        
        # Convert to PIL image for display
        pil_img = Image.fromarray(test_image)
        self.display_query_image(pil_img)
        
        # Find nearest images
        self.show_nearest_images(query_point)
        
        elapsed = time.time() - start_time
        self.time_label.config(text=f"Search time: {elapsed:.3f} seconds")
        self.status_label.config(text=f"Random image #{image_id} loaded and searched successfully")

    def display_query_image(self, pil_img):
        """Display the query image in its dedicated area with better styling"""
        # Resize image for display
        pil_img = pil_img.resize((224, 224), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(pil_img)
        
        # Clear previous image and display new one
        self.query_canvas.delete("all")
        
        # Add a decorative border around the image
        self.query_canvas.create_rectangle(12, 12, 236, 236, 
                                         outline=self.colors["secondary"], 
                                         width=2)
        self.query_canvas.create_image(124, 124, anchor="center", image=img_tk)
        self.query_canvas.image = img_tk  # Keep reference
    
    def show_nearest_images(self, query_point):
        """Find and display the nearest images in a horizontal grid layout."""
        k = 5
        neighbors = rt.kNearestNeighbors(query_point, k)

        if not neighbors:
            self.status_label.config(text="No neighbors found!")
            return

        # Extract only the image IDs and distances
        image_ids = [int(n[0]) for n in neighbors]
        distances = [float(n[1]) for n in neighbors]
        
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        # Load images from STL-10 dataset and display in horizontal grid
        for i, (img_id, distance) in enumerate(zip(image_ids, distances)):
            try:
                img = self.load_image_by_id(img_id)
                # Display in grid layout: col=i (horizontal layout)
                self.display_result_image(img, i, img_id, distance, col=i)
            except ValueError as e:
                print(f"Skipping image {img_id}: {e}")

    def display_result_image(self, img, position, img_id, distance, col=0):
        """Display a result image in the horizontal grid layout"""
        # Create a card-like frame for each result
        result_frame = tk.Frame(self.results_frame, bg=self.colors["light_bg"],
                              highlightbackground=self.colors["neutral"],
                              highlightthickness=1, bd=0)
        result_frame.grid(row=0, column=col, padx=5, pady=5, sticky="nsew")
        
        # Make image cards more compact to fit more in view
        # Process image for display - smaller size for horizontal layout
        img_resized = cv2.resize(img, (130, 130), interpolation=cv2.INTER_LINEAR)
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_resized)
        img_tk = ImageTk.PhotoImage(pil_img)
        
        # Create a canvas for image with border
        img_canvas = tk.Canvas(result_frame, 
                             width=134, 
                             height=134, 
                             bg=self.colors["light_bg"],
                             highlightbackground=self.colors["neutral"],
                             highlightthickness=1)
        img_canvas.pack(padx=5, pady=5)
        
        # Display image in canvas
        img_canvas.create_image(67, 67, anchor="center", image=img_tk)
        img_canvas.image = img_tk  # Keep reference
        
        # Create info panel below the image (vertical stacking)
        info_frame = tk.Frame(result_frame, bg=self.colors["light_bg"])
        info_frame.pack(padx=5, pady=5, fill=tk.X)
        
        # Rank indicator
        rank_label = tk.Label(info_frame, 
                             text=f"Rank #{position+1}", 
                             font=self.fonts["subheader"],
                             fg=self.colors["primary"],
                             bg=self.colors["light_bg"])
        rank_label.pack(anchor="w")
        
        # Image ID
        id_label = tk.Label(info_frame, 
                           text=f"Image ID: {img_id}", 
                           font=self.fonts["small"],
                           fg=self.colors["text_dark"],
                           bg=self.colors["light_bg"])
        id_label.pack(anchor="w")
        
        # Distance formatted with color indicator based on value
        distance_color = self.get_distance_color(distance)
        distance_frame = tk.Frame(info_frame, bg=self.colors["light_bg"])
        distance_frame.pack(anchor="w")
        
        distance_label = tk.Label(distance_frame, 
                                 text=f"Distance: ", 
                                 font=self.fonts["small"],
                                 fg=self.colors["text_dark"],
                                 bg=self.colors["light_bg"])
        distance_label.pack(side=tk.LEFT)
        
        # Use a tk.Label to get colored text
        distance_value = tk.Label(distance_frame, 
                                 text=f"{distance:.4f}", 
                                 font=self.fonts["small"],
                                 fg=distance_color,
                                 bg=self.colors["light_bg"])
        distance_value.pack(side=tk.LEFT)
    
    def get_distance_color(self, distance):
        """Return color based on distance value (closer is better)"""
        if distance < 0.5:
            return self.colors["success"]  # Very close match - green
        elif distance < 1.0:
            return self.colors["secondary"]  # Good match - blue
        else:
            return self.colors["accent"]  # Poor match - red/orange
    
    def load_image_by_id(self, image_id, from_test=False):
        """Load an image from STL-10 dataset by its ID."""
        IMAGE_SIZE = 96 * 96 * 3  # Each image is 96x96x3
        file_path = test_file_path if from_test else train_file_path

        if from_test:
            file_index = image_id  # Direct index for test set
            if image_id >= NUM_TEST_IMAGES:
                raise ValueError("Test image ID out of range.")
        else:
            file_index = image_id
            if image_id >= NUM_TRAIN_IMAGES:
                raise ValueError("Train image ID out of range.")

        with open(file_path, 'rb') as f:
            f.seek(file_index * IMAGE_SIZE)
            raw_data = np.frombuffer(f.read(IMAGE_SIZE), dtype=np.uint8)

        image = raw_data.reshape(3, 96, 96).transpose(1, 2, 0)  # Convert to HWC
        return image

    def extract_features(self, image_path):
        """Load image, preprocess manually (like in C++), and extract features."""
        # Read image using OpenCV
        image = cv2.imread(image_path)

        # Resize to 224x224
        resized = cv2.resize(image, (224, 224))

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Convert to float32 and normalize to [0, 1]
        float_image = rgb_image.astype(np.float32) / 255.0

        # Manual normalization (ImageNet mean/std)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (float_image - mean) / std

        # Convert to tensor: shape [1, 3, 224, 224]
        tensor_image = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float()
        
         # Extract features
        with torch.no_grad():
            features = model(tensor_image)

        # Convert to list
        feature_vector = features.squeeze().tolist()

        # Normalize to unit length (L2)
        norm = np.linalg.norm(feature_vector)
        if norm > 0:
            feature_vector = [f / norm for f in feature_vector]

        return feature_vector

    def extract_features_from_binary(self, image_id):
        image = self.load_image_by_id(image_id, from_test=True)
        
        # Resize to 224x224
        resized = cv2.resize(image, (224, 224))

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Convert to float32 and normalize to [0, 1]
        float_image = rgb_image.astype(np.float32) / 255.0

        # Normalize using ImageNet mean/std
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (float_image - mean) / std
        
        # Convert to tensor
        tensor_image = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float()

        # Extract features
        with torch.no_grad():
            features = model(tensor_image)

        # Normalize feature vector
        feature_vector = features.squeeze().tolist()
        norm = np.linalg.norm(feature_vector)
        if norm > 0:
            feature_vector = [f / norm for f in feature_vector]

        return feature_vector,image


# Create and run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSearchApp(root)
    root.mainloop()