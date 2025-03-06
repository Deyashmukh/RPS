import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import time
import threading
import requests
import json
import sys
from PIL import Image, ImageTk
from io import BytesIO

class ESP32Viewer:
    def __init__(self, esp32_ip='172.20.10.3'):
        # Setup variables
        self.esp32_ip = esp32_ip
        self.running = True
        self.last_update = time.time()
        self.classes = ['Rock', 'Paper', 'Scissors']  # Rock-Paper-Scissors classes
        self.latest_data = None
        self.latest_image = None
        self.last_prediction = None
        
        # Setup main window
        self.root = tk.Tk()
        self.root.title(f"ESP32-S3 Rock-Paper-Scissors - {esp32_ip}")
        self.root.geometry("1000x700")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Build UI
        self.setup_ui()
        
        # Start polling thread
        self.poll_thread = threading.Thread(target=self.poll_data, daemon=True)
        self.poll_thread.start()
        
    def setup_ui(self):
        """Set up the user interface"""
        # Top control panel
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(fill=tk.X)
        
        # Status label
        self.status_var = tk.StringVar(value="Not connected")
        status_label = ttk.Label(control_frame, text="Status:")
        status_label.pack(side=tk.LEFT, padx=(0, 5))
        self.status_indicator = ttk.Label(control_frame, textvariable=self.status_var)
        self.status_indicator.pack(side=tk.LEFT, padx=(0, 10))
        
        # Exit button (prominent red)
        exit_button = tk.Button(
            control_frame, 
            text="EXIT", 
            command=self.on_close,
            bg="red", 
            fg="white",
            font=("Arial", 12, "bold")
        )
        exit_button.pack(side=tk.RIGHT)
        
        # Main content area - split into left/right panels
        content_frame = ttk.Frame(self.root)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for camera
        left_frame = ttk.LabelFrame(content_frame, text="Camera Feed")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.camera_label = ttk.Label(left_frame, text="Waiting for camera feed...")
        self.camera_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Right panel for classification
        right_frame = ttk.LabelFrame(content_frame, text="Classification Results")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Prediction label
        self.prediction_var = tk.StringVar(value="Waiting for results...")
        self.prediction_label = ttk.Label(
            right_frame,
            textvariable=self.prediction_var,
            font=("Arial", 16, "bold")
        )
        self.prediction_label.pack(pady=10)
        
        # Confidence chart
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Initialize plot
        self.bars = self.ax.bar(self.classes, [0] * len(self.classes))
        self.ax.set_ylim(0, 1)
        self.ax.set_title('Classification Confidence')
        self.colors = plt.cm.viridis(np.linspace(0, 1, len(self.classes)))
        for bar, color in zip(self.bars, self.colors):
            bar.set_color(color)
        
        # Log area
        log_frame = ttk.LabelFrame(self.root, text="Log")
        log_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.log_text = tk.Text(log_frame, height=5)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Scrollbar for log
        scrollbar = ttk.Scrollbar(self.log_text, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)
        
    def log(self, message):
        """Add message to log with timestamp"""
        if not self.running:
            return
            
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        # Add to log widget
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        
        # Also print to console
        print(log_entry, end="")
        
    def poll_data(self):
        """Poll ESP32 for data at regular intervals"""
        while self.running:
            try:
                # Try to get both image and data
                self.get_camera_image()
                self.get_classification_data()
                
                # Update status
                if (time.time() - self.last_update) > 5:
                    self.status_var.set("No data received (timeout)")
                    
                # Slight delay to avoid hammering the server
                time.sleep(0.2)
                
            except Exception as e:
                self.log(f"Error during polling: {str(e)}")
                time.sleep(1)  # Back off on error
                
    def get_camera_image(self):
        """Try to get camera image from ESP32"""
        try:
            # Try the capture endpoint
            url = f"http://{self.esp32_ip}/jpg"
            response = requests.get(url, timeout=1.0)
            
            if response.status_code == 200 and response.headers.get('content-type', '').startswith('image/'):
                # Process image
                image = Image.open(BytesIO(response.content))
                self.update_camera_feed(image)
                
                if not self.latest_image:
                    self.log(f"Received first camera image")
                    
                self.latest_image = True
                self.last_update = time.time()
                self.status_var.set("Connected (receiving data)")
                return True
                
            return False
                
        except Exception as e:
            # Don't log every failure to avoid spamming
            return False
            
    def get_classification_data(self):
        """Try to get classification data from ESP32"""
        try:
            # Try the classify endpoint
            url = f"http://{self.esp32_ip}/classify"
            response = requests.get(url, timeout=1.0)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    self.process_data(data)
                    self.last_update = time.time()
                    return True
                except:
                    return False
                    
            return False
                
        except Exception as e:
            # Don't log every failure to avoid spamming
            return False
            
    def process_data(self, data):
        """Process classification data"""
        if not self.running:
            return
            
        try:
            # Store the data
            self.latest_data = data
            
            # Check if data has the expected format
            if 'probabilities' in data and 'predicted_class' in data:
                probabilities = data['probabilities']
                
                # Handle nested list (seen in some outputs)
                if isinstance(probabilities, list) and len(probabilities) > 0 and isinstance(probabilities[0], list):
                    probabilities = probabilities[0]
                    
                # Make sure we have at least 3 probabilities for rock-paper-scissors
                if len(probabilities) < 3:
                    probabilities = probabilities + [0] * (3 - len(probabilities))
                elif len(probabilities) > 3:
                    probabilities = probabilities[:3]
                    
                # Update bars
                for bar, prob in zip(self.bars, probabilities):
                    bar.set_height(prob)
                        
                # Redraw chart
                self.canvas.draw()
                
                # Update prediction text
                pred_class = data['predicted_class']
                if isinstance(pred_class, int) and 0 <= pred_class < len(self.classes):
                    conf = probabilities[pred_class]
                    self.prediction_var.set(f"Prediction: {self.classes[pred_class]} ({conf:.1%})")
                    
                    # Log if prediction changed
                    if self.last_prediction != pred_class:
                        self.log(f"New prediction: {self.classes[pred_class]} ({conf:.1%})")
                        self.last_prediction = pred_class
                else:
                    self.prediction_var.set(f"Prediction: Unknown ({pred_class})")
                    
        except Exception as e:
            self.log(f"Error processing data: {str(e)}")
            
    def update_camera_feed(self, image):
        """Update the camera feed with a new image"""
        if not self.running:
            return
            
        try:
            # Resize image while maintaining aspect ratio
            max_width = 480
            max_height = 360
            width, height = image.size
            
            if width > max_width or height > max_height:
                scale = min(max_width / width, max_height / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = image.resize((new_width, new_height), Image.LANCZOS)
                
            # Convert to tkinter format
            photo = ImageTk.PhotoImage(image)
            
            # Update label
            self.camera_label.config(image=photo, text="")
            self.camera_label.image = photo  # Keep reference
            
        except Exception as e:
            self.log(f"Error updating camera feed: {str(e)}")
            
    def on_close(self):
        """Handle window close event"""
        self.log("Shutting down...")
        self.running = False
        
        try:
            self.root.destroy()
        except:
            pass
            
        sys.exit(0)
        
    def run(self):
        """Start the application"""
        # Start periodic UI update
        self.update_ui()
        self.root.mainloop()
        
    def update_ui(self):
        """Update UI elements periodically"""
        if self.running:
            # Process UI events and schedule next update
            self.root.after(100, self.update_ui)

# Main entry point
if __name__ == "__main__":
    # Get ESP32 IP from command line or use default
    esp32_ip = '100.110.206.99'  # Default
    
    if len(sys.argv) > 1:
        esp32_ip = sys.argv[1]
    
    print(f"Starting ESP32-S3 Rock-Paper-Scissors Viewer - connecting to {esp32_ip}")
    print("Press Ctrl+C or use the EXIT button to quit")
    
    viewer = ESP32Viewer(esp32_ip)
    viewer.run()