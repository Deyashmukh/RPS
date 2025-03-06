import camera
import time
import json
import network
import gc
from microlite.core import TFLiteInterpreter
import socket

class HTTPServer:
    def __init__(self, port=80):
        """Initialize HTTP server on the given port"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Add socket option to allow reuse of local addresses
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Try to bind to the port, if it fails try an alternative port
        try:
            self.socket.bind(('0.0.0.0', port))
            self.port = port
        except OSError as e:
            if e.args[0] == 112:  # EADDRINUSE
                # Try an alternative port
                alt_port = 8080
                print(f'Port {port} in use, trying port {alt_port}')
                self.socket.bind(('0.0.0.0', alt_port))
                self.port = alt_port
            else:
                raise
                
        self.socket.listen(1)
        self.latest_results = {'timestamp': 0, 'probabilities': [0.33, 0.34, 0.33], 'predicted_class': 0}
        print(f'HTTP server started on port {self.port}')
        
    def update_results(self, results):
        """Update the latest classification results"""
        self.latest_results = results
        
    def handle_client(self):
        """Check for and handle any HTTP client connections"""
        # Set socket to non-blocking mode
        self.socket.setblocking(False)
        
        try:
            # Check for connection
            client, addr = self.socket.accept()
            print(f'HTTP client connected from: {addr}')
            
            # Set back to blocking for this client
            client.setblocking(True)
            
            # Get the request
            request = client.recv(1024).decode()
            
            # Parse the request to get the path
            request_line = request.split('\n')[0]
            path = request_line.split(' ')[1]
            
            # Handle different paths
            if path == '/classify' or path == '/data' or path == '/api':
                # Serve JSON classification results
                response = json.dumps(self.latest_results)
                header = 'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\n\r\n'
                client.send((header + response).encode())
                
            elif path == '/jpg' or path == '/capture':
                # Capture and serve a JPEG image
                try:
                    img = camera.capture()
                    header = 'HTTP/1.1 200 OK\r\nContent-Type: image/jpeg\r\nAccess-Control-Allow-Origin: *\r\n\r\n'
                    client.send(header.encode())
                    client.send(img)
                except:
                    # Send error response
                    client.send('HTTP/1.1 500 Internal Server Error\r\n\r\n'.encode())
                    
            else:
                # Default: simple status page
                response = '<html><body><h1>ESP32-S3 Classifier</h1>'
                response += '<p>Status: Running</p>'
                response += f'<p>Latest class: {self.latest_results["predicted_class"]}</p>'
                response += '<p>API Endpoints: <ul>'
                response += '<li><a href="/classify">/classify</a> - JSON classification results</li>'
                response += '<li><a href="/jpg">/jpg</a> - Latest camera image</li>'
                response += '</ul></p>'
                response += '</body></html>'
                
                header = 'HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n'
                client.send((header + response).encode())
                
            # Close the connection
            client.close()
            
        except OSError as e:
            # No connection available (non-blocking socket)
            if e.args[0] == 11:  # EAGAIN error
                pass
            else:
                print(f'HTTP server error: {e}')
        except Exception as e:
            print(f'Error handling HTTP client: {e}')
            
    def __del__(self):
        """Destructor to ensure socket is closed properly"""
        try:
            self.socket.close()
            print('HTTP server socket closed')
        except:
            pass

class RealTimeClassifier:
    def __init__(self, ssid='iPhone', password='123456789'):
        # Configuration from the original EfficientNetB0 model
        self.class_names = ['rock', 'paper', 'scissors']
        self.input_size = 224  # EfficientNetB0 uses 224x224 input
        
        # Initialize components
        self.setup_wifi(ssid, password)
        self.setup_camera()
        self.setup_model()
        
        # Create HTTP server
        self.http_server = HTTPServer()
        
    def setup_wifi(self, ssid, password):
        """Setup WiFi connection"""
        print('Connecting to WiFi...')
        self.wlan = network.WLAN(network.STA_IF)
        self.wlan.active(True)
        self.wlan.connect(ssid, password)
        
        # Wait for connection
        while not self.wlan.isconnected():
            time.sleep(1)
        print('WiFi connected!')
        print('Device IP:', self.wlan.ifconfig()[0])
        
    def setup_camera(self):
        """Initialize camera with settings appropriate for EfficientNetB0 input"""
        print('Initializing camera...')
        try:
            # Initialize camera with no arguments
            camera.init()
            
            try:
                # Try to set higher resolution to match 224x224 target better
                camera.framesize(4)  # Higher resolution if available (VGA or higher)
                camera.pixformat(0)  # RGB565
                camera.quality(10)  # High quality
            except:
                print('Could not set optimal camera parameters, using defaults')
            
            # Let camera warm up
            for _ in range(3):
                try:
                    frame = camera.capture()
                    time.sleep_ms(100)
                except:
                    print('Issue capturing test frame')
                
            print('Camera initialized for EfficientNet model input')
        except Exception as e:
            print(f'Camera init failed: {e}')
            raise
        
    def setup_model(self):
        """Load TFLite model (converted from EfficientNetB0-based model)"""
        print('Loading TFLite model...')
        try:
            import os
            # Check if file exists
            print('Checking for model.tflite...')
            try:
                files = os.listdir()
                print('Files in current directory:', files)
            except:
                print('Could not list files')
                
            # Load the model
            print('Attempting to load EfficientNet-based TFLite model...')
            self.interpreter = TFLiteInterpreter('model.tflite')
            print('Model loaded successfully!')
            
            # Print model information if available
            try:
                print('Model information:')
                input_shape = self.interpreter.get_input_shape()
                output_shape = self.interpreter.get_output_shape()
                print('Input shape:', input_shape)
                print('Output shape:', output_shape)
                
                # Verify shape matches expectations
                expected_input = 224*224*3
                if input_shape and len(input_shape) > 0:
                    if input_shape[0] != expected_input:
                        print(f"Warning: Model input shape {input_shape} doesn't match expected EfficientNet input shape {expected_input}")
            except:
                print('Could not get model information')
                
        except Exception as e:
            print(f'Model loading failed: {e}')
            raise
    
    def preprocess_image(self, frame):
        """Preprocess image matching EfficientNetB0 preprocessing used in the original model"""
        if not isinstance(frame, (bytearray, bytes)):
            return [0.0] * (224*224*3)
            
        print("Processing image of size:", len(frame))
        
        # Extract data and resize to 224x224 (EfficientNetB0 input size)
        processed_data = []
        step = max(1, len(frame) // (224*224*3))
        
        for i in range(0, len(frame) - 3, step):
            # Normalize to 0-1 range (EfficientNet expects values in [0,1])
            r = frame[i] / 255.0
            g = frame[i+1] / 255.0
            b = frame[i+2] / 255.0
            processed_data.extend([r, g, b])
        
        # Ensure exact size matches EfficientNetB0 input
        if len(processed_data) < 224*224*3:
            processed_data.extend([0.0] * (224*224*3 - len(processed_data)))
        else:
            processed_data = processed_data[:224*224*3]
            
        return processed_data
        
    def run_inference(self, input_data):
        """Run inference using the TFLite model converted from EfficientNetB0"""
        print("Running inference with EfficientNet-based model...")
        try:
            # Set input to the model
            self.interpreter.set_input(input_data)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output (softmax probabilities from the final dense layer)
            output = self.interpreter.get_output()
            print("Inference complete")
            print("Raw output (should be 3 class probabilities):", output)
            
            # Verify we have the expected number of classes
            if len(output) != len(self.class_names):
                print(f"Warning: Expected {len(self.class_names)} outputs, got {len(output)}")
            
            return output
        except Exception as e:
            print(f"Inference error: {e}")
            # Return equal probabilities as fallback
            return [0.33, 0.33, 0.34]
        
    def process_result(self, output):
        """Process the raw model output into a standardized format"""
        result = {
            'timestamp': time.ticks_ms(),
            'probabilities': output[:3] if len(output) >= 3 else [0.33, 0.33, 0.34],
            'predicted_class': output.index(max(output[:3])) if len(output) >= 3 else 0
        }
        return result
        
    def run_classification(self):
        """Main classification loop"""
        print('Starting classification loop...')
        last_class = -1  # Track last prediction to show changes
        
        try:
            while True:
                # Capture frame
                try:
                    frame = camera.capture()
                    if not frame:
                        print('Failed to capture frame')
                        time.sleep_ms(100)
                        continue
                except Exception as e:
                    print(f"Camera capture error: {e}")
                    time.sleep_ms(100)
                    continue
                
                # Preprocess image
                input_data = self.preprocess_image(frame)
                if not input_data:
                    time.sleep_ms(50)
                    continue
                
                # Run inference using only the model
                output = self.run_inference(input_data)
                
                # Process results
                result = self.process_result(output)
                
                # Only log when class changes
                if result['predicted_class'] != last_class:
                    print("New prediction:", result)
                    class_name = self.class_names[result['predicted_class']]
                    confidence = max(result['probabilities']) * 100
                    print(f"Class: {class_name}, Confidence: {confidence:.1f}%")
                    last_class = result['predicted_class']
                
                # Update HTTP server with latest results
                self.http_server.update_results(result)
                
                # Handle any pending HTTP requests
                self.http_server.handle_client()
                
                # Memory management
                gc.collect()
                time.sleep_ms(50)
                
        except Exception as e:
            print('Error in main loop:', e)
        finally:
            camera.deinit()
            # Ensure socket is properly closed
            try:
                self.http_server.socket.close()
            except:
                pass

# Run the classifier
if __name__ == '__main__':
    try:
        # Replace with your WiFi credentials
        classifier = RealTimeClassifier(ssid='iPhone', password='123456789')
        classifier.run_classification()
    except Exception as e:
        print(f'Error: {e}')
        try:
            camera.deinit()  # Always cleanup camera
        except:
            pass