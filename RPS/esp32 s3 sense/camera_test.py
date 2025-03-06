import camera
import network
import time
from machine import Pin

# Initialize camera with ESP32-S3 Sense specific configuration
camera.init(0, d0=5, d1=14, d2=4, d3=15, d4=18, d5=23, d6=36, d7=39,
            format=camera.JPEG, framesize=camera.FRAME_SVGA,
            xclk_freq=camera.XCLK_20MHz, href=47, vsync=48,
            reset=-1, sioc=12, siod=13, xclk=10, pclk=11,
            fb_location=camera.PSRAM)

# Capture image buffer
buf = camera.capture()

# Save to file system
with open('image.jpg', 'wb') as f:
    f.write(buf)

# Cleanup resources
camera.deinit()

# Optional: Display image info
print(f"Captured image size: {len(buf)} bytes")
print("Image saved as image.jpg")
