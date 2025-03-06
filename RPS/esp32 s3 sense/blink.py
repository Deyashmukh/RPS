from machine import Pin
from time import sleep

# For XIAO ESP32S3 Sense built-in LED (GPIO21)
led = Pin(21, Pin.OUT)  # Use Pin(2) for other ESP32-S3 boards

while True:
    led.on()     # Turn LED on (might be inverted - see note below)
    sleep(0.5)
    led.off()    # Turn LED off
    sleep(0.5)
