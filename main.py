import torch
import autoit
import time
from PIL import ImageGrab

# Load your YOLO model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

def perform_inference():
    # Take a screenshot
    screenshot = ImageGrab.grab()
    screenshot = screenshot.convert("RGB")

    # Perform inference
    results = model(screenshot)

    # Filter detections based on confidence score (column 4 is confidence)
    detections = results.xyxy[0]
    for detection in detections:
        confidence = detection[4]  # The confidence score is the 5th element
        if confidence >= 0.5:
            return True

    return False

def click_at_coordinates():
    # Move the mouse to coordinates (950, 980) and click
    autoit.mouse_click(x=950, y=980)

def main_loop():
    last_detection_time = time.time()

    while True:
        detected = perform_inference()

        if detected:
            click_at_coordinates()
            last_detection_time = time.time()  # Reset the timer on detection
            time.sleep(2)  # Stop inference for 2 seconds
            for _ in range(3):  # Click three times after detection
                time.sleep(0.05)
                click_at_coordinates()
        else:
            if time.time() - last_detection_time > 15:  # If no detection for 15 seconds
                click_at_coordinates()
                last_detection_time = time.time()  # Reset the timer after forced click

        time.sleep(0.7)  # Perform inference every 0.7 seconds


if __name__ == "__main__":
    main_loop()
