"""
Simple test script for ECG image processing functionality.
This script creates a synthetic ECG image and tests the image processing function.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Import the process_ecg_image function from app.py
from app import process_ecg_image

def create_test_ecg_image():
    """Create a test ECG image with a synthetic signal"""
    # Create a synthetic ECG signal
    np.random.seed(42)
    t = np.linspace(0, 10, 1000)
    ecg = np.sin(2*np.pi*t) + 0.25*np.sin(2*np.pi*50*t) + np.random.normal(0, 0.1, 1000)
    
    # Create an image with the ECG signal
    height, width = 500, 1000
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Scale the signal to fit in the image
    ecg_scaled = (ecg - np.min(ecg)) / (np.max(ecg) - np.min(ecg)) * (height * 0.8)
    ecg_scaled = height // 2 - ecg_scaled
    
    # Draw the ECG signal
    for i in range(len(ecg_scaled) - 1):
        x1, y1 = i, int(ecg_scaled[i])
        x2, y2 = i + 1, int(ecg_scaled[i + 1])
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 2)
    
    # Draw grid lines
    for i in range(0, width, 50):
        cv2.line(img, (i, 0), (i, height), (200, 200, 200), 1)
    for i in range(0, height, 50):
        cv2.line(img, (0, i), (width, i), (200, 200, 200), 1)
    
    # Save the image
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/img')
    output_path = os.path.join(output_dir, 'synthetic_ecg.png')
    cv2.imwrite(output_path, img)
    
    print(f"Synthetic ECG image created at {output_path}")
    
    # Test processing the synthetic image
    ecg_signal = process_ecg_image(output_path)
    
    # Plot the extracted signal
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.title('Synthetic ECG Image')
    img = cv2.imread(output_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis('off')
    
    plt.subplot(2, 1, 2)
    plt.title('Extracted ECG Signal')
    plt.plot(ecg_signal, 'b-')
    plt.grid(True)
    plt.tight_layout()
    
    # Save the result
    plt.savefig(os.path.join(output_dir, 'synthetic_ecg_processed.png'))
    plt.close()
    
    print(f"Synthetic ECG processing test successful! Result saved to {os.path.join(output_dir, 'synthetic_ecg_processed.png')}")

if __name__ == "__main__":
    print("Testing ECG image processing functionality...")
    create_test_ecg_image()
    print("Testing completed!") 