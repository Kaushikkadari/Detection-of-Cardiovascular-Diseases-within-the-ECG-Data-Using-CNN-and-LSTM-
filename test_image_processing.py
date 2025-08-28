"""
Test script for ECG image processing functionality.
This script demonstrates how to extract ECG signals from images.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from cairosvg import svg2png
import tempfile

# Add the parent directory to the path to import from app.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app import process_ecg_image

def test_svg_image():
    """Test processing an SVG image of an ECG"""
    # Path to the sample SVG image
    svg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/img/sample_ecg.svg')
    
    # Convert SVG to PNG for processing
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        svg2png(url=svg_path, write_to=tmp_file.name)
        png_path = tmp_file.name
    
    # Process the image
    try:
        ecg_signal = process_ecg_image(png_path)
        
        # Plot the extracted signal
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.title('Original Image')
        img = cv2.imread(png_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.axis('off')
        
        plt.subplot(2, 1, 2)
        plt.title('Extracted ECG Signal')
        plt.plot(ecg_signal, 'b-')
        plt.grid(True)
        plt.tight_layout()
        
        # Save the result
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/img')
        plt.savefig(os.path.join(output_dir, 'image_processing_test.png'))
        plt.close()
        
        print(f"Test successful! Result saved to {os.path.join(output_dir, 'image_processing_test.png')}")
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
    
    # Clean up the temporary file
    os.unlink(png_path)

def create_test_ecg_image():
    """Create a test ECG image with a synthetic signal"""
    # Create a synthetic ECG signal
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
    test_svg_image()
    create_test_ecg_image()
    print("Testing completed!") 