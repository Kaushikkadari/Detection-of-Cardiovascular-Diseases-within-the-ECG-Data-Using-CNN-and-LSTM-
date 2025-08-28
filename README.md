# CardioDetect: ECG Cardiovascular Disease Detection

A comprehensive web application for detecting cardiovascular diseases using ECG data through advanced deep learning models combining CNN and LSTM architectures.

## 🏥 Project Overview

CardioDetect is an intelligent system that analyzes ECG (Electrocardiogram) data to detect various cardiovascular diseases. The application uses state-of-the-art deep learning techniques combining Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks for accurate disease detection and classification.

## ✨ Features

- **ECG Analysis**: Upload and analyze ECG data files (.dat, .hea, .npy formats)
- **Image Processing**: Process ECG images with advanced computer vision techniques
- **Real-time Detection**: Instant cardiovascular disease detection results
- **Multiple Disease Classes**: Detects various cardiovascular conditions
- **User-friendly Interface**: Modern, responsive web interface
- **Model Training**: Interactive model training interface
- **Visualization**: Detailed results with graphs and charts
- **Batch Processing**: Support for multiple file uploads

## 🛠️ Technology Stack

- **Backend**: Python Flask
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Machine Learning**: TensorFlow/Keras, CNN-LSTM hybrid model
- **Data Processing**: NumPy, Pandas, Scikit-learn
- **Visualization**: Chart.js, Matplotlib
- **Image Processing**: OpenCV, PIL
- **Database**: File-based storage system

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/Kaushikkadari/Detection-of-Cardiovascular-Diseases-within-the-ECG-Data-Using-CNN-and-LSTM-.git
   cd Detection-of-Cardiovascular-Diseases-within-the-ECG-Data-Using-CNN-and-LSTM-
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the application**
   Open your web browser and navigate to `http://localhost:5000`

## 🚀 Usage

### 1. Home Page
- Overview of the application
- Quick access to all features
- System architecture information

### 2. ECG Analysis
1. Navigate to "Analyze ECG" section
2. Upload your ECG data file (.dat, .hea, or .npy format)
3. Click "Analyze" to process the data
4. View detailed results and predictions

### 3. Image Processing
1. Go to "Image Processing" section
2. Upload ECG images
3. Apply various processing techniques
4. View enhanced and processed images

### 4. Model Training
1. Access "Train Model" section
2. Configure training parameters
3. Start training process
4. Monitor training progress

### 5. Results
- View detailed analysis results
- Interactive charts and visualizations
- Disease probability scores
- Download results

## 📁 Project Structure

```
├── app.py                     # Main Flask application
├── requirements.txt           # Python dependencies
├── README.md                 # Project documentation
├── data/                     # Data storage directory
│   └── uploads/             # User uploaded files
├── models/                   # Trained models
│   ├── cnn_lstm_model.h5    # Pre-trained model
│   └── cnn_lstm_model.py    # Model architecture
├── static/                   # Static assets
│   ├── css/
│   │   └── style.css        # Custom stylesheets
│   ├── js/
│   │   └── main.js          # JavaScript functionality
│   └── img/                 # Images and icons
├── templates/               # HTML templates
│   ├── base.html           # Base template
│   ├── index.html          # Home page
│   ├── upload.html         # File upload page
│   ├── results.html        # Results display page
│   ├── image_processing.html # Image processing page
│   ├── train.html          # Model training page
│   └── about.html          # About page
└── test_*.py               # Test files
```

## 🧠 Model Architecture

The system uses a hybrid CNN-LSTM architecture:

- **Convolutional Layers**: Extract spatial features from ECG signals
- **LSTM Layers**: Capture temporal dependencies in the data
- **Dense Layers**: Final classification layers
- **Dropout**: Regularization to prevent overfitting

## 📊 Supported File Formats

- **ECG Data**: .dat, .hea (PhysioNet format)
- **Processed Data**: .npy (NumPy arrays)
- **Images**: .png, .jpg, .jpeg
- **Models**: .h5 (Keras model format)

## 🔧 Configuration

The application can be configured through environment variables:

- `FLASK_ENV`: Set to 'development' for debug mode
- `UPLOAD_FOLDER`: Specify upload directory path
- `MAX_CONTENT_LENGTH`: Set maximum file upload size

## 🧪 Testing

Run the test files to verify functionality:

```bash
python test_image_only.py
python test_image_processing.py
```

## 📈 Performance

- **Accuracy**: High accuracy in cardiovascular disease detection
- **Speed**: Real-time processing for standard ECG files
- **Scalability**: Designed to handle multiple concurrent users

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Kadari Kaushik**
- GitHub: [@Kaushikkadari](https://github.com/Kaushikkadari)
- Project: [CardioDetect](https://github.com/Kaushikkadari/Detection-of-Cardiovascular-Diseases-within-the-ECG-Data-Using-CNN-and-LSTM-)

## 🙏 Acknowledgments

- PhysioNet for ECG datasets
- TensorFlow and Keras communities
- Bootstrap for UI components
- Chart.js for visualization

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Contact: kadarkaushik078@gmail.com

---

**© 2025 Kadari Kaushik. All rights reserved.**
