# HumanVision-Detector

## 🔍 Description
HumanVision-Detector is a real-time human detection system built with OpenCV and HOG (Histogram of Oriented Gradients) descriptor. The system can detect and count people in both images and video streams, making it suitable for various applications including surveillance, crowd monitoring, and foot traffic analysis.

## 🚀 Features
- Real-time human detection in images and videos
- Person counting functionality 
- Bounding box visualization around detected people
- Support for both image and video input
- Configurable output options
- Status display showing detection progress
- Resizable frame processing

## 🛠️ Requirements
- Python 3.6+
- OpenCV
- imutils
- numpy
- argparse

## 📦 Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/HumanVision-Detector.git

# Navigate to the project directory
cd HumanVision-Detector

# Install required packages
pip install opencv-python imutils numpy argparse
```

## 💻 Usage
The detector can be used with either images or videos:

### For Images:
```python
args = {
    'image': 'path/to/your/image.jpg',
    'video': None,
    'output': 'detected_output.jpg'
}
humanDetector(args)
```

### For Videos:
```python
args = {
    'image': None,
    'video': 'path/to/your/video.mp4',
    'output': 'output_video.avi'
}
humanDetector(args)
```

## 🎯 How It Works
1. The system uses HOG (Histogram of Oriented Gradients) descriptor for human detection
2. OpenCV's pre-trained SVM classifier is used for person detection
3. Detection results are visualized with bounding boxes and person count
4. Real-time status and total person count are displayed on the output

## 📊 Performance
- Supports frame resizing for optimal performance
- Configurable detection parameters:
  - winStride: (4, 4)
  - padding: (8, 8)
  - scale: 1.03

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details

## ⚠️ Note
For optimal performance, ensure proper lighting conditions and clear visibility in input media. The detection accuracy may vary based on environmental conditions and image quality.
