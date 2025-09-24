# Morse Code Eye Blink Detection

A computer vision application that translates eye blinks into Morse code messages using OpenCV and webcam input.

## 🎯 Overview

This project uses computer vision to detect eye blinks through a webcam and interprets them as Morse code signals. Short blinks represent dots (.), longer blinks represent dashes (-), and the system automatically decodes the Morse code into readable text messages.

## 📋 Features

- **Real-time eye blink detection** using Haar cascades
- **Morse code translation** with dot/dash timing recognition
- **Automatic letter and word separation** based on timing gaps
- **Robust lighting adaptation** with histogram equalization and contrast adjustment
- **Frame smoothing** to reduce false positives
- **Live message display** on the video feed

## 📁 Project Files

- `morsecode.py` - Main application using Haar cascades (recommended)
- `README.md` - This documentation

## 🛠️ Requirements

### System Requirements
- Python 3.7 - 3.12 (for MediaPipe version)
- Webcam/Camera
- Windows/macOS/Linux

### Python Dependencies
```
opencv-python>=4.5.0
```

For the MediaPipe version (`new.py`), additionally install:
```
mediapipe>=0.8.0
```

## 📦 Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install opencv-python
   ```

   For MediaPipe version (optional):
   ```bash
   pip install mediapipe
   ```
   *Note: MediaPipe currently supports Python 3.12 and below*

3. **Run the application**:
   ```bash
   python morsecode.py
   ```

## 🎮 How to Use

1. **Launch the application** by running `python morsecode.py`
2. **Position yourself** in front of the camera with good lighting
3. **Blink to input Morse code**:
   - **Short blink** (< 2 seconds) = Dot (.)
   - **Long blink** (2-3 seconds) = Dash (-)
4. **Pause between letters** (2+ seconds) for automatic letter separation
5. **Pause longer** (3+ seconds) for word separation
6. **Press ESC** to exit the application

## 📖 Morse Code Reference

| Letter | Code | Letter | Code | Number | Code |
|--------|------|--------|------|---------|------|
| A | .- | N | -. | 0 | ----- |
| B | -... | O | --- | 1 | .---- |
| C | -.-. | P | .--. | 2 | ..--- |
| D | -.. | Q | --.- | 3 | ...-- |
| E | . | R | .-. | 4 | ....- |
| F | ..-. | S | ... | 5 | ..... |
| G | --. | T | - | 6 | -.... |
| H | .... | U | ..- | 7 | --... |
| I | .. | V | ...- | 8 | ---.. |
| J | .--- | W | .-- | 9 | ----. |
| K | -.- | X | -..- |   |       |
| L | .-.. | Y | -.-- |   |       |
| M | -- | Z | --.. |   |       |

## ⚙️ Configuration

You can adjust the timing parameters in the code:

```python
DOT_DURATION   = 2.0    # Maximum duration for a dot (seconds)
DASH_DURATION  = 3.0    # Maximum duration for a dash (seconds)
LETTER_GAP     = 2.0    # Gap between letters (seconds)
WORD_GAP       = 3.0    # Gap between words (seconds)
```

## 🔧 Troubleshooting

### Common Issues

1. **"No module named 'cv2'"**
   ```bash
   pip install opencv-python
   ```

2. **"No module named 'mediapipe'"** (for `new.py`)
   - Install MediaPipe: `pip install mediapipe`
   - Or use Python 3.12 or below
   - Or use `morsecode.py` instead (doesn't require MediaPipe)

3. **Camera not detected**
   - Check if camera is connected and not used by other applications
   - Try changing camera index from `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`

4. **Poor blink detection**
   - Ensure good lighting conditions
   - Position face clearly in frame
   - Adjust contrast/brightness parameters in code

## 🚀 Technical Details

### Haar Cascade Version (`morsecode.py`)
- Uses pre-trained Haar cascade classifiers for face and eye detection
- Implements histogram equalization for lighting robustness
- Frame smoothing with majority voting to reduce false positives
- Compatible with all Python versions supporting OpenCV

### MediaPipe Version (`new.py`)
- Uses MediaPipe Face Mesh for more precise eye landmark detection
- Calculates Eye Aspect Ratio (EAR) for blink detection
- More accurate but requires MediaPipe dependency
- Limited to Python ≤3.12 due to MediaPipe compatibility

## 📈 Future Enhancements

- [ ] GUI interface for easier configuration
- [ ] Sound feedback for blink confirmation
- [ ] Save/load message history
- [ ] Customizable timing presets
- [ ] Multi-language support
- [ ] Machine learning-based blink detection

## 🤝 Contributing

Feel free to contribute by:
- Reporting bugs
- Suggesting new features
- Submitting pull requests
- Improving documentation

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- OpenCV community for computer vision tools
- MediaPipe team for facial landmark detection
- Morse code historical significance in communication

---

**Happy blinking! 👁️**