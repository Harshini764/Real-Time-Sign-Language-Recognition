# Indian Sign Language (ISL) to Speech Recognition

This project uses a trained deep learning model to recognize Indian Sign Language (ISL) gestures (digits 1–9 and letters A–Z) from your webcam in real time and converts them to speech.

## Features

- Real-time webcam capture and gesture recognition
- Supports ISL digits (1–9) and letters (A–Z)
- Displays prediction and confidence on screen
- Speaks out recognized gestures using text-to-speech
- Prediction smoothing for robust output

## Requirements

- Python 3.7+
- [TensorFlow](https://www.tensorflow.org/)
- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)
- [pyttsx3](https://pypi.org/project/pyttsx3/)

Install dependencies with:
```bash
pip install tensorflow opencv-python numpy pyttsx3
```

## Usage

1. **Place your trained model** at `model/isl_model.h5`.
2. **Run the script:**
   ```bash
   python main.py
   ```
3. **Show your hand gesture** inside the blue box on the webcam window.
4. The predicted letter or digit and its confidence will appear on the screen.
5. The system will speak out the recognized gesture when it is stable.
6. **Press `ESC`** to exit.

## How It Works

- The script captures frames from your webcam.
- It defines a region of interest (ROI) where you should show your gesture.
- The ROI is preprocessed and passed to the trained model for prediction.
- Predictions are smoothed using a buffer to avoid false positives.
- When a gesture is confidently and consistently recognized, it is spoken aloud.

## File Structure

```
main.py
model/
  isl_model.h5
```

## Notes

- Make sure your model matches the input size (64x64) and output classes (1–9, A–Z).
- Adjust the ROI or preprocessing as needed for your camera setup.
- You can modify the `labels` list in the script if your model uses a different label order.

## License

This project is for educational
