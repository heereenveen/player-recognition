# Player recognition on football field with Computer Vision

## Installation

1. Clone the repository:
```
git clone https://github.com/heereenveen/player-recognition.git
```
2. Install the required dependencies:
```
pip install -r requirements.txt
```
3. Download the necessary model files:
```
python src/setup.py
```

## Usage

To run the live viewer, use the following command:

```
python src/main.py --source_video_path /path/to/video.mp4 --device cpu
```

Replace `/path/to/video.mp4` with the path to your input video file. The `--device` argument specifies the device to use for inference, either `cpu` or `cuda`.

## API

The main components of the project are:

- `RadarProcessor`: Processes the input video frames and generates annotated frames.
- `TeamClassifier`: Classifies players into teams based on their visual features.
- `ModelFactory`: Provides access to the pre-trained YOLO models for player, ball, and pitch detection.

## License

This project is licensed under the [MIT License](LICENSE).
