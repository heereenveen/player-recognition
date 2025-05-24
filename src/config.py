PROJECT_WORKSPACE = "roboflow-jvuqo"

PLAYERS_DETECTION_PROJECT_NAME = "football-players-detection-3zvbc"
PLAYERS_DETECTION_PROJECT_VERSION = 10
PLAYERS_YOLO_TRAINING_ARGS = {
    "batch_size": 6,
    "epochs": 50,
    "img_size": 1280,
}

BALL_DETECTION_PROJECT_NAME = "football-ball-detection-rejhg"
BALL_DETECTION_PROJECT_VERSION = 2
BALL_YOLO_TRAINING_ARGS = {
    "batch_size": 12,
    "epochs": 50,
    "img_size": 1280,
}

PITCH_DETECTION_PROJECT_NAME = "football-field-detection-f07vi"
PITCH_DETECTION_PROJECT_VERSION = 12
PITCH_YOLO_TRAINING_ARGS = {
    "batch_size": 16,
    "epochs": 100,
    "img_size": 640,
    "mosaic": 0.0,
}

MODELS_FILES = {
    "football-ball-detection.pt": "1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V",
    "football-player-detection.pt": "17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q",
    "football-pitch-detection.pt": "1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf",
}
