import os
from ultralytics import YOLO


class ModelFactory:

    @staticmethod
    def create_player_model(device: str) -> YOLO:
        model_path = os.path.join(
            os.path.dirname(__file__), "data/football-player-detection.pt"
        )
        return YOLO(model_path).to(device=device)

    @staticmethod
    def create_pitch_model(device: str) -> YOLO:
        model_path = os.path.join(
            os.path.dirname(__file__), "data/football-pitch-detection.pt"
        )
        return YOLO(model_path).to(device=device)
