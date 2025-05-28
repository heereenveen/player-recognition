import abc
import typing
import numpy as np
import supervision as sv
from tqdm import tqdm

from models import ModelFactory
from marking.classifier import TeamClassifier

PLAYER_CLASS_ID = 2
GOALKEEPER_CLASS_ID = 1
REFEREE_CLASS_ID = 3
STRIDE = 60
COLORS = ["#FF1493", "#00BFFF", "#FF6347", "#FFD700"]


class RadarStrategy(abc.ABC):
    @abc.abstractmethod
    def process_frame(
        self,
        frame: np.ndarray,
        models: dict,
        team_classifier: TeamClassifier,
        tracker: sv.ByteTrack,
    ) -> np.ndarray:
        pass


class FootballRadarStrategy(RadarStrategy):
    def __init__(self):
        self.ellipse_annotator = sv.EllipseAnnotator(
            color=sv.ColorPalette.from_hex(COLORS), thickness=2
        )
        self.label_annotator = sv.LabelAnnotator(
            color=sv.ColorPalette.from_hex(COLORS),
            text_color=sv.Color.from_hex("#FFFFFF"),
            text_padding=5,
            text_thickness=1,
            text_position=sv.Position.BOTTOM_CENTER,
        )

    def process_frame(
        self,
        frame: np.ndarray,
        models: dict,
        team_classifier: TeamClassifier,
        tracker: sv.ByteTrack,
    ) -> np.ndarray:

        player_result = models["player"](frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(player_result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        crops = [sv.crop_image(frame, xyxy) for xyxy in players.xyxy]
        players_team_id = team_classifier.predict(crops) if crops else np.array([])

        goalkeepers_team_id = self._resolve_goalkeepers_team(
            players, players_team_id, goalkeepers
        )

        all_detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.concatenate(
            [
                players_team_id,
                goalkeepers_team_id,
                np.full(len(referees), REFEREE_CLASS_ID),
            ]
        )

        annotated_frame = frame.copy()
        annotated_frame = self.ellipse_annotator.annotate(
            annotated_frame, all_detections, custom_color_lookup=color_lookup
        )

        labels = [str(tid) for tid in all_detections.tracker_id]
        annotated_frame = self.label_annotator.annotate(
            annotated_frame, all_detections, labels, custom_color_lookup=color_lookup
        )

        return annotated_frame

    def _resolve_goalkeepers_team(
        self,
        players: sv.Detections,
        players_team_id: np.ndarray,
        goalkeepers: sv.Detections,
    ) -> np.ndarray:
        if len(goalkeepers) == 0 or len(players) == 0:
            return np.array([])

        gk_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        pl_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

        if len(players_team_id) == 0:
            return np.zeros(len(goalkeepers), dtype=int)

        team0_centroid = pl_xy[players_team_id == 0].mean(axis=0)
        team1_centroid = pl_xy[players_team_id == 1].mean(axis=0)

        gk_team = []
        for coord in gk_xy:
            d0 = np.linalg.norm(coord - team0_centroid)
            d1 = np.linalg.norm(coord - team1_centroid)
            gk_team.append(0 if d0 < d1 else 1)
        return np.array(gk_team, dtype=int)


class RadarProcessor:
    def __init__(self, device: str = "cpu", stride: int = STRIDE):
        self.device = device
        self.stride = stride
        self.models = {
            "player": ModelFactory.create_player_model(device),
            "pitch": ModelFactory.create_pitch_model(device),
        }
        self.strategy = FootballRadarStrategy()
        self.team_classifier = TeamClassifier(device=device)
        self.tracker = sv.ByteTrack(minimum_consecutive_frames=3)

    def process_frames(self, source_path: str) -> typing.Iterator[np.ndarray]:
        # Збираємо приклади для класифікатора команд
        self._collect_crops(source_path)
        # Генеруємо кадри
        frames = sv.get_video_frames_generator(source_path)
        for frame in frames:
            out = self.strategy.process_frame(
                frame, self.models, self.team_classifier, self.tracker
            )
            yield out  # повертаємо готовий кадр

    def _collect_crops(self, source_path: str) -> None:
        frames = sv.get_video_frames_generator(source_path, stride=self.stride)
        crops: typing.List[np.ndarray] = []
        for frame in tqdm(frames, desc="Collect crops"):
            res = self.models["player"](frame, imgsz=1280, verbose=False)[0]
            det = sv.Detections.from_ultralytics(res)
            players = det[det.class_id == PLAYER_CLASS_ID]
            crops += [sv.crop_image(frame, xyxy) for xyxy in players.xyxy]
        if crops:
            self.team_classifier.fit(crops)
