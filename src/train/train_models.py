import dotenv
import os
import argparse
import config

from roboflow import Roboflow
from yolo_trainer import YoloTrainer

class TrainDataset:
    def __init__(self, api_key: str, project_workspace: str, project_name: str, project_version: str, dataset_version: str = "yolov8"):
        self.api_key = api_key
        self.project_workspace = project_workspace
        self.project_name = project_name
        self.project_version = project_version
        self.dataset_version = dataset_version
    
    def get_train_dataset(self):
        roboflow = Roboflow(api_key=self.api_key)
        project = roboflow.workspace(self.project_workspace).project(self.project_name)
        version = project.version(self.project_version)
        dataset = version.download(self.dataset_version)
        return dataset

def get_config(detection_type: str) -> dict:
    configs = {
        'players': {
            'project_name': config.PLAYERS_DETECTION_PROJECT_NAME,
            'project_version': config.PLAYERS_DETECTION_PROJECT_VERSION,
            'training_args': config.PLAYERS_YOLO_TRAINING_ARGS
        },
        'ball': {
            'project_name': config.BALL_DETECTION_PROJECT_NAME,
            'project_version': config.BALL_DETECTION_PROJECT_VERSION,
            'training_args': config.BALL_YOLO_TRAINING_ARGS
        },
        'pitch': {
            'project_name': config.PITCH_DETECTION_PROJECT_NAME,
            'project_version': config.PITCH_DETECTION_PROJECT_VERSION,
            'training_args': config.PITCH_YOLO_TRAINING_ARGS
        }
    }
    return configs.get(detection_type)

def main(dataset_downloader: TrainDataset, yolo: YoloTrainer) -> None:
    dataset = dataset_downloader.get_train_dataset()
    yolo_trainer = yolo(dataset=dataset.location, **project_config['training_args'])
    yolo_trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train_detection', 
                        choices=['players', 'ball', 'pitch'],
                        type=str,
                        required=True,
                        help='Chose option to train')
    args = parser.parse_args()

    project_config = get_config(args.train_detection)
    
    dotenv.load_dotenv()
    dataset=TrainDataset(api_key=os.getenv('ROBOFLOW_API_KEY'),
                             project_workspace=config.PROJECT_WORKSPACE,
                             project_name=project_config['project_name'],
                             project_version=project_config['project_version'])

    main(dataset_downloader=dataset, yolo=YoloTrainer)