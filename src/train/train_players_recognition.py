import dotenv
import os
from roboflow import Roboflow
from yolo_trainer import YoloTrainer

class TrainPlayersDataset:
    def __init__(self, api_key: str, project_workspace: str, project_name: str, dataset_version: str = "yolov8"):
        self.api_key = api_key
        self.project_workspace = project_workspace
        self.project_name = project_name
        self.dataset_version = dataset_version
    
    def get_train_dataset(self):
        roboflow = Roboflow(api_key=self.api_key)
        project = roboflow.workspace(self.project_workspace).project(self.project_name)
        version = project.version(2)
        dataset = version.download(self.dataset_version)
        return dataset
    
def main(dataset_downloader: TrainPlayersDataset, yolo: YoloTrainer) -> None:
    dataset = dataset_downloader.get_train_dataset()
    yolo_trainer = yolo(dataset=dataset.location)
    yolo_trainer.train()

if __name__ == "__main__":
    dotenv.load_dotenv()

    main(dataset_downloader=TrainPlayersDataset(api_key=os.getenv('ROBOFLOW_API_KEY'),
                             project_workspace="roboflow-jvuqo",
                             project_name="football-players-detection-3zvbc"),
        yolo=YoloTrainer)