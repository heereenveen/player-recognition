from ultralytics import YOLO


class YoloTrainer:
    def __init__(
        self,
        dataset,
        batch_size,
        epochs,
        img_size,
        model_type="yolov8x.pt",
        enable_plots=True,
    ):
        self.dataset = dataset
        self.model_type = model_type
        self.batch_size = batch_size
        self.epochs = epochs
        self.img_size = img_size
        self.enable_plots = enable_plots
        self.model = None

    def load_model(self):
        self.model = YOLO(self.model_type)
        return self.model

    def train(self):
        if self.model is None:
            self.load_model()

        data_yaml = f"{self.dataset}/data.yaml"

        results = self.model.train(
            data=data_yaml,
            batch=self.batch_size,
            epochs=self.epochs,
            imgsz=self.img_size,
            plots=self.enable_plots,
            task="detect",
        )

        return results
