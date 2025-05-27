import abc
import config
import gdown
from pathlib import Path


class DownloadFiles(abc.ABC):
    @abc.abstractmethod
    def download(self, file_id: str, output_path: Path) -> bool:
        pass


class GDownDownload(DownloadFiles):
    def download(self, file_id: str, output_path: Path) -> bool:
        try:
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, str(output_path), quiet=True)
            return True
        except (ConnectionError, TimeoutError) as e:
            print(e)
            return False


class ProjectSetup:
    def __init__(self, strategy: DownloadFiles, data_dir: Path, models: dict):
        self.strategy = strategy
        self.data_dir = data_dir
        self.files = models

    def setup(self) -> bool:
        self.data_dir.mkdir(exist_ok=True)

        for filename, file_id in self.files.items():
            output_path = self.data_dir / filename
            if not output_path.exists():
                print(f"Downloading {filename}...")
                if not self.strategy.download(file_id, output_path):
                    print(f"Not Downloaded {filename}")
                    return False
                print(f"{filename}")

        print("All files is downloaded")
        return True


if __name__ == "__main__":
    setup = ProjectSetup(
        strategy=GDownDownload(), data_dir=Path("src/data"), models=config.VIDEO_FILES
    )
    setup.setup()
