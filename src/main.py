import argparse
from src.marking.radar_processor import RadarProcessor


def main(source_video_path: str, target_video_path: str, device: str) -> None:
    processor = RadarProcessor(device=device)
    processor.process_video(source_video_path, target_video_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Football Classifier Model")
    parser.add_argument("--source_video_path", type=str, required=True)
    parser.add_argument("--target_video_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    main(args.source_video_path, args.target_video_path, args.device)
