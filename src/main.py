import argparse
import cv2
from marking.radar_processor import RadarProcessor


def main(source_video_path: str, device: str) -> None:
    processor = RadarProcessor(device=device)
    for annotated_frame in processor.process_frames(source_video_path):
        cv2.imshow("Annotated Frame", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Football Radar Live Viewer")
    parser.add_argument(
        "--source_video_path", type=str, required=True, help="Шлях до вхідного відео"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Пристрій: cpu або cuda"
    )
    args = parser.parse_args()
    main(args.source_video_path, args.device)
