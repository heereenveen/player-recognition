import typing
import cv2
import numpy
import supervision


class BallObserver:
    def __init__(
        self,
        radius: int = 10,
        buffer_size: int = 5,
        thickness: int = 2,
        color_palette_name: str = "jet",
    ):
        self.color_palette = supervision.ColorPalette.from_matplotlib(
            color_palette_name, buffer_size
        )
        self.buffer: typing.List[numpy.ndarray] = []
        self._buffer_size = buffer_size
        self.radius = radius
        self.thickness = thickness

    def interpolate_radius(self, i: int, max_i: int) -> int:
        if max_i == 1:
            return self.radius
        return int(1 + i * (self.radius - 1) / (max_i - 1))

    def annotate(
        self, frame: numpy.ndarray, detections: supervision.Detections
    ) -> numpy.ndarray:
        xy = detections.get_anchors_coordinates(
            supervision.Position.BOTTOM_CENTER
        ).astype(int)
        self.buffer.append(xy)

        if len(self.buffer) > self._buffer_size:
            self.buffer.pop(0)

        for i, xy in enumerate(self.buffer):
            color = self.color_palette.by_idx(i)
            radius = self.interpolate_radius(i, len(self.buffer))
            for center in xy:
                frame = cv2.circle(
                    frame, tuple(center), radius, color.as_bgr(), self.thickness
                )

        return frame
