import abc
import typing
import numpy
import supervision


class Tracking(abc.ABC):
    @abc.abstractmethod
    def track(
        self, detections: supervision.Detections, buffer: typing.List[numpy.ndarray]
    ) -> supervision.Detections:
        pass


class CentroidTracking(Tracking):
    def track(
        self, detections: supervision.Detections, buffer: typing.List[numpy.ndarray]
    ) -> supervision.Detections:
        if len(detections) == 0:
            return detections

        xy = detections.get_anchors_coordinates(supervision.Position.CENTER)
        buffer.append(xy)

        if buffer:
            centroid = numpy.mean(numpy.concatenate(buffer), axis=0)
            distances = numpy.linalg.norm(xy - centroid, axis=1)
            index = numpy.argmin(distances)
            return detections[[index]]

        return detections


class BallTracker:
    def __init__(self, strategy: Tracking, buffer_size: int = 10):
        self._strategy = strategy
        self._buffer_size = buffer_size
        self.buffer: typing.List[numpy.ndarray] = []

    def update(self, detections: supervision.Detections) -> supervision.Detections:
        result = self._strategy.track(detections, self.buffer)
        if len(self.buffer) > self._buffer_size:
            self.buffer.pop(0)
        return result
