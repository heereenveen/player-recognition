import abc
import typing
import cv2
import numpy


class ViewTransformer(abc.ABC):
    @abc.abstractmethod
    def transform_points(self, points: numpy.ndarray) -> numpy.ndarray:
        pass

    @abc.abstractmethod
    def transform_image(
        self, image: numpy.ndarray, resolution_wh: typing.Tuple[int, int]
    ) -> numpy.ndarray:
        pass


class HomographyMatrixViewTransformer(ViewTransformer):

    def __init__(self, source: numpy.ndarray, target: numpy.ndarray):
        source = source.astype(numpy.float32)
        target = target.astype(numpy.float32)
        self.m, _ = cv2.findHomography(source, target)
        if self.m is None:
            raise ValueError("We cannot calculate Homography matrix")

    def transform_points(self, points: numpy.ndarray) -> numpy.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(numpy.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2).astype(numpy.float32)

    def transform_image(
        self, image: numpy.ndarray, resolution_wh: typing.Tuple[int, int]
    ) -> numpy.ndarray:
        return cv2.warpPerspective(image, self.m, resolution_wh)


class AffineViewTransformer(ViewTransformer):
    def __init__(self, source: numpy.ndarray, target: numpy.ndarray):
        if source.shape[0] < 3 or target.shape[0] < 3:
            raise ValueError("We need 3 source points")

        source = source[:3].astype(numpy.float32)
        target = target[:3].astype(numpy.float32)
        self.m = cv2.getAffineTransform(source, target)

    def transform_points(self, points: numpy.ndarray) -> numpy.ndarray:
        if points.size == 0:
            return points

        ones = numpy.ones((points.shape[0], 1))
        homogeneous_points = numpy.hstack([points, ones])
        transformed = homogeneous_points @ self.m.T
        return transformed.astype(numpy.float32)

    def transform_image(
        self, image: numpy.ndarray, resolution_wh: typing.Tuple[int, int]
    ) -> numpy.ndarray:
        return cv2.warpAffine(image, self.m, resolution_wh)


class TransformationFactory:
    @staticmethod
    def create_homography_transformer(
        source: numpy.ndarray, target: numpy.ndarray
    ) -> ViewTransformer:
        return HomographyMatrixViewTransformer(source, target)

    @staticmethod
    def create_affine_transformer(
        source: numpy.ndarray, target: numpy.ndarray
    ) -> ViewTransformer:
        return AffineViewTransformer(source, target)
