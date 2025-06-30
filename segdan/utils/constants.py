from enum import Enum

VALID_EXTENSIONS: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

class StratificationStrategy(Enum):
    PIXELS = "pixels"
    OBJECTS = "objects"
    RATIO = "pixel_to_object_ratio"

class Framework(Enum):
    HUGGINGFACE = "huggingface"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    OPENCV = "opencv"

class ClusteringModelName(Enum):
    KMEANS = "kmeans"
    AGGLOMERATIVE = "agglomerative"
    DBSCAN = "dbscan"
    OPTICS = "optics"

class LabelFormat(Enum):
    MASK = "mask"
    JSON = "json"
    TXT = "txt"
    COLOR = "color"
    BINARY = "binary"

class SegmentationType(Enum):
    INSTANCE = "instance"
    SEMANTIC = "semantic"
