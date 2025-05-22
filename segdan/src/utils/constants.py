from enum import Enum

VALID_EXTENSIONS: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

class StratificationStrategy(Enum):
    PIXELS = "pixels"
    OBJECTS = "objects"

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

class SegmentationType(Enum):
    INSTANCE = "instance"
    SEMANTIC = "semantic"
    