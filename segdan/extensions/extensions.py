from enum import Enum

from segdan.exceptions.exceptions import ExtensionNotFoundException

class LabelExtensions(Enum):
    JSON = '.json'
    TXT = '.txt'
    PNG = '.png'
    JPG = '.jpg'
    JPEG = '.jpeg'
    TIFF = '.tiff'
    BMP = '.bmp'

    def extensionToEnum(ext):
        for extension in LabelExtensions:
            if ext.lower() == extension.value.lower():  
                return extension
        raise ExtensionNotFoundException(ext)
    
    def enumToExtension(enum):
        return enum.value