from enum import Enum

from src.exceptions.exceptions import ExtensionNotFoundException

class LabelExtensions(Enum):
    JSON = '.json'
    TXT = '.txt'
    PNG = '.png'
    JPG = '.jpg'

    def extensionToEnum(ext):
        for extension in LabelExtensions:
            if ext.lower() == extension.value.lower():  
                return extension
        raise ExtensionNotFoundException(ext)
    
    def enumToExtension(enum):
        return enum.value