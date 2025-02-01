
import torch
import logging
import cv2
import matplotlib.pyplot as plt
import numpy as np

class Utils():

    @staticmethod
    def get_device(logger):

        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Device available in CUDA. Using GPU.")
        else:
            device = torch.device("cpu")
            logger.info("Could not find device in CUDA. Using CPU.")

        return device
    
    @staticmethod
    def overlay_mask_on_image(image_path, mask, alpha=0.5):
        # Cargar imagen original
        img = cv2.imread(image_path)
        h, w, _ = img.shape  # Obtener dimensiones

        # Asegurar que la máscara tenga el mismo tamaño que la imagen
        if mask.shape[:2] != (h, w):
            print(f"Mask shape is {h}x{w} while imgs shape is {img.shape[0]}x{img.shape[1]}")
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Obtener clases únicas en la máscara
        unique_classes = np.unique(mask)
        unique_classes = unique_classes[unique_classes > 0]  # Excluir fondo (0)

        # Asignar colores aleatorios a cada clase
        np.random.seed(42)  # Para mantener consistencia en los colores
        colors = {class_id: np.random.randint(0, 255, size=(3,), dtype=np.uint8) for class_id in unique_classes}

        # Crear una máscara en color
        mask_colored = np.zeros((*mask.shape, 3), dtype=np.uint8)

        for class_id, color in colors.items():
            mask_colored[mask == class_id] = color

        # Mezclar imagen original con la máscara coloreada
        overlay = cv2.addWeighted(img, 1 - alpha, mask_colored, alpha, 0)

        # Mostrar la imagen resultante
        plt.figure(figsize=(10, 5))
        plt.imshow(overlay)
        plt.axis('off')
        plt.title("Máscara sobrepuesta con colores por clase")
        plt.show()

        return overlay
