import os 

from src.extensions.extensions import LabelExtensions

def transform_labels(labels_dir, imgs_dir, input, data, output_path, background, transformerFactory): 
    transformations_path = os.path.join(output_path, "transformations", "multilabel")
        
    os.makedirs(transformations_path, exist_ok=True)

    print(f"Transforming labels from {input} to multilabel. Results are saved in {transformations_path}")

    transformer = transformerFactory.get_converter(input, 'multilabel')

    if input in [LabelExtensions.TXT.value, LabelExtensions.JSON.value]:  
        transformer.transform(input_data=labels_dir, img_dir=imgs_dir, fill_background=background, output_dir=transformations_path)
    
    elif input == "color":
        transformer.transform(input_data=labels_dir, output_dir=transformations_path, color_dict=data["color_dict"])

    elif input == "binary":
        transformer.transform(input_data=labels_dir, img_dir = imgs_dir, output_dir=transformations_path, threshold=data["threshold"])

    return transformations_path