import torch

class HuggingFaceAdapterDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, feature_extractor):
        """
        SMPDataset wrapper for segmentation models from HuggingFace
        """
        self.base_dataset = base_dataset
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, mask = self.base_dataset[idx]  

        image = image.transpose(1, 2, 0)
                    
        encoded = self.feature_extractor(images=image, segmentation_maps=mask, do_resize=False, return_tensors="pt")

        mask_labels = encoded["mask_labels"][0]
        
        res = {
            "pixel_values": encoded["pixel_values"].squeeze(0),  
            "pixel_mask": encoded["pixel_mask"].squeeze(0),  
            "mask_labels": mask_labels,  
            "class_labels": encoded["class_labels"][0],  
        }
                
        return res