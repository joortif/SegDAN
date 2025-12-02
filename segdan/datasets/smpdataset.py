from segdan.datasets.semantic_segmentation_dataset import SemanticSegmentationDataset

class SMPDataset(SemanticSegmentationDataset):
    
    def __init__(self, *args):
        super().__init__(*args)

    def __getitem__(self, i):
        image, mask = super()._get_raw(i)  
        image = image.transpose(2, 0, 1)
        return image, mask