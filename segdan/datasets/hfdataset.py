from datasets.semantic_segmentation_dataset import SemanticSegmentationDataset

class HFDataset(SemanticSegmentationDataset):
    
    def __init__(self, *args):
        super().__init__(*args)

    def __getitem__(self, i):
        image, mask = super()._get_raw(i)  
        return {"image": image, "segmentation_map": mask}