from models.semanticsegmentationmodel import SemanticSegmentationModel
from models.hfstransformermodel import HFTransformerModel
from models.smpmodel import SMPModel

class SemanticSegmentationModelFactory:

    def __init__(self):
        self.framework_map = {
            "smp": (SMPModel, {"in_channels", "t_max"}),
            "huggingface": (HFTransformerModel, {"batch_size"})
        }

    def get_model(self, framework: str, args: dict) -> SemanticSegmentationModel:
        try:
            model_class, required_params = self.converter_map[framework]
        except KeyError:
            raise ValueError(f"Unsupported semantic segmentation framework: {framework}")
        
        missing = required_params - args.keys()
        if missing:
            raise ValueError(f"Missing required context params: {missing} for model {model_class.__name__}")
        
        init_args = {key: args[key] for key in required_params if key in args}
        return model_class(**init_args)