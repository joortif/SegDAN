from imagedatasetanalyzer import HuggingFaceEmbedding, PyTorchEmbedding, TensorflowEmbedding, OpenCVLBPEmbedding, MedImageInsightEmbedding

class EmbeddingFactory:

    def get_embedding_model(self, embedding_info):
        
        model_name = embedding_info.get("name")
        framework = embedding_info.get("framework")

        res_height = embedding_info.get("resize_height")
        res_width = embedding_info.get("resize_width")

        lbp_radius = embedding_info.get("lbp_radius")
        lbp_num_points = embedding_info.get("lbp_num_points")
        lbp_method = embedding_info.get("lbp_method")

        batch_size = embedding_info.get("embedding_batch_size")

        if model_name and model_name.lower() == "other":
            model_name = embedding_info.get("name_other")

        framework_map = {
            'huggingface': lambda: HuggingFaceEmbedding(model_name, batch_size),
            'pytorch': lambda: PyTorchEmbedding(model_name, batch_size),
            'tensorflow': lambda: TensorflowEmbedding(model_name, batch_size, res_height, res_width),
            'opencv': lambda: OpenCVLBPEmbedding(lbp_radius, lbp_num_points, res_height, res_width, batch_size, lbp_method),
            'medimageinsight': lambda: MedImageInsightEmbedding(batch_size)
        }

        if framework not in framework_map:
            raise ValueError(f"Framework '{framework}' not supported. ")
        
        return framework_map[framework]()



