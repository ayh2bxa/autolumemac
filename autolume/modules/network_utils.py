def extract_conv_names(model):
    """Extract convolutional layer names from a model (non-mapping layers)."""
    model_names = [name for name, weight in model.named_parameters() if "mapping" not in name]
    return model_names


def extract_mapping_names(model):
    """Extract mapping layer names from a model."""
    model_names = [name for name, weight in model.named_parameters() if "mapping" in name]
    return model_names
