from transformers import AutoImageProcessor, AutoModelForDepthEstimation, AutoConfig
import torch
import numpy as np
import random
from PUDE.PUDE_implementation import load_model as load_PUDE_model

seed = 42
np.random.seed(seed)
random.seed(seed)

# models and their paths, pude has a local path, the rest are from huggingface
models = {"pude": "PUDE/weightsave/final2/finall2.pth", "depth_anything": "nielsr/depth-anything-small",  "dpt3_1": "Intel/dpt-swinv2-large-384"}
default_image_dim = (576,384)

def get_model_output(model, image_processor, raw_image):
    inputs = preprocess_image(image_processor=image_processor, image=raw_image)
    predicted_depth = predict_depth(model=model, inputs=inputs)
    output = post_process_depth(depth=predicted_depth, image=raw_image)
    return output.flatten()

def preprocess_image(image_processor, image):
    # prepare image for the model
    inputs = image_processor(images=image, return_tensors="pt")
    return inputs

def predict_depth(model, inputs):
    #forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        # output is a tensor
        if isinstance(outputs, torch.Tensor):
            predicted_depth = outputs
        else:
         predicted_depth = outputs.predicted_depth
    return predicted_depth

def post_process_depth(depth, image):
    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        depth.unsqueeze(1),
        size=image.shape[:-1],
        mode="bicubic",
        align_corners=False,
    )
    # visualize the prediction
    output = prediction.squeeze().cpu().numpy()
    return output


def get_PUDE_image_processor(transform):
    def image_processor_PUDE(images, return_tensors="pt"):
        # if images not np array
        if type(images) != np.ndarray:
            images = np.array(images)
        net_img = images
        net_img = transform({"image": net_img})["image"]
        net_img = torch.from_numpy(net_img)
        # reshape the tensor to include batch size of 1
        net_img = net_img.reshape((1, *net_img.shape))
        return {'x':net_img}
       
    return image_processor_PUDE

def get_model_image_processor_pair(model_name, model_path, device):
    if model_name == "pude":
        model, transform = load_PUDE_model(model_path, device)
        image_processor = get_PUDE_image_processor(transform)
    else:
        image_processor = AutoImageProcessor.from_pretrained(model_path)
        model = AutoModelForDepthEstimation.from_pretrained(model_path)
    
    return model, image_processor

def get_two_separate_model_pairs(model_path):
    # Load the model configuration
    config = AutoConfig.from_pretrained(model_path)

    # Create the first instance of the depth model
    depth_model_1 = AutoModelForDepthEstimation.from_pretrained(model_path, config=config)

    # Create the second instance of the depth model with a different memory pointer
    depth_model_2 = AutoModelForDepthEstimation.from_pretrained(model_path, config=config)
    image_processor = AutoImageProcessor.from_pretrained(model_path)
    return depth_model_1, depth_model_2, image_processor, image_processor