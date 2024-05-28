from transformers import AutoImageProcessor, AutoModelForDepthEstimation, AutoConfig
import torch
import numpy as np
import random
from PUDE.PUDE_implementation import load_model as load_PUDE_model
import cv2
from PIL import Image
import Pude_training_loop.pude_utils as pude_utils
from Pude_training_loop.pude_utils import make_image_grid, make_image_grid_title
import unimatch.dataloader.stereo.transforms as unimatch_transforms
from unimatch.unimatch.unimatch import UniMatch

seed = 42
np.random.seed(seed)
random.seed(seed)

#"unimatch/gmstereo-scale2-regrefine3-resumeflowthings-mixdata-train320x640-ft640x960-e4e291fd.pth" # works 
#"unimatch/gmstereo-scale2-resumeflowthings-sceneflow-48020649.pth" # doesnt work 
#"unimatch/gmstereo-scale1-sceneflow-124a438f.pth" # doesnt work 
# "unimatch\gmstereo-scale2-sceneflow-ab93ba6a.pth" # doesnt work 
# "unimatch\gmstereo-scale2-regrefine3-resumeflowthings-eth3dft-a807cb16.pth" # works 

# models and their paths, pude has a local path, the rest are from huggingface
models = {"pude": "PUDE/weightsave/final2/finall2.pth", 
          "depth_anything": "nielsr/depth-anything-small",  
          "dpt3_1": "Intel/dpt-swinv2-large-384", 
          "unimatch": "unimatch/gmstereo-scale2-regrefine3-resumeflowthings-mixdata-train320x640-ft640x960-e4e291fd.pth",
          "new_pude": "new_pude_model_AdamW_6_0-0005_1e-06.pth"}
default_image_dim = (576,384)

def get_model_output(model, image_processor, raw_image, device="cuda", requires_grad=True):
    inputs = preprocess_image(image_processor=image_processor, image=raw_image, device=device)
    predicted_depth = predict_depth(model=model, inputs=inputs, requires_grad=requires_grad)
    output = post_process_depth(depth=predicted_depth, image=raw_image)
    return output.flatten()

def preprocess_image(image_processor, image, device="cuda"):
    # prepare image for the model
    inputs = image_processor(images=image, return_tensors="pt")
    if isinstance(inputs, dict):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)
    else:
        inputs = inputs.to(device)
    return inputs

def predict_depth(model, inputs, requires_grad=True):
    #forward pass
    outputs = model(**inputs)
    if not requires_grad:
        with torch.no_grad():
            outputs = model(**inputs)
    else:
        outputs = model(**inputs)
    # output is a tensor
    if isinstance(outputs, torch.Tensor):
        predicted_depth = outputs
    elif isinstance(outputs, dict):
        if "flow_preds" in outputs.keys():
            predicted_depth = outputs['flow_preds'][-1]  # [1, H, W]
        elif "predicted_depth" in outputs.keys():
            predicted_depth = outputs["predicted_depth"]
    else:
        predicted_depth = outputs.predicted_depth
    return predicted_depth

def post_process_depth(depth, image):
    if isinstance(image, dict):
        image = image['image1']
    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        depth.unsqueeze(1),
        size=image.shape[:-1],
        mode="bicubic",
        align_corners=False,
    )
    # output is a tensor
    output = prediction.squeeze()
    output = (output - output.min()) / (output.max() - output.min()) * 20.0
    
    return output



def evaluate(depth_anything_model, new_pude_model, 
             depth_anything_image_processor, new_pude_image_processor, 
             non_linear_images, device="cuda"):
    def _prepare_image(model_output):
        # prepare images for visualization
        model_output =  model_output.cpu().detach().numpy().reshape(384, 576)
        formatted = (model_output - np.min(model_output)) / (np.max(model_output) - np.min(model_output))
        formatted = (formatted * 255).astype(np.uint8)
        colored_depth = cv2.applyColorMap(formatted, cv2.COLORMAP_INFERNO)[:, :, ::-1]
        depth = Image.fromarray(colored_depth)
        return depth
        
        
    # get model outputs
    depth_anything_output = get_model_output(model=depth_anything_model, 
                                                        image_processor=depth_anything_image_processor, 
                                                        raw_image=non_linear_images, requires_grad=False)
    pude_output = get_model_output(model=new_pude_model,
                                                    image_processor=new_pude_image_processor,
                                                    raw_image=non_linear_images, requires_grad=False)
    
    diff_image = depth_anything_output - pude_output
    non_linear_image = Image.fromarray(non_linear_images)
    depth_images = [non_linear_image, _prepare_image(depth_anything_output), _prepare_image(pude_output), _prepare_image(diff_image), _prepare_image(torch.abs(diff_image))]
    return make_image_grid_title(depth_images, rows=1, cols=5, titles=["Input", "Depth Anything", "PUDE", "Difference", "Absolute Difference"])
    
def evaluate_unimatch(depth_anything_model, unimatch_model, 
             depth_anything_image_processor, unimatch_image_processor, stereo_pair_gen,
             non_linear_images, device="cuda", scaling_factor=1):
 
        
    # get model outputs
    depth_anything_output = get_model_output(model=depth_anything_model, 
                                            image_processor=depth_anything_image_processor, 
                                            raw_image=non_linear_images, requires_grad=False)
    image_2 = stereo_pair_gen.generate_stereo_pair(non_linear_images, depth_anything_output.cpu().detach().numpy(), scaling_factor=scaling_factor)

    unimatch_input = {"image1": non_linear_images, "image2": image_2}
        
    unimatch_output = get_model_output(model=unimatch_model,
                                    image_processor=unimatch_image_processor,
                                    raw_image=unimatch_input, requires_grad=False)
    
    image_grid = pude_utils.unimatch_display_image_with_depth(image=Image.fromarray(non_linear_images),
                                                                depth1=depth_anything_output.cpu().detach().numpy(),
                                                                depth2=unimatch_output.cpu().detach().numpy(),
                                                                shifted_image=Image.fromarray(image_2))
    
    return image_grid



def get_PUDE_image_processor(transform, device):
    def image_processor_PUDE(images, return_tensors="pt"):
        # if images not np array
        if type(images) != np.ndarray or type(images) != torch.Tensor:
            images = np.array(images)
        net_img = images
        net_img = transform({"image": net_img})["image"]
        net_img = torch.tensor(net_img, device=device)
        # reshape the tensor to include batch size of 1
        net_img = net_img.reshape((1, *net_img.shape))
        return {'x':net_img}
       
    return image_processor_PUDE


def get_unimatch_stereo_image_processor(device):
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    def image_processor_unimatch_stereo(images, return_tensors="pt"):
        image1, image2 = images['image1'], images['image2']
        def ensure_image_type(image):
            if not(type(image) is np.ndarray or type(image) is torch.Tensor):
                image = np.array(image)
            return image
        
        image1, image2 = ensure_image_type(image1), ensure_image_type(image2)
        sample = {'left': image1, 'right': image2}
        val_transform_list = [unimatch_transforms.ToTensor(),
                              unimatch_transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                              ]

        val_transform = unimatch_transforms.Compose(val_transform_list)
        sample = val_transform(sample)

        image1 = sample['left'].unsqueeze(0).to(device)  # [1, 3, H, W]
        image2 = sample['right'].unsqueeze(0).to(device)  # [1, 3, H, W]
        attn_type='self_swin2d_cross_swin1d'
        attn_splits_list=[2, 8]
        corr_radius_list=[-1, 4]
        prop_radius_list=[-1, 1]
        num_reg_refine=3
        task='stereo'

        return {'img0': image1, 'img1': image2, 'attn_type': attn_type, 
                'attn_splits_list': attn_splits_list, 'corr_radius_list': corr_radius_list,
                'prop_radius_list': prop_radius_list, 'num_reg_refine': num_reg_refine, 'task': task}
    return image_processor_unimatch_stereo

def get_unimatch_model(model_path):
    model = UniMatch(feature_channels=128,
                     num_scales=2,
                     upsample_factor=4,
                     ffn_dim_expansion=4,
                     num_transformer_layers=6,
                     reg_refine=True,
                     task='stereo')
    model.eval()
    checkpoint_flow = torch.load(model_path)
    if 'model' in checkpoint_flow:
        model.load_state_dict(checkpoint_flow['model'], strict=True)
    else:
        model.load_state_dict(checkpoint_flow, strict=True)
    return model

def get_model_image_processor_pair(model_name, model_path, device):
    if model_name == "pude":
        model, transform = load_PUDE_model(model_path, device)
        image_processor = get_PUDE_image_processor(transform, device)
    elif "unimatch" in model_name:
        model = get_unimatch_model(model_path)
        image_processor = get_unimatch_stereo_image_processor(device)
    elif "new_pude" in model_name:
        model_dic = torch.load(model_path)
        model = AutoModelForDepthEstimation.from_pretrained(models["depth_anything"])
        model.load_state_dict(model_dic)
        image_processor = AutoImageProcessor.from_pretrained(models["depth_anything"])
    else:
        image_processor = AutoImageProcessor.from_pretrained(model_path)
        model = AutoModelForDepthEstimation.from_pretrained(model_path)
    return model.to(device), image_processor

def get_two_separate_model_pairs(model_path, device):
    # Load the model configuration
    config = AutoConfig.from_pretrained(model_path)

    # Create the first instance of the depth model
    depth_model_1 = AutoModelForDepthEstimation.from_pretrained(model_path, config=config)

    # Create the second instance of the depth model with a different memory pointer
    depth_model_2 = AutoModelForDepthEstimation.from_pretrained(model_path, config=config) 
    image_processor = AutoImageProcessor.from_pretrained(model_path, device=device)
    return depth_model_1.to(device), depth_model_2.to(device), image_processor, image_processor