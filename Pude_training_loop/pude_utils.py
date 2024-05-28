import numpy as np
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display
from typing import List
import pickle
import cv2

def return_mask_as_image(mask):
    mask = Image.fromarray(mask.astype(np.uint8)*255).convert("RGB")
    return mask

def display_mask_as_image(mask):
    mask = Image.fromarray(mask.astype(np.uint8)*255).convert("RGB")
    display(mask)

def get_depth_as_image(model_output):
     # prepare images for visualization
    model_output =  model_output.reshape(384, 576)
    formatted = (((model_output-np.min(model_output)) / (np.max(model_output)-np.min(model_output)))*255).astype("uint8")
    colored_depth = cv2.applyColorMap(formatted, cv2.COLORMAP_INFERNO)[:, :, ::-1]
    depth = Image.fromarray(colored_depth)
    return depth


def display_image_with_depth(image:Image, depth1: np.ndarray, depth2: np.ndarray):
    def _prepare_depth(model_output):
        # prepare images for visualization
        model_output =  model_output.reshape(384, 576)
        formatted = (((model_output-np.min(model_output)) / (np.max(model_output)-np.min(model_output)))*255).astype("uint8")
        colored_depth = cv2.applyColorMap(formatted, cv2.COLORMAP_INFERNO)[:, :, ::-1]
        depth = Image.fromarray(colored_depth)
        return depth
    display(make_image_grid([image, _prepare_depth(depth1), _prepare_depth(depth2)], rows=1, cols=3))

def pude_display_image_with_depth(image:Image, depth1: np.ndarray, depth2: np.ndarray):
    def _prepare_depth(model_output):
        # prepare images for visualization
        model_output =  model_output.reshape(384, 576)
        formatted = (((model_output-np.min(model_output)) / (np.max(model_output)-np.min(model_output)))*255).astype("uint8")
        colored_depth = cv2.applyColorMap(formatted, cv2.COLORMAP_INFERNO)[:, :, ::-1]
        depth = Image.fromarray(colored_depth)
        return depth
     
    diff_image = depth1 - depth2
    depth_images = [image, _prepare_depth(depth1), _prepare_depth(depth2), _prepare_depth(diff_image), _prepare_depth(np.abs(diff_image))]
    return make_image_grid_title(depth_images, rows=1, cols=5, titles=["Input", "Depth Anything", "PUDE", "Difference", "Absolute Difference"])
    

def unimatch_display_image_with_depth(image:Image, depth1: np.ndarray, depth2: np.ndarray, shifted_image: Image, scaling_factor: float = 1):
    def _prepare_depth(model_output):
        # prepare images for visualization
        model_output =  model_output.reshape(384, 576)
        formatted = (((model_output-np.min(model_output)) / (np.max(model_output)-np.min(model_output)))*255).astype("uint8")
        colored_depth = cv2.applyColorMap(formatted, cv2.COLORMAP_INFERNO)[:, :, ::-1]
        depth = Image.fromarray(colored_depth)
        return depth
    abs_diff = np.abs(depth1 - depth2)
    return make_image_grid_title([image, _prepare_depth(depth1), shifted_image,  _prepare_depth(depth2), _prepare_depth(abs_diff)], 
                                 rows=1, cols=5, titles=["Original Image", "Parent Depth", f"Shifted Image (factor {scaling_factor:.2f})", "Unimatch Output", "Absolute Difference"])
        
def display_image_depth_shifted_image(image:np.ndarray, image2: np.ndarray, depth: np.ndarray): 
    def _prepare_depth(model_output):
        # prepare images for visualization
        model_output =  model_output.reshape(384, 576)
        formatted = (((model_output-np.min(model_output)) / (np.max(model_output)-np.min(model_output)))*255).astype("uint8")
        colored_depth = cv2.applyColorMap(formatted, cv2.COLORMAP_INFERNO)[:, :, ::-1]
        depth = Image.fromarray(colored_depth)
        return depth
    display(make_image_grid([Image.fromarray(image), _prepare_depth(depth), Image.fromarray(image2)], rows=1, cols=3))


def overlay_color(image, mask, color, alpha):
    """
    Overlay a semi-transparent color over the specified mask region in the image.

    Parameters:
    - image: Original image as a numpy array.
    - mask: Boolean mask indicating the region to overlay the color on.
    - color: RGB color tuple.
    - alpha: Opacity level for the overlay color (0.0 - 1.0).
    
    Returns:
    - Image with the overlay applied.
    """
    # Convert the color to float values
    color_float = np.array(color) * 255
    
    image[mask] = ((1 - alpha) * image[mask] + alpha * color_float).astype(np.uint8)
    
    return image

def make_image_grid(images: List[Image.Image], rows: int, cols: int, resize: int = None) -> Image.Image:
    """
    Prepares a single grid of images. Useful for visualization purposes.
    """
    assert len(images) == rows * cols

    if resize is not None:
        images = [img.resize((resize, resize)) for img in images]

    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))

    return grid

def make_image_grid_title(images: List[Image.Image], titles: List[str], rows: int, cols: int, resize: int = None, font_size: int = 40) -> Image.Image:
    """
    Prepares a single grid of images with titles under each image.
    """
    assert len(images) == rows * cols
    assert len(titles) == rows * cols

    if resize is not None:
        images = [img.resize((resize, resize)) for img in images]

    w, h = images[0].size
    padding = font_size//4 + font_size
    grid = Image.new("RGB", size=(cols * w, rows * (h + padding)))  # Increased height for titles

    draw = ImageDraw.Draw(grid)
    # font = ImageFont.load_default()  # You can choose any font you prefer
    font = ImageFont.truetype("arial.ttf", font_size)  # Change "arial.ttf" to the font file you want to use
    h_ = h+padding

    for i, (img, title) in enumerate(zip(images, titles)):
        img = img.resize((w, h))
        grid.paste(img, box=(i % cols * w, i // cols * h_))
        j = (i // cols)
        draw.text(((i % cols * w), ((i // cols + 1) * (h)+(j*padding)), ((i % cols * w) + w), ((i // cols + 1) * (h)+(j*padding) + padding)),
                   title, fill=(255, 255, 255), font=font)

    return grid

def load_pickle(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data
