import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import display
from typing import List

def produce_images_and_plots(W,H, raw_image, d_D, I, darkest_indices, M):

    # create masks for the darkest points
    darkest_mask = np.zeros(W*H, dtype=bool)
    darkest_mask[darkest_indices] = True
    darkest_mask = darkest_mask.reshape(W, H)
    # create masks for  M set
    M_mask_green = np.zeros(W*H, dtype=bool)
    M_mask_green[M[0,:]] = True
    M_mask_green = M_mask_green.reshape(W, H)
    M_mask_blue = np.zeros(W*H, dtype=bool)
    M_mask_blue[M[1,:]] = True
    M_mask_blue = M_mask_blue.reshape(W, H)

    display(raw_image)
    masks = [darkest_mask, M_mask_green, M_mask_blue]
    mask_colors = [(1,0,0), (0,1,0), (0,0,1)]
    mask_images = [return_mask_as_image(mask) for mask in masks]
    applied_masks = [Image.fromarray(overlay_color(np.array(raw_image), mask=mask, color=color, alpha=0.7)) for mask, color in zip(masks, mask_colors)]
    all_images = mask_images + applied_masks
    make_image_grid(all_images, rows=2, cols=3)

    plt.figure(figsize=(10, 6))
    plt.scatter(d_D[M[0]], I[0, M[0]], c='g', label='G')
    plt.scatter(d_D[M[1]], I[1, M[1]], c='b', label='B')
    plt.xlabel('d_D')
    plt.ylabel('I')
    plt.legend()
    plt.show()

def return_mask_as_image(mask):
    mask = Image.fromarray(mask.astype(np.uint8)*255).convert("RGB")
    return mask

def display_mask_as_image(mask):
    mask = Image.fromarray(mask.astype(np.uint8)*255).convert("RGB")
    display(mask)


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

    display(grid)
