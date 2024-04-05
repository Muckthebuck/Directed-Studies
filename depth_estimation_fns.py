from transformers import AutoImageProcessor, AutoModelForDepthEstimation, DPTForDepthEstimation
import torch
import numpy as np
from PIL import Image
import rawpy
from IPython.display import display, HTML
import tifffile as tiff
import matplotlib.pyplot as plt
from scipy.stats import linregress
import cv2
from sklearn.preprocessing import normalize
import os
from skimage.transform import resize
import random
from sklearn.cluster import KMeans, DBSCAN
from matplotlib.colors import to_rgb
from PUDE.PUDE_implementation import load_model as load_PUDE_model
from typing import List

seed = 42
np.random.seed(seed)
random.seed(seed)

# # 10 random images from each dataset, pre-selected using random without a seed, so the same images are used for all models
# random_images = {'D1': ['T_S03265.ARW', 'T_S03376.ARW', 'T_S03386.ARW', 'T_S03305.ARW', 'T_S03515.ARW', 'T_S03175.ARW', 'T_S03116.ARW', 'T_S03338.ARW', 'T_S03330.ARW', 'T_S03291.ARW'],
#                  'D3': ['T_S04900.ARW', 'T_S04857.ARW', 'T_S04871.ARW', 'T_S04870.ARW', 'T_S04923.ARW', 'T_S04874.ARW', 'T_S04866.ARW', 'T_S04904.ARW', 'T_S04876.ARW', 'T_S04890.ARW'],
#                  'D5': ['LFT_3384.NEF', 'LFT_3381.NEF', 'LFT_3396.NEF', 'LFT_3392.NEF', 'LFT_3375.NEF', 'LFT_3414.NEF', 'LFT_3388.NEF', 'LFT_3385.NEF', 'LFT_3380.NEF', 'LFT_3412.NEF']}
# models and their paths, pude has a local path, the rest are from huggingface
models = {"pude": "PUDE/weightsave/final2/finall2.pth", "depth_anything": "nielsr/depth-anything-small",  "dpt3_1": "Intel/dpt-swinv2-large-384"}
default_image_dim = (576,384)
datasets_dir = 'Datasets/SeaThru'
results_dir = 'Results/SeaThru'
nsample_images = 10


random.seed(seed)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
print("Device is", device)

def display_image_small(im, width=500):
    w, h = im.size
    height = int(h * (width / w))
    image = im.resize((width, height))
    display(image)

def display_depth(depth):
    formatted = (depth * 255 / np.max(depth)).astype("uint8")
    colored_depth = cv2.applyColorMap(formatted, cv2.COLORMAP_INFERNO)[:, :, ::-1]
    depth = Image.fromarray(colored_depth)
    display(depth)

def open_image(path, depth_path, result_ground_truth_image_path,img_dim, save_ground_truth=False, display=False):
    depth_img = tiff.imread(depth_path)
    depth_img = resize(depth_img, (img_dim[1], img_dim[0]), order=1, mode='constant', anti_aliasing=False)
    formatted = (depth_img * 255 / np.max(depth_img)).astype("uint8")
    colored_depth = cv2.applyColorMap(formatted, cv2.COLORMAP_INFERNO)[:, :, ::-1]
    depth = Image.fromarray(colored_depth)
    if save_ground_truth:
        # save the ground truth image
        depth.save(result_ground_truth_image_path, format='png')

    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, img_dim, interpolation=cv2.INTER_CUBIC)
    image= Image.fromarray((image).astype(np.uint8))
    return image, depth_img

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

def post_process_depth(depth, image, predicted_depth_path, display=False):
    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    # visualize the prediction
    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    colored_depth = cv2.applyColorMap(formatted, cv2.COLORMAP_INFERNO)[:, :, ::-1]
    depth = Image.fromarray(colored_depth)
    depth.save(predicted_depth_path, format='png')
    if display:
        display_image_small(depth)
    return output, colored_depth, depth


def plot_one_over_z_vs_d(actual_depth, model_output, save_folder, img_name):
    # Remove zero values
    z = actual_depth.flatten()
    d = model_output.flatten()

    non_zero = np.where(z > 0.33)
    z = 1/z[non_zero]
    d = d[non_zero]

    # normalise d
    d = d/np.max(d)

    plt.figure()
    # Scatter plot
    plt.scatter(z, d, s=1)
    plt.xlabel("1/z")
    plt.ylabel("d")
   
    # Linear regression
    result = linregress(z, d)
    
    # Line of best fit
    fit_x = np.linspace(np.min(z), np.max(z), 100)
    fit_y = result.slope * fit_x + result.intercept
    # new figure
  
    plt.plot(fit_x, fit_y, '-r', label='Line of best fit, r = {:.3f}'.format(result.rvalue))
    
    # Display R-squared value
    plt.legend()
    plt.title(f"1/z vs d for {img_name}")
    # plt.show( )
    plt.savefig(save_folder, dpi=100, bbox_inches='tight')
    plt.close()
    return result.rvalue

def plot_residuals(actual_depth, model_output, img_name, save_folder, plot_without_noise_path):
    # Remove zero values
    z = actual_depth.flatten()
    d = model_output.flatten()
    # print(z.shape)
    non_zero_mask = z > 0.33
    # non_zero = np.where(z > 0.33)
    z = 1/z[non_zero_mask]
    d = d[non_zero_mask]

    # normalise d
    d = d/np.max(d)

    # Linear regression
    result = linregress(z, d)
    r_value = result.rvalue

    # Calculate residuals
    residuals = d - result.slope * z - result.intercept

    abs_residuals = np.abs(residuals)
    threshold = 3.5 * np.std(abs_residuals)

    # Identify the outliers
    outliers_mask = abs_residuals > threshold
    outliers = np.column_stack((z[outliers_mask], residuals[outliers_mask]))
    combined_mask = combine_masks([non_zero_mask, outliers_mask])
    # print(combined_mask.shape)
    # Apply k-means clustering with k=3 to the outliers
    # kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(outliers)
    dbscan = DBSCAN(eps=0.2, min_samples=10).fit(outliers)

    # Plot the outliers
    plt.figure()
    # plt.scatter(z, residuals, color='blue', label='Data points', s=0.5) 
    # 10 colours
    n_clusters = len(set(dbscan.labels_)) if -1 not in dbscan.labels_ else len(set(dbscan.labels_)) - 1
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, n_clusters)]
    # append black for outliers 
    if -1 in dbscan.labels_:
        colors.append((0, 0, 0, 1))
    rgb_colors = [to_rgb(color) for color in colors]


    clusters_masks = []
    for i in set(dbscan.labels_):
        cluster_mask = dbscan.labels_ == i
        clusters_masks.append(combine_masks([combined_mask, cluster_mask]).reshape(actual_depth.shape))
        plt.scatter(outliers[cluster_mask, 0], outliers[cluster_mask, 1], color=colors[i], label=f'Cluster {i}', s=0.5)
 
    # plt.scatter(z[outliers_mask], residuals[outliers_mask], color='red', label='Outliers', s=0.5)
    plt.xlabel('1/z')
    plt.ylabel('Residuals')
    plt.legend()
    plt.title(f'Outliers for {img_name}')
    # plt.show()
    plt.savefig(save_folder, dpi=100, bbox_inches='tight')
    plt.close()

    # now remove the noise cluster and replot the linear regression
    if -1 in dbscan.labels_:
        noise_mask = dbscan.labels_ == -1
        non_noise_mask = ~combine_masks([outliers_mask, noise_mask])
        z = z[non_noise_mask]
        d = d[non_noise_mask]
        result = linregress(z, d)
        r_value = result.rvalue
        fit_x = np.linspace(np.min(z), np.max(z), 100)
        fit_y = result.slope * fit_x + result.intercept
        plt.figure()
        plt.scatter(z, d, s=1)
        plt.plot(fit_x, fit_y, '-r', label='Line of best fit, r = {:.3f}'.format(result.rvalue))
        plt.xlabel("1/z")
        plt.ylabel("d")
        plt.legend()
        plt.title(f"1/z vs d for {img_name} without noise")
        plt.savefig(plot_without_noise_path, dpi=100, bbox_inches='tight')
        plt.close()

    return clusters_masks, rgb_colors, r_value
    

def combine_masks(masks)-> np.ndarray:
    m = masks[0].copy()
    for m2 in masks[1:]:
        # Combine masks
        j = 0
        for i in range(len(m)):
            if m[i] == True:
                m[i] = m[i] and m2[j]
                j+=1
    return m

def underwater_depth_model_analysis(model, image_processor, image_name, dataset_name, raw_image_path, actual_depth_path, results_path, img_dim=(664,443), save_ground_truth=False, display=False):
    # initialise model
    plot_path = f'{results_path}/plot/{image_name}.PNG'
    result_ground_truth_image_path = f'Datasets/{dataset_name}/depth_png/{image_name}.png'
    predicted_depth_path = f'{results_path}/predicted_depth/{image_name}.png'
    outlier_cluster_path = f'{results_path}/outlier_cluster/{image_name}.PNG'
    outlier_image_path = f'{results_path}/outlier_image/{image_name}.PNG'
    plot_without_noise_path = f'{results_path}/plot_without_noise/{image_name}.PNG'
  
    raw_image, actual_depth = open_image(path=raw_image_path, depth_path=actual_depth_path, 
                                         result_ground_truth_image_path=result_ground_truth_image_path, 
                                         img_dim=img_dim, save_ground_truth=save_ground_truth, display=display)
    inputs = preprocess_image(image_processor=image_processor, image=raw_image)
    predicted_depth = predict_depth(model=model, inputs=inputs)
    model_output, formatted, depth_im = post_process_depth(depth=predicted_depth, image=raw_image, predicted_depth_path=predicted_depth_path, display=display)
    rvalue = plot_one_over_z_vs_d(actual_depth, model_output, plot_path, image_name)
    outlier_cluster_masks, colors, r_value_without_noise = plot_residuals(actual_depth, model_output, image_name, outlier_cluster_path, plot_without_noise_path)
    display_outliers_on_image(np.array(raw_image), outlier_cluster_masks, image_name, outlier_image_path, colors=colors)
    return rvalue, r_value_without_noise

# def display_outliers_on_image(image, outlier_cluster_masks, img_name, save_folder, colors):
#     # change the colours of the pixels in the image
#     # red, yellow, pink, black
#     for i, mask in enumerate(outlier_cluster_masks):
#         # print((np.array(colors[i])*255).astype(int))
#         # print(np.rint(np.ndarray(colors[i])* 255).astype(int), type(np.rint(np.ndarray(colors[i])* 255).astype(int)))
#         image[mask] = (np.rint(np.array(colors[i])*255)).astype(int)
#     img = Image.fromarray(image)
#     img.save(save_folder, format='png')

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

def display_outliers_on_image(image, outlier_cluster_masks, img_name, save_folder, colors, alpha=0.7):
    """
    Overlay semi-transparent colors over specified regions in the image based on outlier cluster masks.

    Parameters:
    - image: Original image as a numpy array.
    - outlier_cluster_masks: List of boolean masks indicating the regions to overlay the colors on.
    - img_name: Name of the image (not used in this function).
    - save_folder: Folder path to save the resulting image (not used in this function).
    - colors: List of RGB color tuples corresponding to the colors to overlay.
    - alpha: Opacity level for the overlay colors (0.0 - 1.0).
    
    Returns:
    - Image with the overlay applied.
    """
    for mask, color in zip(outlier_cluster_masks, colors):
        image = overlay_color(image, mask, color, alpha)
    
    img = Image.fromarray(image)
    img.save(save_folder, format='png')
    

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


def process_datasets(datasets_dir, results_dir, model_name, nsample_images, random_images, image_processor, model, default_image_dim, save_ground_truth):
    results_paths = {}
    mean_r_values = {}
    mean_r_values_without_noise = {}
    raw_image_count = 0

    for dataset_dir in os.listdir(datasets_dir):
        dataset_path = datasets_dir + '/' + dataset_dir
        dataset_name = dataset_dir
        if dataset_name =='D1':
            continue
        results_paths[dataset_name] = results_dir + '/' + model_name + '/' + dataset_name
        
        if not os.path.exists(results_paths[dataset_name]):
            os.makedirs(results_paths[dataset_name] + '/plot')
            os.makedirs(results_paths[dataset_name] + '/predicted_depth')
            os.makedirs(results_paths[dataset_name] + '/outlier_cluster')
            os.makedirs(results_paths[dataset_name] + '/outlier_image')
            os.makedirs(results_paths[dataset_name] + '/plot_without_noise')
        
        raw_image_dir = dataset_path + '/linearPNG'
        depth_image_dir = dataset_path + '/depth'
        
        if os.path.isdir(raw_image_dir):
            mean_r_values[(model_name, dataset_name)] = 0
            mean_r_values_without_noise[(model_name, dataset_name)] = 0
            
            if dataset_name not in random_images.keys():  
                all_raw_images = os.listdir(raw_image_dir)
                random_images[dataset_name] = random.sample(all_raw_images, nsample_images)
                
            for raw_image in random_images[dataset_name]:
                raw_image_path = raw_image_dir + '/' + raw_image
                raw_image_name = raw_image[:-4]
                depth_image_path = depth_image_dir + '/depth' + raw_image_name + '.tif'
    
                r_value, r_value_without_noise = underwater_depth_model_analysis(
                    model=model, 
                    image_processor=image_processor, 
                    image_name=raw_image_name, 
                    dataset_name=dataset_name, 
                    raw_image_path=raw_image_path, 
                    actual_depth_path=depth_image_path, 
                    results_path=results_paths[dataset_name], 
                    img_dim=default_image_dim, 
                    save_ground_truth=save_ground_truth, 
                    display=False
                )
                mean_r_values[(model_name, dataset_name)] += r_value
                mean_r_values_without_noise[(model_name, dataset_name)] += r_value_without_noise
                raw_image_count += 1
                
                if raw_image_count % 10 == 0:
                    print(f"Processed {raw_image_count} images.")
                    
            mean_r_values[(model_name, dataset_name)] /= nsample_images
            mean_r_values_without_noise[(model_name, dataset_name)] /= nsample_images
            
    return mean_r_values, mean_r_values_without_noise


