from typing import List, Tuple
import os
from datetime import datetime
import numpy as np
import cv2
import rawpy
from PIL import Image
from skimage.transform import resize
import tifffile as tiff
from Pude_training_loop.model_training import default_image_dim
from torch.utils.data import Dataset

datasets_dir: str = 'Datasets/SeaThru_old'
results_dir: str = 'Results/SeaThru_old/DatasetLoader'
image_dir: str = 'Raw'
ground_truth_dir: str = 'depth'
dataset_names: List[str] = ['D1', 'D3', 'D5']
seed: int = 42
num_images: int = 100

class DatasetLoader(Dataset):
    def __init__(self, dataset_dir: str = datasets_dir, num_images: int = num_images, 
                 img_dim: Tuple[int, int] = default_image_dim,
                 image_dir: str = image_dir, ground_truth_dir: str = ground_truth_dir,
                 dataset_names: List[str] = dataset_names, seed: int = seed) -> None:
        """
        Initialize DatasetLoader instance.

        Parameters:
            dataset_dir (str): Directory containing the datasets.
            num_images (int): Number of images to load.
            img_dim (Tuple[int, int]): Dimensions to resize the images.
            image_dir (str): Directory containing the images.
            ground_truth_dir (str): Directory containing the ground truth depth maps.
            dataset_names (List[str]): Names of the datasets to include.
            seed (int): Seed for random number generation.
        """
        np.random.seed(seed)
        self.seed = seed
        self.dataset_dir = dataset_dir
        self.img_dim = img_dim
        self.dataset_names = dataset_names
        self.image_dir = image_dir 
        self.ground_truth_dir = ground_truth_dir
        self.image_paths = self._get_random_image_paths(num_images)
   
    def _get_random_image_paths(self, total_num_images: int) -> List[str]:
        """
        Get paths of randomly selected images.

        Parameters:
            total_num_images (int): Total number of images to load.

        Returns:
            List[str]: List of paths to randomly selected images.
        """
        num_images_per_dataset = total_num_images // len(self.dataset_names)
        image_paths = []
        ground_truth_paths = []
        for dataset_name in os.listdir(self.dataset_dir):
            if dataset_name not in self.dataset_names:
                continue
            dataset_path = self.dataset_dir + '/'+  dataset_name
            raw_image_dir = dataset_path + '/' + self.image_dir
            all_raw_images = os.listdir(raw_image_dir)
            num_images = num_images_per_dataset if dataset_name != self.dataset_names[-1] else total_num_images - len(image_paths)
            random_images = np.random.choice(all_raw_images, num_images, replace=False)
            for raw_image in random_images:
                image_path = raw_image_dir + '/' + raw_image
                ground_truth_path = dataset_path + '/' + self.ground_truth_dir + '/depth' + raw_image[:-4] + '.tif'
                image_paths.append(image_path)
                ground_truth_paths.append(ground_truth_path)
        # Save image paths to a CSV file for debugging
        current_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        csv_file_name = results_dir + '/' + f'{current_date_time}_seed_{self.seed}.csv'
        with open(csv_file_name, 'w') as f:
            f.write('Image Name, Image Path, Ground Truth Path\n')
            for image_path, ground_truth_path in zip(image_paths, ground_truth_paths):
                f.write(f'{os.path.basename(image_path)[:-4]}, {image_path}, {ground_truth_path}\n')
        return image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Image.Image]:
        """
        Get an item from the dataset.

        Parameters:
            idx (int): Index of the item.

        Returns:
            Tuple[Image.Image, Image.Image]: Tuple containing the linear and non-linear images.
        """
        linear_image, non_linear_image = self._open_image(self.image_paths[idx], self.img_dim)
        return linear_image, non_linear_image
    

    def get_batch(self, batch_size: int) -> Tuple[List[Image.Image], List[Image.Image]]:
        """
        Get a batch of images from the dataset.

        Parameters:
            batch_size (int): Size of the batch.

        Returns:
            Tuple[List[Image.Image], List[Image.Image]]: Tuple containing the linear and non-linear images.
        """
        linear_images = []
        non_linear_images = []
        for i in range(batch_size):
            linear_image, non_linear_image = self._open_image(self.image_paths[i], self.img_dim)
            linear_images.append(linear_image)
            non_linear_images.append(non_linear_image)
        return linear_images, non_linear_images
    
    def _open_image(self, path: str, img_dim: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Open an image from the given path.

        Parameters:
            path (str): Path to the image file.
            img_dim (Tuple[int, int]): Dimensions to resize the image.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing the non-linear images and linear images.
        """
        if path.endswith('.png'):
            linear_image = self._open_png_image(path, img_dim)
            return np.array(linear_image), self._format_linear_image(linear_image)
        elif path.endswith('.ARW') or path.endswith('.NEF'):
            linear_image, non_linear_image = self._open_raw_image(path, img_dim)
            return np.array(non_linear_image), self._format_linear_image(linear_image)
        return None, None
    
    def _open_png_image(self, path: str, img_dim: Tuple[int, int]) -> Image.Image:
        """
        Open a PNG image from the given path.

        Parameters:
            path (str): Path to the PNG image file.
            img_dim (Tuple[int, int]): Dimensions to resize the image.

        Returns:
            Image.Image: PIL Image object representing the image.
        """
        if img_dim is None:
            img_dim = self.img_dim
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, img_dim, interpolation=cv2.INTER_CUBIC)
        image = Image.fromarray((image).astype(np.uint8))
        return image
    
    def _format_linear_image(self, image: Image.Image) -> np.ndarray:
        """
        Format the non-linear image.

        Parameters:
            image (Image): Non-linear image.

        Returns:
            np.ndarray: Array representing the non-linear image.
        """
        I  = np.array([np.array(image.getchannel('G')).flatten(), np.array(image.getchannel('B')).flatten()])/255.0
        return I
    
    def _open_raw_image(self, path: str, img_dim: Tuple[int, int]) -> Tuple[Image.Image, Image.Image]:
        """
        Open an ARW and NEF image from the given path.

        Parameters:
            path (str): Path to the ARW image file.
            img_dim (Tuple[int, int]): Dimensions to resize the image.

        Returns:
            Tuple[Image.Image, Image.Image]: Tuple containing the linear and non-linear images.
        """
        def _convert_to_image(image: np.ndarray, img_dim: Tuple[int, int]) -> Image.Image:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, img_dim, interpolation=cv2.INTER_CUBIC)
            image = Image.fromarray((image).astype(np.uint8))
            return image
        
        def _postprocess_arw(raw: rawpy.RawPy, img_dim: Tuple[int, int]) -> Tuple[Image.Image, Image.Image]:
            linear_image = raw.postprocess(use_camera_wb=True, dcb_enhance=False, half_size=False, no_auto_scale=True, 
                            no_auto_bright=True, gamma=(1,1))
            non_linear_image = raw.postprocess()
            linear_image, non_linear_image = _convert_to_image(linear_image, img_dim), _convert_to_image(non_linear_image, img_dim)
            return linear_image, non_linear_image
        
        if img_dim is None:
            img_dim = self.img_dim
        raw = rawpy.imread(path)

        linear_image, non_linear_image = _postprocess_arw(raw, img_dim)

        return linear_image, non_linear_image

    def open_depth_map(self, path: str, img_dim: Tuple[int, int]) -> np.ndarray:
        """
        Open a depth map from the given path.

        Parameters:
            path (str): Path to the depth map file.
            img_dim (Tuple[int, int]): Dimensions to resize the depth map.

        Returns:
            np.ndarray: Array representing the depth map.
        """
        if img_dim is None:
            img_dim = self.img_dim
        depth_map = tiff.imread(path)
        depth_map = resize(depth_map, img_dim, anti_aliasing=True)
        return depth_map
