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
from IPython.display import display

# datasets_dir: str = 'Datasets/SeaThruNeRF'
# results_dir: str = 'Results/SeaThruNeRF/DatasetLoader'
# image_dir: str = 'images_wb'
# ground_truth_dir: str = 'depth'
# dataset_names: List[str] = ['Curasao', 'Panama', 'JapaneseGardens-RedSea', 'Q_IUI3-RedSea']
# seed: int = 42
# # max 88 in SeaThruNeRF
# total_num_images: int = 70


datasets_dir: str = 'Datasets/SeaThru_Combined'
results_dir: str = 'Results/SeaThru_Combined/DatasetLoader'
image_dir: str = 'images'
ground_truth_dir: str = 'depth'
dataset_names: List[str] = ['Curasao','D3', 'D5', 'Panama', 'JapaneseGardens-RedSea', 'Q_IUI3-RedSea']
seed: int = 42
# max 88 in SeaThruNeRF
total_num_images: int = 182



class DatasetLoader(Dataset):
    def __init__(self, dataset_dir: str = datasets_dir, total_num_images: int = total_num_images, 
                 img_dim: Tuple[int, int] = default_image_dim,
                 image_dir: str = image_dir, ground_truth_dir: str = ground_truth_dir,
                 dataset_names: List[str] = dataset_names, seed: int = seed, save_ground_truths: bool=False) -> None:
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
        self.save_ground_truths = save_ground_truths
        self.image_paths = self._get_random_image_paths(total_num_images)

   
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
        if self.save_ground_truths:
            ground_truth_paths = []
        for dataset_name in os.listdir(self.dataset_dir):
            # print(dataset_name)
            if dataset_name not in self.dataset_names:
                continue
            dataset_path = self.dataset_dir + '/'+  dataset_name
            raw_image_dir = dataset_path + '/' + self.image_dir
            all_raw_images = os.listdir(raw_image_dir)
            
            # num_images = num_images_per_dataset if dataset_name != self.dataset_names[-1] else total_num_images - len(image_paths)
            num_images = len(all_raw_images) if dataset_name != self.dataset_names[-1] else total_num_images - len(image_paths)
            # print(num_images, len(all_raw_images))
            # print(num_images, total_num_images)
            random_images = np.random.choice(all_raw_images, num_images, replace=False)
            for raw_image in random_images:
                image_path = raw_image_dir + '/' + raw_image
                ground_truth_path = dataset_path + '/' + self.ground_truth_dir + '/depth' + raw_image[:-4] + '.tif'
                image_paths.append(image_path)
                if self.save_ground_truths:
                    ground_truth_paths.append(ground_truth_path)
        # Save image paths to a CSV file for debugging
        current_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        csv_file_name = results_dir + '/' + f'{current_date_time}_seed_{self.seed}.csv'

        if self.save_ground_truths:
            with open(csv_file_name, 'w') as f:
                f.write('Image Name, Image Path, Ground Truth Path\n')
                for image_path, ground_truth_path in zip(image_paths, ground_truth_paths):
                    f.write(f'{os.path.basename(image_path)[:-4]}, {image_path}, {ground_truth_path}\n')
        else:
            with open(csv_file_name, 'w') as f:
                f.write('Image Name, Image Path\n')
                for image_path in image_paths:
                    f.write(f'{os.path.basename(image_path)[:-4]}, {image_path}\n')
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

        def _correct_image(img):
            """
            Correct the brightness of an RGB image.

            Parameters:
                img (numpy.ndarray): Input RGB image.

            Returns:
                numpy.ndarray: Image with corrected brightness.
            """
            # Get the shape of the image
            rows, cols, channels = img.shape

            # Compute the average brightness across all channels
            brightness = np.sum(img) / (255 * rows * cols * channels)

            # Define the target minimum brightness
            minimum_brightness = 0.25

            # Compute the brightness ratio
            ratio = brightness / minimum_brightness

            # If the ratio is greater than or equal to 1, the image is already bright enough
            if ratio >= 1:
                # print("Image already bright enough")
                return img

            # Otherwise, adjust brightness to get the target brightness for each channel
            corrected_img = cv2.convertScaleAbs(img, alpha=1 / ratio, beta=0)

            return corrected_img
        if img_dim is None:
            img_dim = self.img_dim
        image = cv2.imread(path)
        image = cv2.resize(image, img_dim, interpolation=cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = _correct_image(image)
        linear_image = Image.fromarray((image).astype(np.uint8))
        return linear_image
    
    def _linear_png_to_non_linear_png(self, linear_image: Image.Image) -> Image.Image:
        """
        Convert a linear PNG image to a non-linear PNG image with white balance correction and gamma correction.

        Parameters:
            linear_image (Image): Linear PNG image.

        Returns:
            Image: Non-linear PNG image.
        """
        # Perform auto white balance correction
        balanced_image = self._auto_white_balance(linear_image)

        # Apply gamma correction
        gamma_corrected_image = self._apply_gamma_correction(balanced_image, gamma=2.2)

        return gamma_corrected_image

    def _auto_white_balance(self, image):
        """
        Perform auto white balance correction on the image using OpenCV.

        Parameters:
            image (numpy.ndarray): Input image.

        Returns:
            numpy.ndarray: Image with auto white balance correction applied.
        """
        # Convert image to LAB color space
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

        # Split the LAB image into channels
        l_channel, a_channel, b_channel = cv2.split(lab_image)

        # Apply the white balance correction on the L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel_eq = clahe.apply(l_channel)

        # Merge the channels back together
        balanced_lab_image = cv2.merge((l_channel_eq, a_channel, b_channel))

        # Convert the LAB image back to RGB color space
        balanced_rgb_image = cv2.cvtColor(balanced_lab_image, cv2.COLOR_LAB2RGB)

        return balanced_rgb_image

    def _apply_gamma_correction(self, image, gamma=2.2):
        """
        Apply gamma correction to the image.

        Parameters:
            image (numpy.ndarray): Input image.
            gamma (float): Gamma value for correction.

        Returns:
            numpy.ndarray: Image with gamma correction applied.
        """
        # Apply gamma correction
        gamma_corrected_image = np.power(image / 255.0, gamma) * 255.0
        gamma_corrected_image = gamma_corrected_image.astype(np.uint8)

        return gamma_corrected_image

    
    def _format_linear_image(self, image: Image.Image) -> np.ndarray:
        """
        Format the non-linear image.

        Parameters:
            image (Image): Non-linear image.

        Returns:
            np.ndarray: Array representing the non-linear image.
        """
        # I = np.array(image)
        # I = (I - I.min()) / (I.max() - I.min())

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
            # image = cv2.cvtColor(image)
            image = cv2.resize(image, img_dim, interpolation=cv2.INTER_CUBIC)
            image = Image.fromarray((image).astype(np.uint8))
            return image
        
        def _postprocess_arw(raw: rawpy.RawPy, img_dim: Tuple[int, int]) -> Tuple[Image.Image, Image.Image]:
            linear_image = raw.postprocess(use_camera_wb=False, dcb_enhance=False, half_size=False, no_auto_scale=True, 
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
