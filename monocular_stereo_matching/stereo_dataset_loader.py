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
from scipy.io import loadmat
import h5py
from Pude_training_loop import pude_utils
from scipy import stats
from matplotlib import pyplot as plt

from Pude_training_loop.pude_utils import get_depth_as_image

datasets_dir: str = 'Datasets/FLSea'
ground_truth_dir: str = 'depth/LFT'
left_image_dir: str = 'imgs/LFT'
right_image_dir: str = 'imgs/RGT'
dataset_names: List[str] = ['rock_garden1']
total_num_pairs: int = 10
seed: int = 42

cam1_file: str = 'Datasets\FLSea\calibration\calibration\CameraParameters1.mat'
cam2_file: str = 'Datasets\FLSea\calibration\calibration\CameraParameters2.mat'
ext_file: str = 'Datasets\FLSea\calibration\calibration\ExtrinsicParameters.mat'


class DatasetLoader(Dataset):
    def __init__(self, dataset_dir: str = datasets_dir, total_num_pairs: int = total_num_pairs, 
                 img_dim: Tuple[int, int] = default_image_dim,
                 left_image_dir: str = left_image_dir, right_image_dir: str = right_image_dir,
                 dataset_names: List[str] = dataset_names, 
                 cam1_file: str = cam1_file, cam2_file: str = cam2_file, ext_file: str = ext_file, 
                 seed: int =seed) -> None:
        """
        Initialize StereoDatasetLoader instance.

        Parameters:
            dataset_dir (str): Directory containing the datasets.
            total_num_pairs (int): Number of stereo image pairs to load.
            img_dim (Tuple[int, int]): Dimensions to resize the images.
            left_image_dir (str): Subdirectory containing the left images.
            right_image_dir (str): Subdirectory containing the right images.
            dataset_names (List[str]): Names of the datasets to include.
            seed (int): Seed for random number generation.
            save_ground_truths (bool): Whether to save ground truth paths to a CSV file for debugging.
        """
        np.random.seed(seed)
        self.dataset_dir = dataset_dir
        self.img_dim = img_dim
        self.dataset_names = dataset_names
        self.left_image_dir = left_image_dir 
        self.right_image_dir = right_image_dir
        self.stereo_params = self._load_all_stereo_params(cam1_file, cam2_file, ext_file)
        self.image_pairs = self._get_random_image_pairs(total_num_pairs)
        self.is_mapping_computed = False
        self.mapping = {'maps': None, 'shift': None}

    
    def _load_all_stereo_params(self, cam1_file: str, cam2_file: str, ext_file: str):
        def _load_stereo_params(params_file: str):
            # Load the .mat file
            mat_file = h5py.File(params_file, 'r')
            # Get the keys (variable names) present in the loaded .mat file
            keys = list(mat_file.keys())
            keys = [k for k in keys if not k.startswith('#')]
            _mat_file = mat_file[keys[0]]
            keys = list(_mat_file.keys())
            dic = {}
            for k in keys:
                dic[k] = _mat_file[k][:]
            
            mat_file.close()

            return dic
        
        def _add_distortion_coeff(cam_params):
            # open cv needs them in k1,k2,p1,p2,k3, where k are radial and p are tangential
            radial_distortion_coeffs = cam_params['RadialDistortion']
            tangential_distortion_coeffs = cam_params['TangentialDistortion']
            distortion_coeffs = [radial_distortion_coeffs[0], radial_distortion_coeffs[1], 
                                  tangential_distortion_coeffs[0], tangential_distortion_coeffs[1], 
                                  radial_distortion_coeffs[2]]
            distortion_coeffs = np.array(distortion_coeffs)
            cam_params['DistortionCoefficient'] = distortion_coeffs
            return cam_params

        cam1_params = _load_stereo_params(cam1_file)
        cam2_params = _load_stereo_params(cam2_file)

        cam1_params = _add_distortion_coeff(cam1_params)
        cam2_params = _add_distortion_coeff(cam2_params)
        extrinsic_params = _load_stereo_params(ext_file)

        stereo_params = {'CameraParameters1': cam1_params, 'CameraParameters2': cam2_params, 'ExtrinsicParameters': extrinsic_params}

        return stereo_params
    



    def _get_random_image_pairs(self, total_num_pairs: int) -> List[Tuple[str, str]]:
        """
        Get paths of sequentially selected stereo image pairs.

        Parameters:
            total_num_pairs (int): Total number of stereo image pairs to load.

        Returns:
            List[Tuple[str, str]]: List of tuples containing paths to sequentially selected stereo image pairs.
        """
        image_pairs = []
        for dataset_name in os.listdir(self.dataset_dir):
            if dataset_name not in self.dataset_names:
                continue
            dataset_path = self.dataset_dir + "/" + dataset_name
            left_image_paths = os.listdir(dataset_path + "/" + self.left_image_dir)
            right_image_paths = os.listdir(dataset_path + "/" + self.right_image_dir)
            depth_map_paths = os.listdir(dataset_path + "/" + ground_truth_dir)
            num_pairs = min(len(left_image_paths), total_num_pairs) if dataset_name != self.dataset_names[-1] else min(len(left_image_paths), max(0, total_num_pairs - len(image_pairs)))
            random_images = np.random.choice(range(len(left_image_paths)), num_pairs, replace=False)
            for pair_idx in random_images:
                left_image_path = dataset_path + "/" +  self.left_image_dir + "/" + left_image_paths[pair_idx]
                right_image_path = dataset_path + "/" + self.right_image_dir + "/" + right_image_paths[pair_idx]
                depth_map_path = dataset_path + "/" + ground_truth_dir + "/" + depth_map_paths[pair_idx]
                image_pairs.append((left_image_path, right_image_path, depth_map_path))
        return image_pairs

    def __len__(self) -> int:
        return len(self.image_pairs)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get a stereo image pair from the dataset.

        Parameters:
            idx (int): Index of the stereo image pair.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing the left and right stereo images.
        """
        left_image, right_image, depth_map = self._open_stereo_image_pair(self.image_pairs[idx], self.img_dim)
        left_image, right_image, depth_map = self.rectify_images(left_image, right_image, depth_map, self.stereo_params)

        # resize the images
        left_image = cv2.resize(left_image, self.img_dim)
        right_image = cv2.resize(right_image, self.img_dim)
        depth_map = resize(depth_map, (self.img_dim[1], self.img_dim[0]), anti_aliasing=True).flatten()
        # display(pude_utils.get_depth_as_image(depth_map))
        disparity_map, valid_indices = self.clean_depth_to_disparity(depth_map, self.stereo_params['CameraParameters1']['IntrinsicMatrix'],
                                                                self.stereo_params['ExtrinsicParameters']['TranslationOfCamera2'])
        return left_image, right_image, disparity_map, valid_indices

    def _open_stereo_image_pair(self, paths: Tuple[str, str], img_dim: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Open a stereo image pair from the given paths.

        Parameters:
            paths (Tuple[str, str]): Paths to the left and right images.
            img_dim (Tuple[int, int]): Dimensions to resize the images.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing the left and right stereo images.
        """
        left_image = self._open_image(paths[0], img_dim)
        right_image = self._open_image(paths[1], img_dim)
        depth_map = self.open_depth_map(paths[2], img_dim)
        return left_image, right_image, depth_map

        
    def _open_image(self, path: str, img_dim: Tuple[int, int]) -> np.ndarray:
        """
        Open an image from the given path.

        Parameters:
            path (str): Path to the image file.
            img_dim (Tuple[int, int]): Dimensions to resize the image.

        Returns:
            np.ndarray: Array representing the image.
        """

        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    

    def rectify_images(self, image_left, image_right, depth, stereo_params):
        # Find the largest non-black rectangle in the image
        def _largest_rectangle(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                max_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(max_contour)
                return x, y, w, h
            return 0, 0, img.shape[1], img.shape[0]

         # Undistort images
        undistorted_image_left = image_left
        undistorted_image_right = image_right
        if self.is_mapping_computed:
            map1x, map1y, map2x, map2y = self.mapping['maps']
            x, y, h, w = self.mapping['shift']
        else:
            # Extract camera parameters and rectification transformations
            camera_matrix_left = stereo_params['CameraParameters1']['IntrinsicMatrix']
            distortion_coeffs_left = stereo_params['CameraParameters1']['DistortionCoefficient']
            camera_matrix_right = stereo_params['CameraParameters2']['IntrinsicMatrix']
            distortion_coeffs_right = stereo_params['CameraParameters2']['DistortionCoefficient']

            R1to2 = stereo_params['ExtrinsicParameters']['RotationOfCamera2']
            T1to2 = stereo_params['ExtrinsicParameters']['TranslationOfCamera2']
            

            # Compute rectification transformations
            R1, R2, P1, P2, Q, validRoi1, validRoi2 = cv2.stereoRectify(camera_matrix_left, distortion_coeffs_left, camera_matrix_right, 
                                                                        distortion_coeffs_right,  (undistorted_image_left.shape[1], undistorted_image_left.shape[0]), 
                                                                        R1to2, T1to2, alpha=1, flags=cv2.CALIB_ZERO_DISPARITY)
            xl, yl, wl, hl = validRoi1
            xr, yr, wr, hr = validRoi2

            x = max(xl, xr)
            w = min(xl + wl, xr + wr) - x

            y = max(yl, yr)
            h = min(yl + hl, yr + hr) - y

            
            # Generate the rectification maps for both cameras
            map1x, map1y = cv2.initUndistortRectifyMap(camera_matrix_left, distortion_coeffs_left, R1, P1, (undistorted_image_left.shape[1], undistorted_image_left.shape[0]), cv2.CV_32FC1)
            map2x, map2y = cv2.initUndistortRectifyMap(camera_matrix_right, distortion_coeffs_right, R2, P2, (undistorted_image_left.shape[1], undistorted_image_left.shape[0]), cv2.CV_32FC1)
            self.mapping['maps'] = (map1x, map1y, map2x, map2y)
            self.mapping['shift'] = (x,y,h,w)


        # Remap the undistorted images to rectify them
        rectified_image_left = cv2.remap(undistorted_image_left, map1x, map1y, interpolation=cv2.INTER_LINEAR)[y:y+h, x:x+w]
        rectified_image_right = cv2.remap(undistorted_image_right, map2x, map2y, interpolation=cv2.INTER_LINEAR)[y:y+h, x:x+w]
        rectified_depth = cv2.remap(depth, map1x, map1y, interpolation=cv2.INTER_LINEAR)[y:y+h, xl:xl+w]
        # display(Image.fromarray(rectified_image_left))
        # display(Image.fromarray(rectified_image_right))
        # display(get_depth_as_image(rectified_depth))
        return rectified_image_left, rectified_image_right, rectified_depth

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
        return depth_map
    
    def clean_depth_to_disparity(self, depth_map, intrinsic_matrix, translation_vector):
        disparity_map = self.depth_to_disparity(depth_map, intrinsic_matrix, translation_vector)
        clean_disparity = self.find_and_remove_outliers(disparity_map, z_threshold=2.5)
        clean_disparity = self.process_disparity_map(clean_disparity, dilation_iterations=2, kernel_size=3)
        valid_indices = clean_disparity > 0
        return clean_disparity, valid_indices

    def depth_to_disparity(self, depth_map, intrinsic_matrix, translation_vector):
        # Extract intrinsic parameters
        fx = intrinsic_matrix[0, 0]
        baseline = np.linalg.norm(translation_vector)
        # Initialize disparity map
        disparity_map = np.zeros_like(depth_map, dtype=np.float32)

        # Calculate disparity using valid depth values
        valid_indices = (depth_map > 0.4) & (depth_map < 12)
        disparity_map[valid_indices] = fx * baseline / depth_map[valid_indices]
        return disparity_map
    
    def process_disparity_map(self, disparity_map, kernel_size=3, dilation_iterations=2):
        # Threshold the edge map to obtain a binary image
        _, binary_disparity = cv2.threshold(disparity_map, 0, 1, cv2.THRESH_BINARY)
        # Invert the binary disparity map
        binary_disparity = 1 - binary_disparity
        
        # Dilate the binary disparity map
        kernel = np.ones((kernel_size, kernel_size), np.uint8)  # Adjust kernel size as needed
        dilated_disparity = cv2.dilate(binary_disparity, kernel, iterations=dilation_iterations)

        # Flatten the dilated disparity map
        dilated_disparity = dilated_disparity.flatten()
        # Set the disparity map to 0 where the dilated disparity map is 1
        disparity_map[dilated_disparity == 1] = 0
        
        # Rescale the disparity map
        min_val = np.min(disparity_map)
        max_val = np.max(disparity_map)
        if max_val > min_val:
            disparity_map = (disparity_map - min_val) / (max_val - min_val)
            disparity_map = disparity_map * 20
        else:
            disparity_map.fill(10)
        return disparity_map
    
    
        
    def find_and_remove_outliers(self, ground_truth: np.ndarray, bins: int = 100, range = (0, 20), z_threshold: float = 3.0, show_plots: bool = False):
        """
        Identify and plot outliers in the ground truth data using IQR and Z-score methods.

        Parameters:
            ground_truth (np.ndarray): Ground truth data.
            bins (int): Number of bins for the histogram.
            range (Tuple[int, int]): Range of the histogram.
            z_threshold (float): Z-score threshold for outlier detection.
            show_plots (bool): Whether to display plots.
        
        Returns:
            np.ndarray: Modified ground truth data.
        """
        
        # Calculate the Interquartile Range (IQR)
        Q1 = np.percentile(ground_truth, 25)
        Q3 = np.percentile(ground_truth, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identify IQR outliers
        iqr_mask = (ground_truth < lower_bound) | (ground_truth > upper_bound)
        iqr_outliers = ground_truth[iqr_mask]
        
        # Calculate Z-scores and identify Z-score outliers
        z_scores = stats.zscore(ground_truth)
        z_mask = np.abs(z_scores) > z_threshold
        z_outliers = ground_truth[z_mask]
        
        if show_plots:
            g_min, g_max = np.min(ground_truth), np.max(ground_truth)
            g_range = (g_min, g_max)
            
            # Plot the original histogram
            plt.hist(ground_truth, bins=bins, range=g_range, alpha=0.5, label='Ground Truth', log=True)
            plt.xlabel('Disparity')
            plt.ylabel('Log Frequency')
            plt.title('Original Histogram')
            plt.legend()
            plt.show()
            
            # Plot the histogram with IQR outliers
            plt.hist(ground_truth, bins=bins, range=g_range, alpha=0.5, label='Ground Truth', log=True)
            plt.hist(iqr_outliers, bins=bins, range=g_range, alpha=0.5, color='r', label='IQR Outliers', log=True)
            plt.xlabel('Disparity')
            plt.ylabel('Log Frequency')
            plt.legend()
            plt.title('Histogram with IQR Outliers')
            plt.show()
            
            # Plot the histogram with Z-score outliers
            plt.hist(ground_truth, bins=bins, range=g_range, alpha=0.5, label='Ground Truth', log=True)
            plt.hist(z_outliers, bins=bins, range=g_range, alpha=0.5, color='g', label='Z-score Outliers', log=True)
            plt.xlabel('Disparity')
            plt.ylabel('Log Frequency')
            plt.legend()
            plt.title('Histogram with Z-score Outliers')
            plt.show()
        
        # Combine outliers from both methods
        combined_mask = iqr_mask | z_mask
        
        # Create modified ground truth by setting outliers to zero
        modified_ground_truth = np.copy(ground_truth)
        modified_ground_truth[combined_mask] = 0
        
        # Rescale the modified ground truth
        min_val = np.min(modified_ground_truth)
        max_val = np.max(modified_ground_truth)
        if max_val > min_val:
            modified_ground_truth = (modified_ground_truth - min_val) / (max_val - min_val)
            modified_ground_truth = modified_ground_truth * (range[1] - range[0]) + range[0]
        else:
            modified_ground_truth.fill((range[1] + range[0]) / 2)
        
        if show_plots:
            # Plot the histogram with outliers set to 0
            plt.hist(modified_ground_truth, bins=bins, range=range, alpha=0.5, label='Modified Ground Truth', log=True)
            plt.xlabel('Disparity')
            plt.ylabel('Log Frequency')
            plt.legend()
            plt.title('Histogram after Removing Outliers')
            plt.show()
        
        return modified_ground_truth


