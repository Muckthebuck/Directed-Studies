import numpy as np
from scipy.optimize import minimize
from typing import List, Tuple

from Pude_training_loop.model_training import default_image_dim
from Pude_training_loop.loss_functions import get_medium_transmission_vectorized
from Pude_training_loop.data_logger import Data_logger

class UnderwaterParameterFinder:
    def __init__(self, farthest_points_percentage: float =  0.15, 
                 darkest_points_percentage: float = 0.20, 
                 M_range: Tuple[float, float] = (5, 100), 
                 N_1: int = 500, N_2: int = 200, tau_thresholds: Tuple[float, float] = (0.1, 0.4)):
        self.farthest_points_percentage = farthest_points_percentage
        self.darkest_points_percentage = darkest_points_percentage
        self.M_range = M_range
        self.N_1 = N_1
        self.N_2 = N_2
        self.tau_thresholds = tau_thresholds
    
    def algorithm_1(self, d_D: np.ndarray, I: np.ndarray, data_logger: Data_logger = None, non_linear_image = None):
        d = np.array([d_D, d_D])
        hat_nu, hat_mu, hat_B_infty, M_1 = self.find_underwater_parameters(d, I, N=self.N_1, cluster_size=5)
        
   
        tau = get_medium_transmission_vectorized(d, hat_nu, hat_mu)
        # tau_threshold1 = self.get_tau_threshold(hat_nu)
        # tau_threshold2  = self.get_tau_threshold(hat_nu, a=10, b=1.01)
        # tau_mask =  (tau[:,]<tau_threshold1[:, np.newaxis]) | (tau[:,]>tau_threshold2[:, np.newaxis])
        # tau_threshold1 = self.tau_thresholds[0]
        # tau_threshold2 = self.tau_thresholds[1]
        # tau_mask = (tau[:,]<tau_threshold1) | (tau[:,]>tau_threshold2)
        tau_mask = tau<self.tau_thresholds[0]
        
        I_new = I.copy()    
        I_new[tau_mask] = 1.0
        d[tau_mask] = 0.0

        if data_logger is not None:
            data_logger.save_M_plots(W=default_image_dim[1], H=default_image_dim[0], raw_image=non_linear_image, d_D=d_D, I=I, M=M_1, M_idx=1, hat_nu=hat_nu, hat_mu=hat_mu, hat_B_infty=hat_B_infty)
            data_logger.save_tau_plots(W=default_image_dim[1], H=default_image_dim[0], I_new=I_new)
        
        # find number of pixels left after filtering in each channel
        t = tau.shape[1] - np.sum(tau_mask, axis=1)
        # check if there are enough pixels left to run the second optimization step
        if np.sum((t-self.N_2)>=0)<tau.shape[0]:
            # cant run the second optimization step, due to not enough pixels
            # currently just returning results from the first step, with flag
            return hat_nu, hat_mu, hat_B_infty, False
        
        hat_nu_2, hat_mu, hat_B_infty, M_2 = self.find_underwater_parameters(d, I_new, N=self.N_2, 
                                                                      initial_v=hat_nu, initial_mu=hat_mu, 
                                                                      initial_B_infty=hat_B_infty, keep_B_infty=True, 
                                                                      cluster_radius=40, cluster_size=2)
        
        if data_logger is not None:
            data_logger.save_M_plots(W=default_image_dim[1], H=default_image_dim[0], raw_image=non_linear_image, d_D=d_D, I=I, M=M_2, M_idx=2, hat_nu=hat_nu_2, hat_mu=hat_mu, hat_B_infty=hat_B_infty)

        # check the new estimation isnt trash, 
        # can happen if the image has dark spots in a small clustered area
        # check the hat_nu_2 is not too far from hat_nu
        if np.linalg.norm(hat_nu - hat_nu_2) > 10:
            hat_nu_3, hat_mu, hat_B_infty, M_3 = self.find_underwater_parameters(d, I_new, N=self.N_2, 
                                                                      initial_v=hat_nu, initial_mu=hat_mu, 
                                                                      initial_B_infty=hat_B_infty, keep_B_infty=True, 
                                                                      cluster_radius=20, cluster_size=5)
            if data_logger is not None:
                data_logger.save_M_plots(W=default_image_dim[1], H=default_image_dim[0], raw_image=non_linear_image, d_D=d_D, I=I, M=M_3, M_idx=3, hat_nu=hat_nu_3, hat_mu=hat_mu, hat_B_infty=hat_B_infty)
            
            hat_nu = np.minimum(hat_nu, hat_nu_2)
            hat_nu = np.minimum(hat_nu, hat_nu_3)
        
        return hat_nu, hat_mu, hat_B_infty, True
    
    def find_underwater_parameters(self, d_D: np.ndarray, I: np.ndarray, N: int, 
                                   initial_v=[1, 1], initial_mu=1, initial_B_infty=[1, 1], 
                                   keep_B_infty=False, cluster_radius=48, cluster_size=12):
        
        darkest_indices = None
        if not keep_B_infty:
            # Step 1: Find farthest 15% points
            farthest_indices = self._find_farthest_points(d_D)
            # Step 2: Select darkest 20% points per channel
            darkest_points = self._select_darkest_points(I, farthest_indices)
             # Step 3: Compute B_infty for each channel
            initial_B_infty = self._compute_B_infty(darkest_points)
        
        # Step 4: Select N darkest points from each channel
        M = self._find_set_M(I=I, d_D=d_D, N=N, M_range=self.M_range, 
                             default_image_dim=default_image_dim, 
                             cluster_radius=cluster_radius, cluster_size=cluster_size)
        # Select elements from I using indices in M
       
        I_new = self.select_elements_with_indices(I, M)
        d_new = self.select_elements_with_indices(d_D, M)

        # Step 5: Solve least-squares problem
        hat_nu, hat_mu, hat_B_infty = self._solve_optimisation_problem(d_new, I_new, initial_v=initial_v, initial_mu=initial_mu,
                                                                initial_B_infty=initial_B_infty, keep_B_infty=keep_B_infty)
        
        return hat_nu, hat_mu, hat_B_infty, M
    
    def get_tau_threshold(self, x: np.ndarray, a: np.ndarray = 0.7, b: float=1) -> np.ndarray:
        """
        calculates tau threshold based on calcualted nu for each channel
        x: nu 
        Evaluate the function f(x) = (-b*x) / (x + a) + 1 for vectors x and a.
        
        Parameters:
            x (float): Input vector.
            a (float): Parameter a.
            b (float): Parameter b.
        
        Returns:
            np.ndarray: Output vector.
        """
        return (-b*x) / (x + a) + 1
    
    def select_elements_with_indices(self, I, M):
        # Create an array of indices corresponding to the rows of I
        row_indices = np.arange(len(M))
        # Use advanced indexing to select elements from I using indices in M
        I_new = I[row_indices[:, None], M]
        return I_new

    def _find_farthest_points(self, d_D: np.ndarray) -> np.ndarray:
        """
        Find the indices of the farthest points in the depth data.

        Parameters:
            d_D (np.ndarray): Array containing depth data.
 
        Returns:
            np.ndarray: Array containing the indices of the farthest points.
        """
        num_samples = d_D.shape[1]
        num_points = int(num_samples * (self.farthest_points_percentage))
        farthest_indices = np.argsort(d_D, axis=1)[:, :num_points]
        return farthest_indices

    def _select_darkest_points(self, I: np.ndarray, farthest_indices: np.ndarray) -> np.ndarray:
        """
        Select the darkest points from the farthest points per channel.

        Parameters:
            I (np.ndarray): Array containing image data.
            farthest_indices (np.ndarray): Indices of the farthest points.
            percentage (float): Percentage of darkest points to select. Default is 20.

        Returns:
            np.ndarray: Array containing the darkest points per channel.
        """
        num_points = int(farthest_indices.shape[1] * self.darkest_points_percentage)
        darkest_points = np.sort(np.take_along_axis(I, farthest_indices, axis=1), axis=1)[:, :num_points]
        return darkest_points

    def _compute_B_infty(self, darkest_indices: np.ndarray) -> np.ndarray:
        """
        Compute B_infty for each channel.

        Parameters:
            darkest_indices (np.ndarray): Array containing the darkest points per channel.

        Returns:
            np.ndarray: Array containing B_infty values for each channel.
        """
        B_infty = np.median(darkest_indices, axis=1)
        return B_infty

    def _select_closest_points_within_range(self, d: np.ndarray, percentage_range: Tuple[float, float] = (10, 40)) -> np.ndarray:
        """
        Select points within a specified range of percentiles in the depth data.

        Parameters:
            d (np.ndarray): Array containing depth data.
            percentage_range (Tuple[float, float]): Range of percentiles to consider. Default is (10, 40).

        Returns:
            np.ndarray: Array containing the indices of points within the specified range.
        """
        high_percentile = np.percentile(d, 100 - percentage_range[0])
        low_percentile = np.percentile(d, 100 - percentage_range[1])
        within_range_indices = np.where((d >= low_percentile) & (d <= high_percentile))[0]
        return within_range_indices

    # def _find_set_M(self, I: np.ndarray, d: np.ndarray, percentage_range: Tuple[float, float] = (10, 40), num_points: int = 10) -> np.ndarray:
    #     """
    #     Find a set of points M within a specified range and select the N darkest points from each channel.

    #     Parameters:
    #         I (np.ndarray): Array containing image data.
    #         d (np.ndarray): Array containing depth data.
    #         percentage_range (Tuple[float, float]): Range of percentiles to consider. Default is (10, 40).
    #         num_points (int): Number of darkest points to select. Default is 10.

    #     Returns:
    #         np.ndarray: Array containing the indices of selected points.
    #     """
    #     points_within_range = self._select_closest_points_within_range(d, percentage_range)
    #     selected_indices = []
    #     for i in range(len(I)):
    #         sorted_indices = points_within_range[np.argsort(I[i, points_within_range])]
    #         selected_indices.append(sorted_indices[:num_points])
    #     return np.array(selected_indices)

    def _find_set_M(self, I: np.ndarray, d_D: np.ndarray, N: int, cluster_radius: int = 48, 
                cluster_size: int = 12, M_range: tuple = (10, 60), 
                default_image_dim: tuple = (576, 384), is_verbose: bool = False) -> np.ndarray:
        """
        Selects a set of N darkest pixels for each channel based on intensity and depth criteria.

        Parameters:
            I (np.ndarray): Intensity values of shape (num_channels, height, width).
            d_D (np.ndarray): Depth values of shape (num_channels, height, width).
            N (int): Number of darkest pixels to select for each channel.
            cluster_radius (int, optional): Maximum radius to consider for clustering. Defaults to 48.
            cluster_size (int, optional): Maximum size of clusters to consider. Defaults to 12.
            M_range (tuple, optional): Range of percentile values to consider for depth filtering. Defaults to (10, 60).
            default_image_dim (tuple, optional): Default dimensions of the image. Defaults to (576, 384).
            is_verbose (bool, optional): Whether to print verbose output. Defaults to False.

        Returns:
            np.ndarray: Selected indices of shape (num_channels, N).
        """
        num_channels = I.shape[0]
        reshaped_I = I.reshape(num_channels, default_image_dim[1], default_image_dim[0])
        
        selected_indices = []
        for channel in range(num_channels):
            # Sort the pixel values for the current channel 
            darkest_indices = np.unravel_index(np.argsort(reshaped_I[channel].flatten()), reshaped_I[channel].shape)
            # Keep track of selected pixels 
            selected_pixels = [darkest_indices[0][0] * reshaped_I.shape[2] + darkest_indices[1][0]]
            
            selected_coords = np.array([[darkest_indices[0][0], darkest_indices[1][0]]])

            # Keep track of the number of pixels selected within max_distance for each pixel
            pixels_within_distance_count = np.zeros_like(reshaped_I[channel], dtype=int)
            _percentile_h = 100 - M_range[1] 
            _percentile_l = 100 - M_range[0] 
            indx_wtih_d_D_max = np.where(d_D[channel] != 0.0)
            _cluster_radius = cluster_radius
            _cluster_size = cluster_size
            
            while len(selected_pixels) < N:
                # Calculate percentiles 
                percentile_l = np.percentile(d_D[channel][indx_wtih_d_D_max], _percentile_l)
                percentile_h = np.percentile(d_D[channel][indx_wtih_d_D_max], _percentile_h)
                
                # Iterate through remaining pixels to check distance
                for i in range(1, len(darkest_indices[0])):
                    current_row, current_col = darkest_indices[0][i], darkest_indices[1][i]
                    current_pixel = current_row * reshaped_I.shape[2] + current_col
                    current_coord = (current_row, current_col)

                    if current_pixel in selected_pixels or I[channel, current_pixel] == 1.0:
                        continue
                    
                    # Calculate Euclidean distances between current pixel and selected pixels
                    distances = np.sqrt(np.sum((selected_coords - current_coord) ** 2, axis=1))

                    # Count the number of pixels within max_distance
                    pixels_within_distance = np.sum(distances <= _cluster_radius)

                    is_valid_d_D = (d_D[channel, current_pixel] > percentile_h) and (d_D[channel, current_pixel] < percentile_l)

                    # Check if the number of pixels within max_distance exceeds cluster_size
                    if is_valid_d_D and pixels_within_distance <= _cluster_size:
                        selected_pixels.append(current_pixel)
                        selected_coords = np.append(selected_coords, [current_coord], axis=0)
                        
                        # Update the count of pixels selected within max_distance for each pixel
                        pixels_within_distance_count[current_row, current_col] = pixels_within_distance

                        # Break if N pixels are selected
                        if len(selected_pixels) == N:
                            # remove the outliers from the selected pixels
                            M = np.array(selected_pixels)
                            x = np.exp(-1/(1+d_D[channel, M]))
                            lingress = np.polyfit(x, I[channel, M], 1)
                            # find the residuals
                            residuals = I[channel, M] - (lingress[0]*x + lingress[1])
                            # find the outliers
                            not_outliers = np.where(residuals < 0.08)
                            selected_pixels = M[not_outliers].tolist()
                            if len(selected_pixels) == N:
                                break
                
                if len(selected_pixels) < N:
                    if is_verbose:
                        print(f"Altering params for channel: {channel}, selected: {len(selected_pixels)}/{N}")
                    if _percentile_h == 1 and _percentile_l == 98:
                        if len(selected_pixels) < N:
                            if is_verbose:
                                print("Increasing Cluster size")
                            if _cluster_size == cluster_size*4:
                                if is_verbose:
                                    print("Decreasing cluster max radial")
                                if _cluster_radius == (cluster_radius)//2:
                                    if is_verbose:
                                        print("Could not find enough pixels")
                                    # Find the selected pixels not in the darkest indices
                                    M = np.array(selected_pixels)
                                    darkest_indices_flatten = darkest_indices[0] * reshaped_I.shape[2] + darkest_indices[1]
                                    not_in_darkest_indices = np.setdiff1d(darkest_indices_flatten, M)
                                    # Append the remaining pixels from the darkest indices to get N pixels
                                    selected_pixels = np.append(M, not_in_darkest_indices[:N - len(selected_pixels)]).tolist()
                                    break
                                _cluster_radius = max(_cluster_radius - 4, (cluster_radius // 2))
                            _cluster_size = min((cluster_size * 4), _cluster_size + 2)
                    _percentile_h  = max(1, _percentile_h - 4)
                    _percentile_l  = min(98, _percentile_l + 2)
                    if is_verbose:
                        print(f"percentile_h: {_percentile_h}, percentile_l: {_percentile_l}, cluster_radius: {_cluster_radius}, cluster_size: {_cluster_size}")
      
            selected_indices.append(selected_pixels)

        return np.array(selected_indices)



    def _solve_optimisation_problem(self, d, I, initial_v=[1, 1], initial_mu=1, initial_B_infty=[1, 1], keep_B_infty=False):
        """
        Optimize parameters nu, mu, and B_infty for both G and B.

        Parameters:
        d: list
            List of distances
        I: list
            List of arrays for I_G and I_B.
        initial_v: list, optional
            Initial guess for the nu. Default is [1, 1].
        initial_mu: float, optional
            Initial guess for the mu. Default is 1.
        initial_B_infty: list, optional
            Initial guess for the B_infty. Default is [1, 1].

        Returns:
        hat_nu: numpy array
            Optimized values of nu for G and B.
        hat_mu: float
            Optimized value of mu.
        hat_B_infty: numpy array
            Optimized values of B_infty for G and B.
        """
        num_channels = I.shape[0]
        # Vectorized objective function
        def __objective(params):
            nu = params[:num_channels]
            mu = params[num_channels]
            if keep_B_infty:
                B_infty = initial_B_infty
            else:
                B_infty = params[num_channels + 1:]
            # Calculate predicted values
            total_epsilon = np.sum(np.abs(I - B_infty[:, np.newaxis] * (1 - np.exp(-nu[:, np.newaxis] / (d + mu)))))
            return total_epsilon
        

        initial_guess = np.concatenate((initial_v, [initial_mu], initial_B_infty))
        result = minimize(__objective, initial_guess, method='BFGS')
        hat_nu = result.x[:num_channels]
        hat_mu = result.x[num_channels]
        if keep_B_infty:
            hat_B_infty = initial_B_infty
        else:
            hat_B_infty = result.x[num_channels + 1:]
        return hat_nu, hat_mu, hat_B_infty
        

