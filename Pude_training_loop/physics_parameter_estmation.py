import numpy as np
from scipy.optimize import minimize
from typing import List, Tuple

class UnderwaterParameterFinder:
    def __init__(self, farthest_points_percentage: float =  0.15, 
                 darkest_points_percentage: float = 0.20, 
                 M_range: Tuple[float, float] = (10, 40), 
                 N_1: int = 500, N_2: int = 200):
        self.farthest_points_percentage = farthest_points_percentage
        self.darkest_points_percentage = darkest_points_percentage
        self.M_range = M_range
        self.N_1 = N_1
        self.N_2 = N_2
    
    def algorithm_1(self, d_D: np.ndarray, I: np.ndarray):
        hat_nu, hat_mu, hat_B_infty= self.find_underwater_parameters(d_D, I, N=self.N_1)
        
        hat_nu, hat_mu, hat_B_infty = self.find_underwater_parameters(d_D, I, N=self.N_2, 
                                                                      initial_v=hat_nu, initial_mu=hat_mu, 
                                                                      initial_B_infty=hat_B_infty, keep_B_infty=True)
        return hat_nu, hat_mu, hat_B_infty
    
    def find_underwater_parameters(self, d_D: np.ndarray, I: np.ndarray, N: int, initial_v=[1, 1], initial_mu=1, initial_B_infty=[1, 1], keep_B_infty=False):
        # Step 1: Find farthest 15% points
        farthest_indices = self._find_farthest_points(d_D)
        
        # Step 2: Select darkest 20% points per channel
        darkest_indices = self._select_darkest_points(I, farthest_indices)

        # Step 3: Compute B_infty for each channel
        if not keep_B_infty:
            initial_B_infty = self._compute_B_infty(I, darkest_indices)
        
        # Step 4: Select N darkest points from each channel
        M = self._find_set_M(I,d_D,percentage_range=self.M_range, num_points=N)
    # Select elements from I using indices in M
        I_new = []
        for i in range(len(M)):
            I_new.append(I[i, M[i]])
        I_new = np.array(I_new)

        # Step 5: Solve least-squares problem
        hat_nu, hat_mu, hat_B_infty = self._solve_optimisation_problem(d_D[M], I_new, initial_v=initial_v, initial_mu=initial_mu,
                                                                initial_B_infty=initial_B_infty, keep_B_infty=keep_B_infty)
        
        return hat_nu, hat_mu, hat_B_infty
    
    def _find_farthest_points(self, d_D: np.ndarray) -> np.ndarray:
        """
        Find the indices of the farthest points in the depth data.

        Parameters:
            d_D (np.ndarray): Array containing depth data.
 
        Returns:
            np.ndarray: Array containing the indices of the farthest points.
        """
        num_samples = d_D.shape[0]
        num_points = int(num_samples * (self.farthest_points_percentage))
        farthest_indices = np.argsort(d_D)[:num_points]
        return farthest_indices

    def _select_darkest_points(self, I: np.ndarray, farthest_indices: np.ndarray) -> np.ndarray:
        """
        Select the darkest points from the farthest points per channel.

        Parameters:
            I (np.ndarray): Array containing image data.
            farthest_indices (np.ndarray): Indices of the farthest points.
            percentage (float): Percentage of darkest points to select. Default is 20.

        Returns:
            np.ndarray: Array containing the indices of the darkest points per channel.
        """
        num_samples = len(farthest_indices)
        num_points = int(num_samples * self.darkest_points_percentage)
        darkest_indices = []
        for i in range(I.shape[0]):
            sorted_indices = farthest_indices[np.argsort(I[i, farthest_indices])]
            darkest_indices.append(sorted_indices[:num_points])
        return np.array(darkest_indices)

    def _compute_B_infty(self, I: np.ndarray, darkest_indices: np.ndarray) -> np.ndarray:
        """
        Compute B_infty for each channel.

        Parameters:
            I (np.ndarray): Array containing image data.
            darkest_indices (np.ndarray): Array containing the indices of the darkest points per channel.

        Returns:
            np.ndarray: Array containing B_infty values for each channel.
        """
        B_infty = np.zeros(I.shape[0])
        for i in range(I.shape[0]):
            B_infty[i] = np.median(I[i, darkest_indices[i]])
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

    def _find_set_M(self, I: np.ndarray, d: np.ndarray, percentage_range: Tuple[float, float] = (10, 40), num_points: int = 10) -> np.ndarray:
        """
        Find a set of points M within a specified range and select the N darkest points from each channel.

        Parameters:
            I (np.ndarray): Array containing image data.
            d (np.ndarray): Array containing depth data.
            percentage_range (Tuple[float, float]): Range of percentiles to consider. Default is (10, 40).
            num_points (int): Number of darkest points to select. Default is 10.

        Returns:
            np.ndarray: Array containing the indices of selected points.
        """
        points_within_range = self._select_closest_points_within_range(d, percentage_range)
        selected_indices = []
        for i in range(len(I)):
            sorted_indices = points_within_range[np.argsort(I[i, points_within_range])]
            selected_indices.append(sorted_indices[:num_points])
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
        

