import numpy as np
from typing import Dict, List, Callable
from scipy.stats import linregress
import matplotlib.pyplot as plt

class StereoEval:
    """
    Class for evaluating stereo model output against ground truth using various metrics.
    """

    def __init__(self) -> None:
        """
        Initialize the StereoEval class.
        """
        self.criteria_functions: Dict[str, Callable] = {
            'abs_rel_error': self.abs_rel_error,
            'squared_rel_error': self.squared_rel_error,
            'rmse': self.rmse,
            'rmse_log': self.rmse_log,
            'silog': self.silog,
            'accuracy_1.05': self.generate_accuracy_function(1.05),
            'accuracy_1.05^2': self.generate_accuracy_function(1.05 ** 2),
            'accuracy_1.05^3': self.generate_accuracy_function(1.05 ** 3),
            'r_value': self.r_value
        }

    def abs_rel_error(self, predicted: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        Compute the mean absolute value of the relative error (AbsRel).

        Args:
            predicted: Predicted values.
            ground_truth: Ground truth values.

        Returns:
            AbsRel value.
        """
        abs_rel = np.abs(predicted - ground_truth) / ground_truth
        return np.mean(abs_rel)

    def squared_rel_error(self, predicted: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        Compute the squared relative error (Sq. Rel).

        Args:
            predicted: Predicted values.
            ground_truth: Ground truth values.

        Returns:
            Sq. Rel value.
        """
        squared_rel = ((predicted - ground_truth) / ground_truth) ** 2
        return np.mean(squared_rel)

    def rmse(self, predicted: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        Compute the root mean square error (RMSE).

        Args:
            predicted: Predicted values.
            ground_truth: Ground truth values.

        Returns:
            RMSE value.
        """
        rmse = np.sqrt(np.mean((predicted - ground_truth) ** 2))
        return rmse

    def rmse_log(self, predicted: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        Compute the root mean square logarithmic error (RMSE log).

        Args:
            predicted: Predicted values.
            ground_truth: Ground truth values.

        Returns:
            RMSE log value.
        """
        rmse_log = np.sqrt(np.mean((np.log(predicted) - np.log(ground_truth)) ** 2))
        return rmse_log

    def silog(self, predicted: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        Compute the scale invariant logarithmic error (SILog).

        Args:
            predicted: Predicted values.
            ground_truth: Ground truth values.

        Returns:
            SILog value.
        """
        ei = np.log(predicted) - np.log(ground_truth)
        silog = np.sqrt(np.mean(ei ** 2) - (np.mean(ei) ** 2))
        return silog

    def generate_accuracy_function(self, threshold: float) -> Callable:
        """
        Generate a function to compute accuracy with a specified threshold.

        Args:
            threshold: Threshold value.

        Returns:
            Accuracy function.
        """
        def accuracy(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
            """
            Compute the accuracy with a specified threshold.

            Args:
                predicted: Predicted values.
                ground_truth: Ground truth values.

            Returns:
                Accuracy value.
            """
            threshold_values = np.maximum(predicted / ground_truth, ground_truth / predicted)
            acc = np.mean(threshold_values < threshold)
            return acc

        return accuracy


    def r_value(self, predicted: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        Compute the rvalue between predicted and ground truth values.

        Args:
            predicted: Predicted values.
            ground_truth: Ground truth values.
        
        Returns:
            R value.
        """
        # remove outliers using the residuals of the predicted and ground truth values
        def _fit_lingress(z,d):
            # Linear regression
            result = linregress(z, d)
            # Line of best fit
            fit_x = np.linspace(np.min(z), np.max(z), 100)
            fit_y = result.slope * fit_x + result.intercept
            return result, fit_x, fit_y
            
        def _remove_outlier(z,d):
            # Linear regression
            result, fit_x, fit_y = _fit_lingress(z,d)
            # Calculate residuals
            residuals = d - result.slope * z - result.intercept

            abs_residuals = np.abs(residuals)
            threshold = 4 * np.std(abs_residuals)

            # Identify the non-outliers
            non_outliers_mask = abs_residuals < threshold

            return z[non_outliers_mask], d[non_outliers_mask]
        if self.rplot_path is not None:
            z, d = _remove_outlier(ground_truth, predicted)
            # Linear regression
            result, fit_x, fit_y = _fit_lingress(z,d)
            r = result.rvalue
      
            # plot the data
            # Scatter plot
            plt.scatter(z, d, s=1)
            plt.xlabel("1/z")
            plt.ylabel("d")
        
            plt.plot(fit_x, fit_y, '-r', label='Line of best fit, r = {:.3f}'.format(result.rvalue))
            plt.ylim(bottom=0)
            # Display R-squared value
            plt.legend()
            plt.title(f"1/z vs d, r = {result.rvalue:.3f}")
            # save the plot with a new name for each image
            plt.savefig(self.rplot_path)
            # close the plot
            plt.close()
        else:
            r = 0
        return r

        

    def evaluate(self, predicted: np.ndarray, ground_truth: np.ndarray, rplot_path: str) -> np.ndarray:
        """
        Evaluate the model output against ground truth using all criteria functions.

        Args:
            predicted: Predicted values.
            ground_truth: Ground truth values.

        Returns:
            NumPy array containing all evaluation metric values.
        """
        def _ensure_non_zero(arr1, arr2):
            non_zero_mask = arr1 > 0
            return arr1[non_zero_mask], arr2[non_zero_mask]

        predicted, ground_truth = _ensure_non_zero(predicted, ground_truth)
        ground_truth, predicted = _ensure_non_zero(ground_truth, predicted)
        self.rplot_path = rplot_path
        results: List[float] = [
            function(predicted, ground_truth) for function in self.criteria_functions.values()
        ]
        return np.array(results)
