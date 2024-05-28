import numpy as np
from typing import Dict, List, Callable

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
            'accuracy_1.05^3': self.generate_accuracy_function(1.05 ** 3)
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

    def evaluate(self, predicted: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
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

        results: List[float] = [
            function(predicted, ground_truth) for function in self.criteria_functions.values()
        ]
        return np.array(results)
