import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from Pude_training_loop.pude_utils import return_mask_as_image, make_image_grid, overlay_color


def get_medium_transmission_vectorized(d_D, nu, mu):
    """
    Compute medium transmission for each channel across all samples using vectorized operations.

    Parameters:
        d_D (numpy.ndarray): Array of size (samples,) containing depth values.
        nu (numpy.ndarray): Array of size (channels,) containing nu values.
        mu (float): Scalar value of mu.

    Returns:
        numpy.ndarray: Array of size (channels, samples) containing medium transmission for each channel.
    """
    # Expand dimensions of nu to match the shape of d_D
    nu_expanded = np.expand_dims(nu, axis=1)  # Shape: (channels, 1)

    # Compute medium transmission for each channel across all samples using vectorized operations
    medium_transmission = np.exp(-nu_expanded / (d_D + mu))

    return medium_transmission


def compute_lb_loss(I, t_hat, B_infty):
    """
    Compute lower loss function

    Parameters:
        I (numpy.ndarray): Input image array of shape (channels, HW).
        t_hat (numpy.ndarray): Predicted medium transmission array of shape (channels, HW).
        B_infty (numpy.ndarray): Predicted background light array of shape (channels,).

    Returns:
        tuple: numpy.ndarray containing the loss values
    """
    # Compute the intermediate expressions
    intermediate_bl1 = -I + (1 - t_hat) * B_infty[:, np.newaxis]
    intermediate_bl2 = -t_hat - intermediate_bl1

    loss_bl1 = np.sum(np.maximum(intermediate_bl1, 0))/I.shape[1]
    loss_bl2 = np.sum(np.maximum(intermediate_bl2, 0))/I.shape[1]
    
    return loss_bl1, loss_bl2

def compute_lu_loss(I, t_hat_D, t_hat_P, B_infty):
    """
    Compute loss function.

    Parameters:
        I (List[numpy.ndarray]): Input image array of shape (channels, npoints).
        t_hat (List[numpy.ndarray]): Predicted medium transmission array of shape (channels, npoints).
        B_infty (numpy.ndarray): Predicted background light array of shape (channels,).

    Returns:
        tuple: the loss value
    """
    # Compute the intermediate expressions
    loss_lu = 0
    for c in range(len(I)):
        intermediate_lu = - (1-t_hat_P[c]) * B_infty[c] - t_hat_D[c] + I[c]
        loss_lu += np.sum(np.maximum(intermediate_lu, 0))/I[c].shape[0]

    return loss_lu

def find_M_b(I, B_infty_hat, t_hat_D, gamma=0.6):
    """
    Find set M_b based on the given condition.

    Parameters:
        I (numpy.ndarray): Input image array of shape (channels, samples).
        B_infty_hat (numpy.ndarray): Predicted background light array of shape (channels,).
        t_hat_D (numpy.ndarray): Predicted dehazed medium transmission array of shape (channels, samples).
        gamma (float): Threshold hyperparameter.

    Returns:
        numpy.ndarray: Boolean array indicating the set M_b.
    """
    # Compute the expression (B_infty_hat * (1 - t_hat_D)) / I
    expression = (B_infty_hat[:, np.newaxis] * (1 - t_hat_D)) / I

    # Find indices where the expression is greater than or equal to gamma
    M_b_indices = expression >= gamma
    return M_b_indices

def compute_loss_bounds(I, t_hat_P, t_hat_D, B_infty, alphas=[1,1,2], show_MB=False, default_image_dim=(384, 576)):
    """
    Compute bound loss

    Parameters:
        I (numpy.ndarray): Input image array of shape (channels, HW).
        t_hat_P (numpy.ndarray): PuDE Predicted medium transmission array of shape (channels, HW).
        t_hat_D (numpy.ndarray): DPT Predicted medium transmission array of shape (channels, HW).
        B_infty (numpy.ndarray): Predicted background light array of shape (channels,).

    Returns:
        tuple:  the loss values
    """
    
    loss_bl1, loss_bl2 = compute_lb_loss(I, t_hat_P, B_infty)
    M_mask = find_M_b(I, B_infty, t_hat_D)
    I_new = [I[0, M_mask[0]], I[1, M_mask[1]]]
    t_hat_P_new = [t_hat_P[0, M_mask[0]], t_hat_P[1, M_mask[1]]]
    t_hat_D_new = [t_hat_D[0, M_mask[0]], t_hat_D[1, M_mask[1]]]
    loss_bu= compute_lu_loss(I_new, t_hat_D_new, t_hat_P_new, B_infty)

    lb = alphas[0]*loss_bl1 + alphas[1]*loss_bl2 + alphas[2]*loss_bu

    
    if show_MB:
        print(f"L_bl1: {loss_bl1}, L_bl2: {loss_bl2}, L_bu: {loss_bu}")
        print(f"lb: {lb}")
        # print(loss_bl1, loss_bl2, loss_bu)
        images = []
        for i in range(2):
            images.append(return_mask_as_image(M_mask[i].reshape(default_image_dim[1], default_image_dim[0])))
        make_image_grid(images, rows=1, cols=2)
    return lb

