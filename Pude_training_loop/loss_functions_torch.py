from typing import List, Tuple
import torch
import torch.nn as nn
from Pude_training_loop.model_training import default_image_dim


class PUDELoss(nn.Module):
    def __init__(self, alphas: List[int] = [1, 1, 2], betas: List[int] = [5, 100, 0], trimming_rate: float = 0.3):
        super(PUDELoss, self).__init__()
        self.alphas = alphas
        self.betas = betas
        self.trimming_rate = trimming_rate

    def forward(self, d_P: torch.Tensor, d_D: torch.Tensor, I: torch.Tensor, t_hat_nu: torch.Tensor, t_hat_mu: torch.Tensor, B_infty: torch.Tensor) -> torch.Tensor:
        """
        Compute bound loss

        Parameters:
            d_P (torch.Tensor): Tensor of shape (samples,) containing PUDE depth values.
            d_D (torch.Tensor): Tensor of shape (samples,) containing DPT/Depthanything depth values.
            I (torch.Tensor): Input image tensor of shape (channels, HW).
            t_hat_nu (torch.Tensor): Predicted nu tensor of shape (channels,).
            t_hat_mu (torch.Tensor): Predicted mu tensor of shape (channels, ).
            B_infty (torch.Tensor): Predicted background light tensor of shape (channels,).

        Returns:
            torch.Tensor:  the loss values
        """
        similarity_loss = self.trimmed_similarity_loss(d_daughter=d_P, d_parent=d_D)
        edge_aware_smoothness_loss = self.edge_aware_smoothness_loss(d_P, I)
        if self.betas[1]!=0:
            t_hat_D = self.get_medium_transmission_vectorized_torch(d_D, t_hat_nu, t_hat_mu)
            t_hat_P = self.get_medium_transmission_vectorized_torch(d_P, t_hat_nu, t_hat_mu)
            bound_loss = self.compute_loss_bounds_torch(I, t_hat_P, t_hat_D, B_infty, self.alphas)
            loss = self.betas[0] * similarity_loss + self.betas[1] * bound_loss + self.betas[2] * edge_aware_smoothness_loss
        else:
            loss = self.betas[0]*similarity_loss + self.betas[2]*edge_aware_smoothness_loss
        # loss.requires_grad = True
        return loss
    
    def compute_gradients(self, tensor: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
        dy = tensor[:, 1:, :] - tensor[:, :-1, :]
        dx = tensor[:, :, 1:] - tensor[:, :, :-1]
        return dx, dy


    def edge_aware_smoothness_loss(self, d, I):
        # Reshape d and I to their 2D spatial dimensions
        d = d.view(1, default_image_dim[1], default_image_dim[0])
        I = I.view(2, default_image_dim[1], default_image_dim[0])
        # Mean-normalize the inverse depth
        mean_d = torch.mean(d, dim=[1, 2], keepdim=True)
        d_star = d / mean_d
        
        # Compute gradients of mean-normalized inverse depth
        d_star_dx, d_star_dy = self.compute_gradients(d_star)
        
        # Compute gradients of the image
        I_dx, I_dy = self.compute_gradients(I)
        
        # Compute the exponent of the negative absolute gradients of the image
        weight_x = torch.exp(-torch.abs(I_dx))
        weight_y = torch.exp(-torch.abs(I_dy))
        
        # Compute the smoothness terms
        smoothness_x = torch.abs(d_star_dx) * weight_x
        smoothness_y = torch.abs(d_star_dy) * weight_y
        # Calculate the smoothness loss
        loss = smoothness_x.mean() + smoothness_y.mean()

        return loss

    
    def trimmed_similarity_loss(self, d_daughter: torch.Tensor, d_parent: torch.Tensor) -> torch.Tensor:
        """
        Compute the trimmed similarity loss.

        Parameters:
            d_daughter (torch.Tensor): Tensor of shape (samples,) containing daughter depth values.
            d_parent (torch.Tensor): Tensor of shape (samples,) containing parent depth values.

        Returns:
            torch.Tensor: Trimmed median loss value.
        """
        # Calculate absolute differences between daughter and parent depths
        abs_diff = torch.abs(d_daughter - d_parent)

        # Sort the absolute differences in decending order
        sorted_diff, sorted_indices = torch.sort(abs_diff, descending=True)
     
        # Calculate the number of pixels to not trim
        num_pixels = int((1-self.trimming_rate) * len(sorted_diff))

        # trim the top N% pixels
        sorted_diff = sorted_diff[:num_pixels]

        # calculate the median of the trimmed differences
        median_diff = torch.median(sorted_diff)

        # Divide absolute differences by median of parent depths
        scaled_diff = abs_diff / (torch.finfo(torch.float32).eps+median_diff)

        # take the mean of the scaled differences
        loss = torch.mean(scaled_diff)

        return loss
    
    def compute_loss_bounds_torch(self, I: torch.Tensor, t_hat_P: torch.Tensor, t_hat_D: torch.Tensor, B_infty: torch.Tensor, alphas: List[int] = [1,1,2]) -> torch.Tensor:
        """
        Compute bound loss

        Parameters:
            I (torch.Tensor): Input image tensor of shape (channels, HW).
            t_hat_P (torch.Tensor): PuDE Predicted medium transmission tensor of shape (channels, HW).
            t_hat_D (torch.Tensor): DPT Predicted medium transmission tensor of shape (channels, HW).
            B_infty (torch.Tensor): Predicted background light tensor of shape (channels,).

        Returns:
            torch.Tensor:  the loss values
        """
        loss_bl1, loss_bl2 = self.compute_lb_loss_torch(I, t_hat_P, B_infty)
        M_mask = self.find_M_b_torch(I, B_infty, t_hat_D)
        M_nums = torch.sum(M_mask, dim=1)
        loss_bu = self.compute_lu_loss_torch(I*M_mask, t_hat_D*M_mask, t_hat_P*M_mask, B_infty, M_nums)
        lb = alphas[0]*loss_bl1 + alphas[1]*loss_bl2 + alphas[2]*loss_bu
        return lb

    
    def get_medium_transmission_vectorized_torch(self, d_D: torch.Tensor, nu: torch.Tensor, mu: torch.float) -> torch.Tensor:
        """
        Compute medium transmission for each channel across all samples using vectorized operations.

        Parameters:
            d_D (torch.Tensor): Tensor of size (samples,) containing depth values.
            nu (torch.Tensor): Tensor of size (channels,) containing nu values.
            mu (float): Scalar value of mu.

        Returns:
            torch.Tensor: Tensor of size (channels, samples) containing medium transmission for each channel.
        """
        # Expand dimensions of nu to match the shape of d_D
        nu_expanded = nu.unsqueeze(1)  # Shape: (channels, 1)
        # Compute medium transmission for each channel across all samples using vectorized operations
        medium_transmission = torch.exp(-nu_expanded / (d_D + mu))

        return medium_transmission


    def compute_lb_loss_torch(self, I: torch.Tensor, t_hat: torch.Tensor, B_infty: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute lower loss function

        Parameters:
            I (torch.Tensor): Input image tensor of shape (channels, HW).
            t_hat (torch.Tensor): Predicted medium transmission tensor of shape (channels, HW).
            B_infty (torch.Tensor): Predicted background light tensor of shape (channels,).

        Returns:
            tuple: torch.Tensor containing the loss values
        """
        # Compute the intermediate expressions
        intermediate_bl1 = -I + (1 - t_hat) * B_infty[:, None]
        intermediate_bl2 = -t_hat - intermediate_bl1

        loss_bl1 = torch.sum(torch.relu(intermediate_bl1))/I.shape[1]
        loss_bl2 = torch.sum(torch.relu(intermediate_bl2))/I.shape[1]
        
        return loss_bl1, loss_bl2


    def compute_lu_loss_torch(self, I: torch.Tensor, t_hat_D: torch.Tensor, t_hat_P: torch.Tensor, B_infty: torch.Tensor, n_valid_points: torch.Tensor) -> torch.Tensor:
        """
        Compute upper bound loss function.

        Parameters:
            I (torch.Tensor]): Input image tensor of shape (channels, npoints).
            t_hat_D (torch.Tensor): Predicted dehazed medium transmission tensor of shape (channels, npoints).
            t_hat_P (torch.Tensor): Predicted medium transmission tensor of shape (channels, npoints).
            B_infty (torch.Tensor): Predicted background light tensor of shape (channels,).
            n_valid_points (torch.Tensor): Number of valid points tensor of shape (channels,).
        Returns:
            torch.Tensor: the loss value
        """
        # Compute the intermediate expressions
        intermediate_lu = - (1 - t_hat_P) * B_infty - t_hat_D + I
        # Sum over channels
        loss_lu = torch.sum(torch.sum(torch.relu(intermediate_lu), dim=1) / n_valid_points)
        return loss_lu


    def find_M_b_torch(self, I: torch.Tensor, B_infty_hat: torch.Tensor, t_hat_D: torch.Tensor, gamma: float = 0.6) -> torch.Tensor:
        """
        Find set M_b based on the given condition.

        Parameters:
            I (torch.Tensor): Input image tensor of shape (channels, samples).
            B_infty_hat (torch.Tensor): Predicted background light tensor of shape (channels,).
            t_hat_D (torch.Tensor): Predicted dehazed medium transmission tensor of shape (channels, samples).
            gamma (float): Threshold hyperparameter.

        Returns:
            torch.Tensor: Boolean tensor indicating the set M_b.
        """
        # Compute the expression (B_infty_hat * (1 - t_hat_D)) / I
        expression = (B_infty_hat * (1 - t_hat_D)) / (I+torch.finfo(torch.float32).eps)

        # Find indices where the expression is greater than or equal to gamma
        # Use ReLU to threshold the expression
        M_b_indices = torch.relu(expression - gamma)
        M_b_indices = 1-torch.relu(1-M_b_indices)
        return M_b_indices

