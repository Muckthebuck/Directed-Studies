import csv
import pickle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from Pude_training_loop.pude_utils import make_image_grid, return_mask_as_image, overlay_color
from Pude_training_loop.loss_functions import get_medium_transmission_vectorized


class Data_logger:
    def __init__(self, results_dir) -> None:
        self.results_dir = results_dir
        self.results_images_dir  = f"{results_dir}/images"
        self.results_params_dir = f"{results_dir}/params"
        self.result_images=[]
        self.parameter_results=[]
        self.i=0

    def update_current_idx(self, i):
        self.i = i

    def insert_image_result(self, image):
        pass

    def log_data(self,param):
        self.parameter_results.append(param)
        self.i+=1
        
    def save_images(self, i):
        pass
    
    def save_data(self):
        # Save parameter results in CSV file
        csv_filename = f"{self.results_params_dir}/underwater_parameter_results.csv"
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Index', 'Parameter(nu, mu, Binfty, valid_estimate)'])
            for idx, param in enumerate(self.parameter_results):
                writer.writerow([idx, param])

        # Save parameter results in pickle file
        pickle_filename = f"{self.results_params_dir}/parameter_results.pickle"
        with open(pickle_filename, 'wb') as file:
            pickle.dump(self.parameter_results, file)


    def save_M_plots(self, W,H, raw_image, d_D, I, M, M_idx, hat_nu, hat_mu, hat_B_infty):
        image_grid = self.produce_images_and_plots(W,H, raw_image, d_D, I, M, hat_nu, hat_mu, hat_B_infty)
        image_grid.save(f"{self.results_images_dir}/{self.i}_M_{M_idx}_plots.png")

    def save_tau_plots(self, W,H, I_new):
        images = []
        for i in range(I_new.shape[0]):
            images.append(Image.fromarray((I_new[0].reshape(W, H)*255.0).astype(np.uint8)))
        image_grid = make_image_grid(images, rows=1, cols=len(images))
        image_grid.save(f"{self.results_images_dir}/{self.i}_tau_plots.png")
         

    def produce_images_and_plots(self, W,H, raw_image, d_D, I, M,  hat_nu, hat_mu, hat_B_infty) -> Image.Image:
        # create masks for  M set
        M_mask_green = np.zeros(W*H, dtype=bool)
        M_mask_green[M[0,:]] = True
        M_mask_green = M_mask_green.reshape(W, H)
        M_mask_blue = np.zeros(W*H, dtype=bool)
        M_mask_blue[M[1,:]] = True
        M_mask_blue = M_mask_blue.reshape(W, H)
        masks = [M_mask_green, M_mask_blue]
        mask_colors = [(0,1,0), (0,0,1)]
        mask_images = [return_mask_as_image(mask) for mask in masks]
        applied_masks = [Image.fromarray(overlay_color(np.array(raw_image), mask=mask, color=color, alpha=0.7)) for mask, color in zip(masks, mask_colors)]
        # Create scatter plot
        dpi = 100
        plt.figure(figsize=(H/dpi, W/dpi))
        plt.scatter(d_D[M[0]], I[0, M[0]], c='g', label='G')
        plt.scatter(d_D[M[1]], I[1, M[1]], c='b', label='B')
        
        # create l
        x_vals = np.linspace(np.min(d_D), np.max(d_D), 800)
        pred_I = hat_B_infty[:, np.newaxis] * (1-get_medium_transmission_vectorized(x_vals, hat_nu, hat_mu))
        # 
        plt.plot(x_vals, pred_I[0], c='m', label='Fitted green channel')
        plt.plot(x_vals, pred_I[1], c='c', label='Fitted blue channel')

        plt.xlabel('d_D')
        plt.ylabel('I')
        plt.legend()
        # Save scatter plot as an image
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        scatter_plot_img = Image.open(buf)
        # Close the plot to prevent it from displaying
        plt.close()
        raw_image_ = Image.fromarray(raw_image)
        all_images = [scatter_plot_img] + mask_images + [raw_image_]  + applied_masks
        image_grid = make_image_grid(all_images, rows=2, cols=3)
        return image_grid
    
