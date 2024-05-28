import numpy as np
import cv2
from IPython.display import display
from PIL import Image

default_image_dim = (576,384)


class Stereo_Pair_Generator:
    def __init__(self, image_dim=default_image_dim, max_disparity=64):
        self.image_dim = image_dim

    def generate_stereo_pair(self, image, disparity, scaling_factor=1):
        """
        Generates a stereo pair by warping the left image to the right image based on the disparity map.
        :param image: A numpy array of shape (height, width, 3) representing the left image.
        :param disparity: A numpy array of shape (height, width) representing the disparity map.
        :return: A tuple containing the left and right images as numpy arrays.
        """

        # Warp the left image to the right image based on the disparity map
        warped_image = self._warp_stereo_image_1(image, disparity, scalingfactor=scaling_factor)
        # Crop the image to remove black regions
        # cropped_image = self._crop_image(warped_image)
        cropped_image = warped_image
        # Inpaint damaged region using Exemplar-based inpainting
        inpainted_img = self._inpaint(cropped_image)
        return inpainted_img

    def _inpaint(self, cropped_image):
        # Create a mask where black pixels are set to 1
        mask = np.all(cropped_image == [0, 0, 0], axis=-1).astype(np.uint8)
        # Inpaint damaged region using Exemplar-based inpainting
        inpainted_img = cv2.inpaint(cropped_image, mask, inpaintRadius=4, flags=cv2.INPAINT_NS)

        return inpainted_img
    
    def _warp_stereo_image_1(self, stereo_image, disparity, scalingfactor=1):
        """
        Warps the left image to the right image based on the disparity map.
        :param stereo_image: A numpy array of shape (height, width, 3) representing the stereo image.
        :param disparity: A numpy array of shape (height, width) representing the disparity map.
        :return: A numpy array of shape (height, width, 3) representing the warped image.
        """
        disparity = disparity.reshape(stereo_image.shape[:-1])
        height, width = disparity.shape
        warped_image = np.zeros_like(stereo_image)
        warped_disparity = np.zeros_like(disparity)

        for y in range(height):
            for x in range(width):
                # Compute the new x-coordinate based on the disparity map
                new_x = int(np.ceil(x - disparity[y, x]*scalingfactor))
                # new_x = int(((new_x/(width-1))-0.5)*2)
                if new_x >= 0 and new_x < width:
                    # check if the new_x already exists in the warped image
                    if warped_disparity[y, new_x] == 0:
                        warped_image[y, new_x] = stereo_image[y, x]
                        warped_disparity[y, new_x] = disparity[y, x]
                    else:
                        # check if the new disparity is smaller than the existing disparity
                        if disparity[y, x] < warped_disparity[y, new_x]:
                            warped_image[y, new_x] = stereo_image[y, x]
                            warped_disparity[y, new_x] = disparity[y, x]

        return warped_image

    def _warp_stereo_image(self, stereo_image, disparity, scalingfactor=1):
            """
            Warps the left image to the right image based on the disparity map.
            :param stereo_image: A numpy array of shape (height, width, 3) representing the stereo image.
            :param disparity: A numpy array of shape (height,width) representing the disparity map.
            :return: A numpy array of shape (height, width, 3) representing the warped image.
            """
            disparity = disparity.reshape(stereo_image.shape[:-1])
            height, width = disparity.shape
            max_disparity = np.max(disparity)
            max_shift = int(max_disparity*scalingfactor)
            new_width = width + 2*max_shift
            warped_image = np.zeros((height, new_width, 3), dtype=np.uint8)

            for y in range(height):
                for x in range(width):
                    # Compute the new x-coordinate based on the disparity map
                    new_x = x + int(disparity[y, x]*scalingfactor)
                    # Copy the pixel from the left image to the warped image
                    warped_image[y, new_x] = stereo_image[y, x]

            # Crop the warped image to remove empty bits on the edges
            # warped_image = warped_image[:, max_shift:max_shift+width]

            return warped_image

    # def _warp_stereo_image(self, stereo_image, disparity, scalingfactor=20):
    #     """
    #     Warps the left image to the right image based on the disparity map.
    #     :param stereo_image: A numpy array of shape (height, width, 3) representing the stereo image.
    #     :param disparity: A numpy array of shape (height*width) representing the disparity map.
    #     :param width: Width of the image.
    #     :return: A numpy array of shape (height, width, 3) representing the warped image.
    #     """
    #     height, width = stereo_image.shape[:-1]
    #     max_disparity = np.max(disparity)
    #     max_shift = int(max_disparity/scalingfactor)
    #     new_width = width + 2*max_shift
    #     warped_image = np.zeros((height, new_width, 3), dtype=np.uint8)

    #     for y in range(height):
    #         for x in range(width):
    #             # Compute the index in the 1D disparity array
    #             index = y * width + x
    #             # Compute the new x-coordinate based on the disparity map
    #             new_x = x + int(disparity[index]/scalingfactor)
    #             # Ensure new_x is within the bounds of the warped image
    #             new_x = min(max_shift + new_x, new_width - 1)
    #             # Copy the pixel from the left image to the warped image
    #             warped_image[y, new_x] = stereo_image[y, x]

    #     return warped_image

    
    def _crop_image(self, image):
    
        def __crop_center(image, target_height, target_width):
            # Get the dimensions of the original image
            image_height, image_width = image.shape[:2]

            # Calculate the starting point for the crop
            start_x = max(0, (image_width - target_width) // 2)
            start_y = max(0, (image_height - target_height) // 2)

            # Calculate the end point for the crop
            end_x = min(image_width, start_x + target_width)
            end_y = min(image_height, start_y + target_height)

            # Perform the crop
            cropped_image = image[start_y:end_y, start_x:end_x]

            return cropped_image
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find contours of non-black regions
        contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
        # Sort contours by area in descending order
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        x, y, w, h = cv2.boundingRect(contours[0])
        # Crop the image using the bounding box
        cropped_image = image[y:y+h, x:x+w]
        # crop the image to the center of the original dimensions
        cropped_image = __crop_center(cropped_image, self.image_dim[1], self.image_dim[0])
        return cropped_image

