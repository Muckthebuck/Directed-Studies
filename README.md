# Self Supervised Underwater Stereo Depth Estimation

![image](https://github.com/user-attachments/assets/776f497a-9853-42b1-9a19-1b82a70eb1ad)

Acquiring high-quality training data for underwater depth estimation has posed numerous challenges, and due to the domain gap, in-air models cannot be directly applied to underwater scenarios. To tackle this problem, we introduce a pipeline that enables a novel transfer learning-based method to fine-tune stereo depth estimation models specifically for underwater environments. In-air methods have already shown significant improvement by incorporating synthetic labels for unlabelled data produced using existing depth estimation models. We incorporate this approach in our pipeline to generate stereo pairs from monocular datasets, utilizing a fine-tuned underwater monocular model to produce ground truth labels. To evaluate the zero-shot capabilities of our approach, we randomly selected 100 images from each of the four publicly available datasets provided in FLSea. Our fine-tuned stereo model demonstrates improved zero-shot performance on these datasets, showcasing increased depth resolution and accuracy in underwater scenarios.




Full report available at [Full_Report](Directed_Studies__Self_Supervised_Underwater_Stereo_Depth_Estimation_1172562.pdf)
