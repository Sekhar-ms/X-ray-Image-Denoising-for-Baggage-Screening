# X-ray-Image-Denoising-for-Baggage-Screening
This code is Poisson-Gaussian Denoising for security X-ray images. This is adopted from [Self-Supervised Poisson-Gaussian Denoising](https://arxiv.org/abs/2002.09558#:~:text=Self%2Dsupervised%20models%20for%20denoising,such%20as%20low%2Dlight%20microscopy.) 

# ABSTRACT
The utilization of dual-energy X-ray detection technology in security inspection plays a crucial role in ensuring public safety and preventing crimes. However, the X-ray images generated in such security checks often suffer from substantial noise, degrading the image quality and hindering accurate judgments by security inspectors. Existing deep learning-based denoising methods have limitations, such as reliance on large training datasets and clean reference images, which are not readily available in security inspection scenarios. In this work, we addressed the denoising problem of X-ray images with a Poisson-Gaussian noise model, without requiring clean reference images for training.

To overcome these challenges, we employed the Blindspot neural network, which effectively removes noise from X-ray images. Additionally, we incorporated image fusion algorithms, leveraging image spatial information and wavelet-based fusion rules to enhance visualization. Experimental evaluations on a real X-ray image dataset demonstrated the effectiveness of our approach, achieving favorable BRISQUE scores across different baggage scenes. The denoised images obtained through our method, combined with image fusion, enable security inspectors to make more accurate judgments and enhance public safety in security inspection settings.

We implemented this code for 16-bit grayscale X-ray images.

# Environment
The environment used to run this code is mentioned in ```environment.yml```

# Dependencies
- Python 3   
- tensorflow=2.9.1  
- tensorflow-gpu=2.5.0   
- opencv-python=4.6.0.66   
- tqdm=4.64.1
- MATLAB with Image Processing toolbox
- Windows Operating Syastem

# Training 
python train_fmd.py --path ./dataset --dataset non_fmd --mode uncalib
# Testing
python test1.py --path ./dataset --dataset non_fmd --mode uncalib

# About dataset and data pre-processing
- Create 20 folders with names 1 to 20. According to this code, 1 to 19 folders are used in training, and 20th folders are used for testing.
- During training and testing, the network takes only images of size 512 x 512.
- If your images are not of that shape, you need to mirror pad your images. For our experiment, we mirror-padded our images to 1024 x 1024 and split each image into four 512 x 512 images.
- After denoising those splits you can merge those splits and crop out the mirror padded part.
- Code for mirror padding and splitting is given in ```padding_splitting.ipynb```
- Code for merging the denoised images and cropping the mirror padded part is given in ```merge_crop.ipynb```

# Evaluation metric
- To evaluate the quality of the denoised images, we used the BRISQUE score as our metric.
- For this, we used MATLAB's inbuilt function ```brisque``` from Image processing toolbox.

# Image fusion and pseudo coloring
- Applied two fusion algorithms. One is image fusion using Image spatial Information and the other one is Wavelet-based fusion.
- Image fusion using image spatial information and pseudo coloring is in ```coloring.ipynb```
- And wavelet-based fusion code is in ```wavelet.ipynb```

## References
<a id="1">[1]</a> 
Khademi, Wesley & Rao, Sonia & Minnerath, Clare & Hagen, Guy & Ventura, Jonathan. (2021). Self-Supervised Poisson-Gaussian Denoising. 2130-2138. 10.1109/WACV48630.2021.00218. 

<a id="1">[2]</a> 
Zheng, Yue. “X-Ray Image Processing and Visualization for Remote Assistance of Airport Luggage Screeners.” (2004).

<a id="1">[3]</a> 
A. Mittal, A. K. Moorthy and A. C. Bovik, "No-Reference Image Quality Assessment in the Spatial Domain," in IEEE Transactions on Image Processing, vol. 21, no. 12, pp. 4695-4708, Dec. 2012, doi: 10.1109/TIP.2012.2214050.
