# X-ray-Image-Denoising-for-Baggage-Screening
This code is Poisson-Gaussian Denoising for security X-ray images. This is adopted from Self-Supervised Poisson-Gaussian Denoising(https://arxiv.org/abs/2002.09558#:~:text=Self%2Dsupervised%20models%20for%20denoising,such%20as%20low%2Dlight%20microscopy.) by Wesley Khademi, Sonia Rao, Clare Minnerath, Guy Hagen and Jonathan Ventura. Self-Supervised Poisson-Gaussian Denoising. IEEE Winter Conference on Applications of Computer Vision (WACV) 2021.
# ABSTRACT
The utilization of dual-energy X-ray detection technology in security inspection plays a crucial role in ensuring public safety and preventing crimes. However, the X-ray images generated in such security checks often suffer from substantial noise, degrading the image quality and hindering accurate judgments by security inspectors. Existing deep learning-based denoising methods have limitations, such as reliance on large training datasets and clean reference images, which are not readily available in security inspection scenarios. In this work, we addressed the denoising problem of X-ray images with a Poisson-Gaussian noise model, without requiring clean reference images for training.

To overcome these challenges, we employed the Blindspot neural network, which effectively removes noise from X-ray images. Additionally, we incorporated image fusion algorithms, leveraging image spatial information and wavelet-based fusion rules to enhance visualization. Experimental evaluations on a real X-ray image dataset demonstrated the effectiveness of our approach, achieving favorable BRISQUE scores across different baggage scenes. The denoised images obtained through our method, combined with image fusion, enable security inspectors to make more accurate judgments and enhance public safety in security inspection settings.

We implemented this code 16-bit grayscale X-ray images.

# Environment
Environment used to run this code is mentioned in ```environment.yml```

# Dependencies
Python 3
tensorflow=2.9.1
tensorflow-gpu=2.5.0 
opencv-python=4.6.0.66 
tqdm=4.64.1 
