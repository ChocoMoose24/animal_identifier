## Animal Image Classification — Anya Dash


# What this project is

This is a CNN (Convolutional Neural Network) that has been trained off of the Animal-Image-Dataset(90 animals) on Kaggle. You can upload a photo to the website and the computer will output 3 of the highest probabilities for 3 animals and the highest one is your predicted animal.

# How it works

First, it converts the image to a 3D tensor (H x W x 3 channels ((For RGB values.)))
It does this because each image is not yellow, purple, or orange. They are instead 3 grids stacked on top of each other: A red grid, blue grid, and yellow grid. Each pixel in the grid has a value of 0-255 showing how strong the color is at that specific pixel.

# Filters
Small matrices such as 3 x 3 grids ( Called Kernels or Filters) will slide across the map to calculate activation values. It extracts features such as edges, lines and textures. Kernels/ filters will spot basic things like that.
This now leads us into something called hierarchical feature extraction

# feature maps
A feature map is the output grid of a filter. Feature maps may detect the absence or presence of specific features in the image. High values indicate the specific feature is there, while a lower value indicates that its absence.

# hierarchical feature extraction
The lower layers of a feature map detect things such as lines and edges. The higher layers will detect more complex things, such as part of objects or objects.

We also use Global Average Pooling so that we get the mean of each color, somewhat like a score for each channel (The different grids.) This way rather than an entire grid, you have a singular number. Plus, now there is no need to add any flatten layers.

# Base model
The model uses a base model called MobileV2, trained off of more than 1.4 million images. This is heavily useful because training a model from scratch takes a lot of time and still isn't that accurate. MobileV2 definitely improved its time and accuracy, but the accuracy and loss got really weird at the end. So, to fix this we did something called fine tuning.

# fine tuning
When I first uploaded the base model, we froze each layer, meaning that I made sure the base model was untrainable. But, this built a wall for how far the accuracy could actually go. So, I figured out how many of the layers were the lower level feature maps (18) because I don't want to waste time retraining the line and edge detector, due to the fact they are found in every image so the detector would already be good. So, we froze those layers and started training only the part of objects and object detectors for a more accurate model. This is called fine-tuning.

# Loss chart
<img width="693" height="506" alt="image" src="https://github.com/user-attachments/assets/a3c01fcf-cef8-4155-834c-08c8bfdecb97" />


# Application
## Layout
- Banner: **Animal Image Classification** / by Anya Dash
- Left: upload button (camera or gallery on mobile) + image preview
- Right: top 3 class predictions with probabilities

It works by uploading an image after you wait for the model to load, then the model will predict what animal the image is. You can see the top 3 predictions on the right side.
<img src="blob:chrome-untrusted://media-app/e7e64fbb-079a-4c88-ac97-810c9212b086" alt="Screenshot 2026-07-28 2.21.35 PM.png"/><img width="1488" height="1008" alt="image" src="https://github.com/user-attachments/assets/c739a077-8d1a-4531-ae7d-77f328f59440" />
<img src="blob:chrome-untrusted://media-app/64999211-44b5-46b2-9638-b668a53b9ec7" alt="Image_20260727_124329_373.webp"/><img width="2285" height="1368" alt="image" src="https://github.com/user-attachments/assets/19bd5a1b-7728-497e-bd0a-ca1e9e418b9c" />

Also, the model constantly mistakes my dog for a rhinoceros.

<img src="blob:chrome-untrusted://media-app/ae34b690-4ae0-4d24-833b-4aa5810196e5" alt="Image_20260727_124329_414.webp"/><img width="2178" height="1273" alt="image" src="https://github.com/user-attachments/assets/80aba131-efa9-4588-9995-3e9f1c5e376b" />

<img src="blob:chrome-untrusted://media-app/18e03c67-cfc6-4408-86f5-c9c6fd6b620a" alt="Image_20260727_124329_077.webp"/><img width="2324" height="1374" alt="image" src="https://github.com/user-attachments/assets/01b2375e-44c8-4113-84e4-5d65fee5119c" />






Static web app for animal image classification with TensorFlow.js.
Designed to host on **GitHub Pages**.

