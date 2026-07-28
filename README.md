# Animal Image Classification — Anya Dash

According to scientists, animals are multicellular organisms belonging to the kingdom Animalia. They eat organic material, breathe oxygen, are capable of movement, reproduce sexually, and undergo a specific developmental stage, known as the blastula, during embryonic growth.

Source: Elephango

# What this project is

This is a CNN (Convolutional Neural Network) that has been trained off of the Animal-Image-Dataset(90 animals) on Kaggle.

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
# Our previous loss graph: 
<img src="blob:chrome-untrusted://media-app/01448396-ed4c-42d2-b619-56d5dfedf2da" alt="Screenshot 2026-07-28 2.07.12 PM.png"/><img width="727" height="503" alt="image" src="https://github.com/user-attachments/assets/26e17bc8-47f9-4522-aa6b-f231f4592d03" />

# fine tuning
When I first uploaded the base model, we froze each layer, meaning that I made sure the base model was untrainable. But, this built a wall for how far the accuracy could actually go. So, I figured out how many of the layers were the lower level feature maps (18) because I don't want to waste time retraining the line and edge detector, due to the fact they are found in every image so the detector would already be good. So, we froze those layers and started training only the part of objects and object detectors for a more accurate model. This is called fine-tuning.

# Our new loss chart (Yes, I ran for more epochs.)
<img src="blob:chrome-untrusted://media-app/22b53570-d475-4d74-ba6f-5d5ca10966e2" alt="Screenshot 2026-07-27 1.23.50 PM.png"/><img width="683" height="512" alt="image" src="https://github.com/user-attachments/assets/29291d12-fe85-4cf8-8672-b31009082332" />



Static web app for animal image classification with TensorFlow.js.
Designed to host on **GitHub Pages**.

## Layout

- Banner: **Animal Image Classification** / by Anya Dash
- Left: upload button (camera or gallery on mobile) + image preview
- Right: top 3 class predictions with probabilities
