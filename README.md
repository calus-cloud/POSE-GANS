# PoseGAN – Pose-Based Human Image Generation

## Objective
The objective of this project is to implement Deformable GANs for generating person images conditioned on a given pose.
Specifically, given an image of a person and a target pose, the goal is to synthesize a new image of that same person in the specified pose.

## Approach
1) We started by learning the basic concepts of Neural Networks and Machine Learning.


2) Using this knowledge, we implemented a handwritten digit recognition model (MNIST dataset) using simple 1- and 2-layer Artificial Neural Networks (ANN) with NumPy from scratch.


3) We learned the basic concepts of optimizers, hyperparameter tuning, and Convolutional Neural Networks (CNNs), and studied various architectures.


4) Then, we implemented the MNIST model using the PyTorch framework — first with a 2-layer ANN and then with a deep CNN architecture.


5) We further implemented an object detection model using the CIFAR-10 dataset with a deep CNN architecture, including batch normalization and dropout regularization.


6) Next, we implemented Generative Adversarial Networks (GANs) to generate digits on the MNIST dataset, and later extended it to Conditional GANs (cGANs) on the same dataset.


7) Finally, we implemented the Pose-based Human Image Generation architecture in PyTorch for the Market-1501 dataset, taking reference from a research paper-https://openaccess.thecvf.com/content_cvpr_2018/papers/Siarohin_Deformable_GANs_for_CVPR_2018_paper.pdf

## Custom Dataset – MarketPoseDataset
This project includes a custom PyTorch dataset class called MarketPoseDataset, designed for pose-guided image generation.
The dataset follows a structure where each pair consists of images of the same person in different poses, along with their corresponding pose maps.


Each dataset item returns a dictionary containing:

Source image

Target image

Source pose map

Target pose map


Pose Heatmaps
We extract 18 human joint keypoints from each image and use YOLO-based pose estimation to generate corresponding heatmaps.
 These heatmaps are then converted into tensors for training.
Pairing
We use the Market-1501 dataset from Kaggle.
It contains multiple images of the same person.
We create pairs of two images per person (source and target) based on their IDs.

## Model Architecture Overview
This project implements a pose-guided person image generation model using a Generative Adversarial Network (GAN) framework.

It includes two main components:

Generator (Pose-Guided Deformable Generator): Generates realistic images of a person in the desired target pose.

Discriminator: Evaluates the generated images to ensure they look realistic and preserve the person’s identity.

Together, these components enable the model to synthesize high-quality person images conditioned on a given pose.

MODEL ARCHITECTURE
<center><img src="./thumbnails/model architecture.jpg"></center>


