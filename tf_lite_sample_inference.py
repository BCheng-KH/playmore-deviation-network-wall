import numpy as np
import torch
from modeling.net import SemiADNet
from datasets import mvtecad
import cv2
import os
import argparse
from modeling.layers import build_criterion
#from utils import aucPerformance
#from scipy.ndimage.filters import gaussian_filter
import os, sys
from datasets.base_dataset import BaseADDataset
from PIL import Image
from torchvision import transforms

import onnx

from onnx_tf.backend import prepare

import tensorflow as tf

import scipy.stats as st

np.seterr(divide='ignore',invalid='ignore')

# select the model to load
model_name = "leakage"
experiment="experiment_11"






def transform(img_size):
    composed_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return composed_transforms
def load_image(path):
    if 'npy' in path[-3:]:
        img = np.load(path).astype(np.uint8)
        img = img[:, :, :3]
        return Image.fromarray(img)
    return Image.open(path).convert('RGB')

transformer = transform(448)

# select the image to inference
image = load_image(f"./{experiment}/test2/Crack/IMG_0361.JPG")
sample_input = transformer(image).view(1, 3, 448, 448)

# the tflite model is loaded. adjust the load path as needed
tflite_model_path = f"./{experiment}/{model_name}_tflite.tflite"

interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_data = np.array(sample_input)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
print("sample output: ")
# the anomoly score is output
print(output_data)
# the anomoly score is converted to a probability and output
print(f'{(1-(2*(1-st.norm.cdf(output_data[0][0]))))*100}%')