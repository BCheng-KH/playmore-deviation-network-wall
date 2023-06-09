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


np.seterr(divide='ignore',invalid='ignore')

# choose which model to convert. these can be found in for example: ./{experiment}/crack.pkl
model_name = "leakage"
experiment="experiment_11"


class model_args:
    def __init__(self, ramdn_seed = 42, no_cuda = False, weight_name = f'{model_name}.pkl', experiment_dir = './experiment', n_scales = 2, backbone = 'resnet18', topk = 0.1, img_size = 448):
        self.ramdn_seed = ramdn_seed
        self.no_cuda = no_cuda
        self.weight_name = weight_name
        self.experiment_dir = experiment_dir
        self.n_scales = n_scales
        self.backbone = backbone
        self.topk = topk
        self.img_size = img_size
        self.cuda = None



# change the experiment direcotry in the line below
args = model_args(ramdn_seed = 42, no_cuda = True, weight_name = f'{model_name}.pkl', experiment_dir = f'./{experiment}', n_scales = 2, backbone = 'resnet18', topk = 0.1, img_size=448)

# this loads in the model in pytorch
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.ramdn_seed)
model = SemiADNet(args)
model.load_state_dict(torch.load(os.path.join(args.experiment_dir, args.weight_name), map_location=torch.device('cpu')))
#model = model.cuda()
model.eval()


# this creates the sample input.
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
# change the sample input image below
image = load_image(f"./{experiment}/test/crack/000.png")
sample_input = transformer(image).view(1, 3, 448, 448)
output = model(sample_input)
print("first sample output: ")
print(output)

# this is where the onnx file will be created
onnx_model_path = f"./{experiment}/{model_name}.onnx"
torch.onnx.export(
    model,                  # PyTorch Model
    sample_input,                    # Input tensor
    onnx_model_path,        # Output file (eg. 'output_model.onnx')
    opset_version=12,       # Operator support version
    input_names=['input'],   # Input tensor name (arbitary)
    output_names=['output'] # Output tensor name (arbitary)
)

# this checks the model is generated properly
onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)
onnx.helper.printable_graph(onnx_model.graph)

# this converts the onnx model to a tensorflow model. change where you want the tensorflow model to go below
tf_rep = prepare(onnx_model)
tf_model_path = f"./{experiment}/{model_name}_tf"
tf_rep.export_graph(tf_model_path)

# this converts the tensorflow model to a tflite model
converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
tflite_model = converter.convert()

# this is where the tflite model is stored, change that below
tflite_model_path = f"./{experiment}/{model_name}_tflite.tflite"
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

# this tests the tflite model
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_data = np.array(sample_input)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
print("second sample output: ")
print(output_data)

# finally, your tflite model can be found for example: ./{experiment}/crack_tflite.tflite