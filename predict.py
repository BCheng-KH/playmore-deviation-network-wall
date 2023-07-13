import numpy as np
import torch
from modeling.net import SemiADNet
from datasets import mvtecad
import cv2
import os
import argparse
from modeling.layers import build_criterion
from utils import aucPerformance
from scipy.ndimage.filters import gaussian_filter
import os, sys
from datasets.base_dataset import BaseADDataset
from PIL import Image
from torchvision import transforms


np.seterr(divide='ignore',invalid='ignore')



def convert_to_grayscale(im_as_arr):
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    if im_max > 0:
        grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im


def show_cam_on_image_only(img, mask, out_dir, name):

    img1 = img.copy()
    img[:, :, 0] = img1[:, :, 2]
    img[:, :, 1] = img1[:, :, 1]
    img[:, :, 2] = img1[:, :, 0]

    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)

    path_cam_img = os.path.join(out_dir, name + ".jpg")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cv2.imwrite(path_cam_img, np.uint8(255 * cam))
def return_cam_on_image_only(img, mask):

    img1 = img.copy()
    img[:, :, 0] = img1[:, :, 2]
    img[:, :, 1] = img1[:, :, 1]
    img[:, :, 2] = img1[:, :, 0]

    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)

    return np.uint8(255 * cam)

def load_image(path):
    if 'npy' in path[-3:]:
        img = np.load(path).astype(np.uint8)
        img = img[:, :, :3]
        return Image.fromarray(img)
    return Image.open(path).convert('RGB')
def transform(img_size):
    composed_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return composed_transforms
def predict(model, image, img_size):
    model.zero_grad()
    transformer = transform(img_size)
    inputs = transformer(image).view(1, 3, img_size, img_size).cuda()
    inputs.requires_grad = True
    output = model(inputs)
    outlier_score = output.data.cpu().numpy()[0][0]
    output.backward()

    grad = inputs.grad
    grad_temp = convert_to_grayscale(grad.cpu().numpy().squeeze(0))
    grad_temp = grad_temp.squeeze(0)
    grad_temp = gaussian_filter(grad_temp, sigma=4)
    return outlier_score, grad_temp
def predict_score(model, image, img_size):
    model.zero_grad()
    transformer = transform(img_size)
    inputs = transformer(image).view(1, 3, img_size, img_size).cuda()
    inputs.requires_grad = True
    output = model(inputs)
    outlier_score = output.data.cpu().numpy()[0][0]

    return outlier_score

def predict_with_args(args):
    model = SemiADNet(args)
    model.load_state_dict(torch.load(os.path.join(args.experiment_dir, args.weight_name)))
    model = model.cuda()
    model.eval()


    img_dir = args.image_dir
    image = load_image(img_dir)
    outlier_score, grad_temp = predict(model, image, args.img_size)
    raw = np.float32(cv2.resize(np.array(image), (args.img_size, args.img_size))) / 255
    show_cam_on_image_only(raw, grad_temp, args.output_dir, args.output_name)
    return outlier_score
    
class model_args:
    def __init__(self, ramdn_seed = 42, no_cuda = False, weight_name = 'model.pkl', experiment_dir = './experiment', n_scales = 2, backbone = 'resnet18', topk = 0.1, img_size = 448):
        self.ramdn_seed = ramdn_seed
        self.no_cuda = no_cuda
        self.weight_name = weight_name
        self.experiment_dir = experiment_dir
        self.n_scales = n_scales
        self.backbone = backbone
        self.topk = topk
        self.img_size = img_size
        self.cuda = None

def predict_from_args(image_dir, output_dir, output_name, img_size = 448, ramdn_seed = 42, no_cuda = False, weight_name = 'model.pkl', experiment_dir = './experiment', n_scales = 2, backbone = 'resnet18', topk = 0.1):
    args = model_args(ramdn_seed = ramdn_seed, no_cuda = no_cuda, weight_name = weight_name, experiment_dir = experiment_dir, n_scales = n_scales, backbone = backbone, topk = topk, img_size=img_size)
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.ramdn_seed)
    model = SemiADNet(args)
    model.load_state_dict(torch.load(os.path.join(args.experiment_dir, args.weight_name)))
    model = model.cuda()
    model.eval()
    image = load_image(image_dir)
    outlier_score, grad_temp = predict(model, image, img_size)
    raw = np.float32(cv2.resize(np.array(image), (img_size, img_size))) / 255
    show_cam_on_image_only(raw, grad_temp, output_dir, output_name)
    return outlier_score

def generate_args(img_size = 448, ramdn_seed = 42, no_cuda = False, weight_name = 'model.pkl', experiment_dir = './experiment', n_scales = 2, backbone = 'resnet18', topk = 0.1):
    args = model_args(ramdn_seed = ramdn_seed, no_cuda = no_cuda, weight_name = weight_name, experiment_dir = experiment_dir, n_scales = n_scales, backbone = backbone, topk = topk, img_size=img_size)
    return args
def load_model(args):
    model = SemiADNet(args)
    model.load_state_dict(torch.load(os.path.join(args.experiment_dir, args.weight_name)))
    model = model.cuda()
    model.eval()
    return model
def predict_with_model(model, image_dir, output_dir, output_name, img_size = 448):
    image = load_image(image_dir)
    outlier_score, grad_temp = predict(model, image, img_size)
    raw = np.float32(cv2.resize(np.array(image), (img_size, img_size))) / 255
    return_cam_on_image_only(raw, grad_temp)
    return outlier_score
def predict_score_with_model(model, image_dir, img_size = 448):
    image = load_image(image_dir)
    outlier_score = predict_score(model, image, img_size)
    return outlier_score
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ramdn_seed", type=int, default=42, help="the random seed number")
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--weight_name', type=str, default='model.pkl', help="the name of model weight")
    parser.add_argument('--image_dir', type=str, default='./data/mvtec_anomaly_detection', help="dataset root")
    parser.add_argument('--experiment_dir', type=str, default='./experiment', help="experiment dir root")
    parser.add_argument('--output_dir', type=str, default='carpet', help="the subclass of the datasets")
    parser.add_argument('--output_name', type=str, default='carpet', help="the subclass of the datasets")
    parser.add_argument('--img_size', type=int, default=448, help="the image size of input")
    parser.add_argument("--n_scales", type=int, default=2, help="number of scales at which features are extracted")
    parser.add_argument('--backbone', type=str, default='resnet18', help="the backbone network")
    parser.add_argument("--topk", type=float, default=0.1, help="the k percentage of instances in the topk module")
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.ramdn_seed)
    outlier_score = predict_with_args(args=args)
    print("Anomaly Score: %.4f" % outlier_score)
