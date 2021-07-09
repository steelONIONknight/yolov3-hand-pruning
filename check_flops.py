from matplotlib.pyplot import get
import torch
from torch.nn.modules import activation
import torchvision
from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
from ptflops import get_model_complexity_info

img_size = 320

with torch.cuda.device(0):

	model = Darknet("cfg/prune_0.8_yolov3-hand.cfg", img_size)

	weights = "weights/yolov3_hand_regular_pruning_0.8percent.weights"

	_ = load_darknet_weights(model, weights)

	macs, params = get_model_complexity_info(model, (3, 320, 224), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
	print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
	print('{:<30}  {:<8}'.format('Number of parameters: ', params))
