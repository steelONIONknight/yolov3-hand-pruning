from matplotlib.pyplot import get
import torch
from torch.nn.modules import activation
import torchvision
from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *


def print_res(module, input, output):
    print(input[0].shape)
    print(input)
    print(output[0].shape)
    print(output)

img_size = 320

model = Darknet("cfg/prune_0.7_yolov3-hand.cfg", img_size)

weights = "weights/yolov3_hand_normal_pruning_0.7percent.weights"

_ = load_darknet_weights(model, weights)

model.cuda().eval()



# for name, m in model.named_modules():
#     print(name)
#     if name == "module_list.105.Conv2d":
#         print(name)
#         if isinstance(m, torch.nn.Conv2d):
#             m.register_forward_hook(print_res)
        




dataset = LoadImages("data/samples/woman.jpg", img_size=img_size)

# for path, img, im0s, vid_cap in dataset:
#     img = torch.from_numpy(img).cuda()

#     if img.ndimension() == 3:
#         img = img.unsqueeze(0)

#     pred = model(img)[0]
#     print(pred)


for path, img, im0s, vid_cap in dataset:
    img = torch.from_numpy(img).cuda()

    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    pred = model(img)[0]
    print(pred)
    

    
