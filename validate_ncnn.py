import cv2
from models import *
from utils.utils import *
from utils.datasets import *
from onnx import helper
import ncnn


yolov3 = ncnn.Net()
yolov3.opt.use_vulkan_compute = False
yolov3.load_param("weights/export_regular_prune0.8-sim-opt-fp16.param")
yolov3.load_model("weights/export_regular_prune0.8-sim-opt-fp16.bin")
# yolov3.load_param("weights/export-sim-opt-fp16.param")
# yolov3.load_model("weights/export-sim-opt-fp16.bin")
dataset = LoadImages("data/samples/woman.jpg", img_size=320)

classes = ["hand"]
save_img = True
s = ""

for path, img, im0s, vid_cap in dataset:
    t = time.time()

    # Get detections
    in_mat = ncnn.Mat(img)
    
    img = torch.from_numpy(img)
    img = np.expand_dims(img, axis=0)

    ex = yolov3.create_extractor()
    ex.set_num_threads(4)
    ex.input("input.1", in_mat)

    ret, pred = ex.extract("823")
    pred = np.array(pred)
    pred = torch.from_numpy(pred)
    pred = pred.unsqueeze(0)
    # print(pred.shape)

    pred = non_max_suppression(pred, 0.3, 0.5)
    for i, det in enumerate(pred):  # detections per image
        p, s, im0 = path, '', im0s

        save_path = "output/output_ncnn.jpg"
        s += '%gx%g ' % img.shape[2:]  # print string
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, classes[int(c)])  # add to string

            # Write results
            for *xyxy, conf, _, cls in det:
                if save_img:  # Add bbox to image
                    label = '%s %.2f' % (classes[int(cls)], conf)
                    #plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
                    plot_one_box(xyxy, im0, label=None, color=1)

        print('%sDone. (%.3fs)' % (s, time.time() - t))

        # Save results (image with detections)
        if save_img:
            if dataset.mode == 'images':
                cv2.imwrite(save_path, im0)

