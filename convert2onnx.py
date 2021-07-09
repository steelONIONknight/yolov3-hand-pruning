import cv2
from models import *
import onnx
import onnxruntime as ort
from utils.utils import *
from utils.datasets import *
from onnx import helper

# image = cv2.imread("data/samples/woman.jpg")
# image = cv2.resize(image, (320, 192))
# image = np.array(image, dtype=np.float32)
# image = np.expand_dims(image, axis=0)
# image.resize(1, 3, 320, 192)


onnx_model = onnx.load_model("weights/export_prune0.7-sim.onnx")
sess = ort.InferenceSession(onnx_model.SerializeToString())
sess.set_providers(['CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name
pred = sess.get_outputs()[0].name


dataset = LoadImages("data/samples/woman.jpg", img_size=320)

classes = ["hand"]
save_img = True
s = ""

for path, img, im0s, vid_cap in dataset:
    t = time.time()

    # Get detections
    img = torch.from_numpy(img)
    img = np.expand_dims(img, axis=0)
    
    output = sess.run([pred], {input_name: img})
    
    pred = torch.from_numpy(output[0])
    pred = pred.unsqueeze(dim=0)
    

    pred = non_max_suppression(pred, 0.3, 0.5)
    for i, det in enumerate(pred):  # detections per image
        p, s, im0 = path, '', im0s

        save_path = "output/output_onnx.jpg"
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

