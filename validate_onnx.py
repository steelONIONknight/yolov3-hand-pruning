from tool import Tool
import cv2
from models import *
import onnx
import onnxruntime as ort
from utils.utils import *
from utils.datasets import *


# an example 


# t = Tool("weights/export.onnx")
# node_name = t.get_node_by_name("LeakyRelu_1")
# t.add_extra_output(node_name, "LeakyRelu_1_output")
# t.export("weights/export_test.onnx")


def print_node_val(onnx_path, name):
    t = Tool(onnx_path)
    node_name = t.get_node_by_name(name)
    t.add_extra_output(node_name, name + "_output")
    t.export("weights/export_test_" + name + ".onnx")




def test(layer_name):
    onnx_model = onnx.load_model("weights/export_test_" + layer_name + ".onnx")
    sess = ort.InferenceSession(onnx_model.SerializeToString())
    sess.set_providers(['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name


    res = sess.get_outputs()[0].name
    conv = sess.get_outputs()[1].name

    dataset = LoadImages("data/samples/woman.jpg", img_size=320)

    classes = ["hand"]
    save_img = True
    s = ""

    for path, img, im0s, vid_cap in dataset:
        t = time.time()

        # Get detections
        img = torch.from_numpy(img)
        img = np.expand_dims(img, axis=0)
        print(img.shape)

        
        output = sess.run([res, conv], {input_name: img})
        print(output[1].shape)
        print(output[1])

if __name__ == "__main__":
    print_node_val("weights/export.onnx", "Concat_303")
    test("Concat_303")