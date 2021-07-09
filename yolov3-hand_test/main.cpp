#include <iostream>
#include "net.h"
int main()
{
    ncnn::Net yolov3;
    yolov3.opt.num_threads = 1;
    yolov3.load_param("/home/lifang/YOLOv3-complete-pruning/weights/export_regular_prune0.8-sim-opt-fp16.param");
    yolov3.load_model("/home/lifang/YOLOv3-complete-pruning/weights/export_regular_prune0.8-sim-opt-fp16.bin");
    ncnn::Extractor ex = yolov3.create_extractor();
    ncnn::Mat in(224, 320, 3);
    in.fill(0.1f);
    ncnn::Mat out;
    ncnn::Mat out1;
    ex.input("input.1", in);
    ex.extract("441", out1);
    ex.extract("823", out);
    return 0;
}
