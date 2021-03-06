# YOLOv3-hand-pruning

本项目以[coldlarry](https://github.com/coldlarry/YOLOv3-complete-pruning)的项目为基础，向他们表示感谢！

具体配置参照[coldlarry](https://github.com/coldlarry/YOLOv3-complete-pruning)的配置。

完成对于yolov3-hand模型的剪枝，部署（使用ncnn推理框架）。

## 结果

x64平台，ncnn部署，推理时间为均值，推理循环256次，16线程，CPU为Intel(R) Xeon(R) CPU E5-2678 v3 @ 2.50GHz。对应benchmark.py文件。


| model                        | 参数量 | 压缩率 | 计算复杂度 | 推理时间 | 加速比 | mAP   |
| ---------------------------- | ------ | ------ | ---------- | -------- | ------ | ----- |
| yolov3-hand                  | 61.52M | 0%     | 13.57 GMac | 102.03ms | 0%     | 0.819 |
| yolov3-hand-prune0.7         | 13.61M | 78%    | 6.31 GMac  | 70.98ms  | 30.4%  | 0.79  |
| yolov3-hand-regular_prune0.8 | 12.9M  | 79%    | 6.08 GMac  | 52.41ms  | 48.6%  | 0.788 |

## 注意

onnx转ncnn注意维度，ncnn目前（2021/7/9）仅支持三个维度，超过三个维度ncnn不会得到正确结果，需要手动调整部分算子的配置。
