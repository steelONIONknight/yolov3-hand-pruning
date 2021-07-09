import sys
import ncnn
import time
import cv2
from models import *
from utils.utils import *
from utils.datasets import *

g_warmup_loop_count = 8
g_loop_count = 4
g_enable_cooling_down = True

g_vkdev = None
g_blob_vkallocator = None
g_staging_vkallocator = None

g_blob_pool_allocator = ncnn.UnlockedPoolAllocator()
g_workspace_pool_allocator = ncnn.PoolAllocator()

def benchmark(comment, _in, opt):
    _in.fill(0.1)

    g_blob_pool_allocator.clear()
    g_workspace_pool_allocator.clear()

    if opt.use_vulkan_compute:
        g_blob_vkallocator.clear()
        g_staging_vkallocator.clear()

    net = ncnn.Net()
    net.opt = opt
    print(net.opt.use_packing_layout)
    if net.opt.use_vulkan_compute:
        net.set_vulkan_device(g_vkdev)

    net.load_param(comment + ".param")

    net.load_model(comment + ".bin")

    input_names = net.input_names()
    output_names = net.output_names()
    
    if g_enable_cooling_down:
        time.sleep(10)
    
    # warm up
    for i in range(g_warmup_loop_count):
        # test with statement
        with net.create_extractor() as ex:
            ex.input(input_names[0], _in)
            ex.extract(output_names[0])
            

    time_min = sys.float_info.max
    time_max = -sys.float_info.max
    time_avg = 0.0

    for i in range(g_loop_count):
        start = time.time()

        # test net keep alive until ex freed
        ex = net.create_extractor()
        ex.input(input_names[0], _in)
        ex.extract(output_names[0])

        end = time.time()

        timespan = end - start

        time_min = timespan if timespan < time_min else time_min
        time_max = timespan if timespan > time_max else time_max
        time_avg += timespan

    time_avg /= g_loop_count

    print(
        "%20s  min = %7.2f  max = %7.2f  avg = %7.2f"
        % (comment, time_min * 1000, time_max * 1000, time_avg * 1000)
    )


if __name__ == "__main__":
    loop_count = 16
    # num_threads = ncnn.get_cpu_count()
    num_threads = 1
    powersave = 0
    gpu_device = -1
    cooling_down = 0
    use_vulkan_compute = gpu_device != -1

    g_enable_cooling_down = cooling_down != 0

    g_loop_count = loop_count

    g_blob_pool_allocator.set_size_compare_ratio(0.0)
    g_workspace_pool_allocator.set_size_compare_ratio(0.5)

    if use_vulkan_compute:
        g_warmup_loop_count = 10

        g_vkdev = ncnn.get_gpu_device(gpu_device)

        g_blob_vkallocator = ncnn.VkBlobAllocator(g_vkdev)
        g_staging_vkallocator = ncnn.VkStagingAllocator(g_vkdev)

    opt = ncnn.Option()
    opt.lightmode = True
    opt.num_threads = num_threads
    opt.blob_allocator = g_blob_pool_allocator
    opt.workspace_allocator = g_workspace_pool_allocator
    if use_vulkan_compute:
        opt.blob_vkallocator = g_blob_vkallocator
        opt.workspace_vkallocator = g_blob_vkallocator
        opt.staging_vkallocator = g_staging_vkallocator

    opt.use_winograd_convolution = True
    opt.use_sgemm_convolution = True
    opt.use_int8_inference = False
    opt.use_vulkan_compute = use_vulkan_compute
    # opt.use_fp16_packed = True
    # opt.use_fp16_storage = True
    # opt.use_fp16_arithmetic = True
    # opt.use_int8_storage = True
    # opt.use_int8_arithmetic = True
    # opt.use_packing_layout = True
    # opt.use_shader_pack8 = False
    # opt.use_image_storage = False

    ncnn.set_cpu_powersave(powersave)
    ncnn.set_omp_dynamic(0)
    ncnn.set_omp_num_threads(num_threads)

    print("loop_count =", loop_count)
    print("num_threads =", num_threads)
    print("powersave =", ncnn.get_cpu_powersave())
    print("gpu_device =", gpu_device)
    print("cooling_down =", g_enable_cooling_down)

    dataset = LoadImages("data/samples/woman.jpg", img_size=320)
    for path, img, im0s, vid_cap in dataset:
        in_mat = ncnn.Mat(img)
        benchmark("weights/export_prune0.7-sim-opt-fp16", in_mat, opt)
        benchmark("weights/export-sim-opt-fp16", in_mat, opt)
        benchmark("weights/export_regular_prune0.8-sim-opt-fp16", in_mat, opt)

