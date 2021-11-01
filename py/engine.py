import os 
import cv2 
import random 
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt 
import torch
from torch._C import dtype 
from utils import nms, scale_coords, show, scale_coords2


# TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
TRT_LOGGER = trt.Logger()
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


def GiB(val):
    return val * 1 << 30  # 1 << 30  ->  1GB



def get_engine(onnx_file_path, engine_file_path="", max_batch_size=1):

    def build_engine():
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, builder.create_builder_config() as config,\
                trt.OnnxParser(network, TRT_LOGGER) as parser, trt.Runtime(TRT_LOGGER) as runtime:
            # config.max_workspace_size = 1 << 28  # 256MB
            config.max_workspace_size = GiB(max_batch_size)
            builder.max_batch_size = max_batch_size

            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)

            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                if not parser.parse(model.read()):
                    print('Error: Failed to parse the ONNX file')
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None 

            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            # network.get_input(0).shape = [1, 3, 608, 608]
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating engine")
            with open(engine_file_path, "wb") as f:
                f.write(plan)
                
            return engine 

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size 
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers 
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings 
        bindings.append(int(device_mem))
        # Append to the appropriate list 
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))


    return inputs, outputs, bindings, stream 


def do_inference(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference 
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream 
    stream.synchronize()
    # return only the host outputs 
    return [out.host for out in outputs]


def preprocess(img_path, input_shape=(640, 640)):
    im0 = cv2.imread(img_path)
    img = cv2.resize(cv2.cvtColor(im0, cv2.COLOR_BGR2RGB), input_shape, interpolation=cv2.INTER_LINEAR)
    img = np.array(img, dtype=np.float32) / 255. 
    img = np.expand_dims(img.transpose((2, 0, 1)), axis=0)
    img = np.ascontiguousarray(img)
    print(f'Input shape: {img.shape}')

    return img, im0


def main():

    onnx_file_path = '../yolov5s.onnx'
    engine_file_path = './yolov5s.trt'

    with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = allocate_buffers(engine)
        # Set host input to the image. 
        img, im0 = preprocess('../data/bus.jpg')
        inputs[0].host = img 
        trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        print('trt outputs: ', type(trt_outputs[0]), trt_outputs[0].shape, trt_outputs[0].reshape((-1, 85)).shape)
        print('Complete Inference')

    return trt_outputs[0].reshape((-1, 85)), img, im0


if __name__ == '__main__':
    # engine = get_engine('../yolov5s.onnx', './yolov5s_fromonnx.trt')
    
    with open('../data/class_names.txt', 'r') as f:
        class_names = [x.strip() for x in f.readlines()]

    class_colors = [random.choices(range(256), k=3) for _ in range(len(class_names))]

    pred, img, im0 = main()
    pred = torch.from_numpy(pred).unsqueeze(0).float()
    pred = nms(pred)[0]

    # print result 
    s = ''
    for c in pred[:, -1].unique():
        n = (pred[:, -1] == c).sum() 
        s += f"{n} {class_names[int(c)]}{'s' * (n > 1)}, " 

    print(f'detection results: {s}')

    # save 
    print(f'img: {img.shape}, im0: {im0.shape}')
    pred[:, :4] = scale_coords2(img.shape[2:], pred[:, :4], im0.shape)

    show(im0.copy(), pred, class_names, class_colors)
