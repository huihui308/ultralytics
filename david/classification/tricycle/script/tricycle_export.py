#!/usr/bin/python3
# Ultralytics YOLO 🚀, AGPL-3.0 license
# https://docs.ultralytics.com/tasks/classify/
"""
$ python3 tricycle_export.py --dynamic --simplify --weights=runs/classify/tricycle_300_128_20241122/weights/last.pt
"""
import torch
from ultralytics import YOLO
import os, sys, argparse, warnings


def suppress_warnings():
    warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)


def parse_args():
    parser = argparse.ArgumentParser(description='DeepStream YOLOv8 conversion')
    parser.add_argument('-w', '--weights', required=True, help='Input weights (.pt) file path (required)')
    parser.add_argument('-s', '--size', nargs='+', type=int, default=[640], help='Inference size [H,W] (default [640])')
    parser.add_argument('--opset', type=int, default=16, help='ONNX opset version')
    parser.add_argument('--simplify', action='store_true', help='ONNX simplify model')
    parser.add_argument('--dynamic', action='store_true', help='Dynamic batch-size')
    parser.add_argument('--batch', type=int, default=1, help='Static batch-size')
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise SystemExit('Invalid weights file')
    if args.dynamic and args.batch > 1:
        raise SystemExit('Cannot set dynamic batch-size and static batch-size at same time')
    return args


def main(args):
    suppress_warnings()
    print('\nStarting: %s' % args.weights)
    # Load a model
    model = YOLO(args.weights)  # load a custom trained
    # Export the model
    model.export(format='onnx')
    if args.dynamic:
        model.export(
            format="onnx",      # 导出格式为 ONNX
            imgsz=(128, 128),   # 设置输入图像的尺寸
            keras=False,        # 不导出为 Keras 格式
            optimize=False,     # 不进行优化 False, 移动设备优化的参数，用于在导出为TorchScript 格式时进行模型优化
            half=False,         # 不启用 FP16 量化
            int8=False,         # 不启用 INT8 量化
            dynamic=True,       # 不启用动态输入尺寸
            simplify=args.simplify,     # 简化 ONNX 模型
            opset=args.opset,   # 使用最新的 opset 版本
            workspace=4.0,      # 为 TensorRT 优化设置最大工作区大小（GiB）
            nms=False,          # 不添加 NMS（非极大值抑制）
            # batch=1,            # 指定批处理大小
            device="cpu"        # 指定导出设备为CPU或GPU，对应参数为"cpu" , "0"
        )
    else:
        model.export(
            format="onnx",      # 导出格式为 ONNX
            imgsz=(128, 128),   # 设置输入图像的尺寸
            keras=False,        # 不导出为 Keras 格式
            optimize=False,     # 不进行优化 False, 移动设备优化的参数，用于在导出为TorchScript 格式时进行模型优化
            half=False,         # 不启用 FP16 量化
            int8=False,         # 不启用 INT8 量化
            # dynamic=True,       # 不启用动态输入尺寸
            simplify=args.simplify,     # 简化 ONNX 模型
            opset=args.opset,   # 使用最新的 opset 版本
            workspace=4.0,      # 为 TensorRT 优化设置最大工作区大小（GiB）
            nms=False,          # 不添加 NMS（非极大值抑制）
            batch=args.batch,   # 指定批处理大小
            device="cpu"        # 指定导出设备为CPU或GPU，对应参数为"cpu" , "0"
        )


if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))