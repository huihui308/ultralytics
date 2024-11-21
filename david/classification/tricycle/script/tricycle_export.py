#!/usr/bin/python3
# Ultralytics YOLO 🚀, AGPL-3.0 license
# https://docs.ultralytics.com/tasks/classify/
from ultralytics import YOLO


if __name__ == '__main__':
    # Load a model
    model = YOLO('runs/classify/tricycle_20241121/weights/best.pt')  # load a custom trained

    # Export the model
    model.export(format='onnx')
    model.export(
        format="onnx",      # 导出格式为 ONNX
        imgsz=(128, 128),   # 设置输入图像的尺寸
        keras=False,        # 不导出为 Keras 格式
        optimize=False,     # 不进行优化 False, 移动设备优化的参数，用于在导出为TorchScript 格式时进行模型优化
        half=False,         # 不启用 FP16 量化
        int8=False,         # 不启用 INT8 量化
        dynamic=True,       # 不启用动态输入尺寸
        # simplify=True,      # 简化 ONNX 模型
        opset=None,         # 使用最新的 opset 版本
        workspace=4.0,      # 为 TensorRT 优化设置最大工作区大小（GiB）
        nms=False,          # 不添加 NMS（非极大值抑制）
        # batch=1,            # 指定批处理大小
        device="cpu"        # 指定导出设备为CPU或GPU，对应参数为"cpu" , "0"
    )