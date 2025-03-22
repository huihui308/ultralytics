
# pip install grad-cam -i https://pypi.tuna.tsinghua.edu.cn/simple
import torch    
import cv2
import numpy as np
from ultralytics import YOLO
import torchvision.transforms as transforms
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from PIL import Image

COLORS = np.random.uniform(0, 255, size=(80, 3))


# 解析YOLOv5的检测结果
def parse_detections(results):
    print("-------------------------")
    print(len(results))
    # Iterate over each result (for batch processing, there may be multiple results)
    for result in results:
        # Get bounding boxes (xyxy format), confidence scores, and class IDs
        infer_boxes = result.boxes.xyxy  # Box coordinates in [x1, y1, x2, y2] format
        confidences = result.boxes.conf  # Confidence scores
        class_ids = result.boxes.cls  # Class IDs

    # Convert tensors to numpy arrays for easier handling
    infer_boxes = infer_boxes.cpu().numpy()
    confidences = confidences.cpu().numpy()
    class_ids = class_ids.cpu().numpy()

    print(len(infer_boxes))
    print(len(confidences))
    print(len(class_ids))

    print("-------------------------")

    boxes, colors, names = [], [], []

    for i, box in enumerate(infer_boxes):
        if confidences[i] < 0.2:
            continue
        xmin, ymin, xmax, ymax = map(int, box)  # Extract coordinates
        # xmin = int(detections["xmin"][i])
        # ymin = int(detections["ymin"][i])
        # xmax = int(detections["xmax"][i])
        # ymax = int(detections["ymax"][i])
        name = class_ids[i]
        category = int(class_ids[i])
        color = COLORS[category]

        boxes.append((xmin, ymin, xmax, ymax))
        colors.append(color)
        names.append( str(name) )
    
    print("-------------------------")
    # detections = results.pandas().xyxy[0]
    # detections = result.boxes  # Access the boxes attribute

    # # detections = detections.to_dict()
    # boxes, colors, names = [], [], []

    # for i in range(len(detections["xmin"])):
    #     confidence = detections["confidence"][i]
    #     if confidence < 0.2:
    #         continue
    #     xmin = int(detections["xmin"][i])
    #     ymin = int(detections["ymin"][i])
    #     xmax = int(detections["xmax"][i])
    #     ymax = int(detections["ymax"][i])
    #     name = detections["name"][i]
    #     category = int(detections["class"][i])
    #     color = COLORS[category]

    #     boxes.append((xmin, ymin, xmax, ymax))
    #     colors.append(color)
    #     names.append(name)
    return boxes, colors, names


# 将检测结果画在图片上
def draw_detections(boxes, colors, names, img):
    for box, color, name in zip(boxes, colors, names):
        xmin, ymin, xmax, ymax = box
        # print(xmin, ymin, xmax, ymax, type(name))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(img, name, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, lineType=cv2.LINE_AA)
    return img


# 对检测框内部的特征进行归一化，将检测框外部的特征设置为0
def renormalize_cam_in_bounding_boxes(boxes, colors, names, image_float_np, grayscale_cam):
    """Normalize the CAM to be in the range [0, 1] 
    inside every bounding boxes, and zero outside of the bounding boxes. """
    renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
    for x1, y1, x2, y2 in boxes:
        renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())    
    renormalized_cam = scale_cam_image(renormalized_cam)
    eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
    image_with_bounding_boxes = draw_detections(boxes, colors, names, eigencam_image_renormalized)
    return image_with_bounding_boxes


# 1.读图片，做预处理
image_url = "images/3.jpg"
img = cv2.imread(image_url)[:,:,[2,1,0]]
img = cv2.resize(img, (640, 640))
rgb_img = img.copy()     # rgb_img作为网络输入端的图片
img = np.float32(img) / 255    # 使用show_cam_on_image进行可视化时使用


# transform = transforms.ToTensor()
# input_tensor = transform(img).unsqueeze(0)    # 在获取特征图时作为输入数据使用
# Step 2: Preprocess the input image
preprocess = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
image_path = 'images/3.jpg'
image = Image.open(image_path).convert('RGB')
input_tensor = preprocess(image).unsqueeze(0)


# 2.模型初始化
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model_path = "runs/detect/yolo12x-4heads-900epoches-visdrone-20250310/weights/best.pt"  # Path to your local .pt file
model = YOLO(model_path)

print("------------------------ 0")
model.eval()
model.cpu()

print("------------------------ 1")
# # Alternatively, iterate through the model's layers
# torch_model = model.model
# for name, module in torch_model.named_modules():
#     print(name, module)

# 3.模型推理、推理结果可视化
results = model([rgb_img])
boxes, colors, names = parse_detections(results)
detections = draw_detections(boxes, colors, names, rgb_img.copy())
cv2.imwrite('./detection.png', detections[:,:,[2,1,0]])

print("------------------------ 2")
# 4.指定要可视化的特征的层，这里指定detect的前一个层
# target_layer = [model.model.model.model[-2]]
# target_layer = dict([*model.model.named_modules()])["model.7"]
target_layer = [model.model.model[-2]]

# Step 1: Define a feature extractor
class YOLOFeatureExtractor(torch.nn.Module):
    def __init__(self, model, target_layer_name):
        super(YOLOFeatureExtractor, self).__init__()
        self.model = model
        self.target_layer_name = target_layer_name

    def forward(self, x):
        for name, module in self.model.model.named_children():
            x = module(x)
            if name == self.target_layer_name:
                return x  # Return the feature map at the target layer
        return x  # Default return (last layer)

# Print top-level layers
# for name, module in model.model.named_children():
#     print(name, module)

print("------------------------ 3")
# Step 2: Initialize the feature extractor
target_layer_name = "model.7"  # Replace with the correct layer name
target_layer = YOLOFeatureExtractor(model, target_layer_name)
# target_layer = model.model.layer4[-1]

target_layer = [model.model.model[-2]]

print("------------------------ 4")
# 5.实例化EigenCAM、得到可视化特征，并显示在原图上
# cam = EigenCAM(model, target_layer, use_cuda=False)
cam = EigenCAM(model=model, target_layers=target_layer)

print("------------------------ 5")
# grayscale_cam = cam(tensor)[0, :, :]
print(type(input_tensor))
grayscale_cam = cam(input_tensor=input_tensor)[0]

print("------------------------ 6")
# cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
cam_image = show_cam_on_image(img, grayscale_cam)
cv2.imwrite('./cam_image.png', cam_image[:,:,[2,1,0]])

print("------------------------ 7")
# 6.对检测框内部的特征进行归一化，将检测框外部的特征设置为0
renormalized_cam_image = renormalize_cam_in_bounding_boxes(boxes, colors, names, img, grayscale_cam)
cv2.imwrite('./renormalized_cam_image.png', renormalized_cam_image[:,:,[2,1,0]])
cv2.imwrite('./all.png', np.hstack((rgb_img, cam_image, renormalized_cam_image))[:,:,[2,1,0]])


# Step 1: Define a feature extractor
class YOLOFeatureExtractor(torch.nn.Module):
    def __init__(self, model, target_layer_name):
        super(YOLOFeatureExtractor, self).__init__()
        self.model = model
        self.target_layer_name = target_layer_name

    def forward(self, x):
        for name, module in self.model.model.named_children():
            x = module(x)
            if name == self.target_layer_name:
                return x  # Return the feature map at the target layer
        return x  # Default return (last layer)

# Step 2: Initialize the feature extractor
target_layer_name = "model.7"  # Replace with the correct layer name
feature_extractor = YOLOFeatureExtractor(model, target_layer_name)

print("1111111111111111")
# Step 3: Initialize EigenCAM
cam = EigenCAM(model=model, target_layers=[model.model.model[-2]])

print("2222222222222222")
# Step 4: Generate Grad-CAM
raw_output = feature_extractor(input_tensor)
grayscale_cam = cam(input_tensor=input_tensor)[0]

print("3333333333333333")
# Step 5: Visualize Grad-CAM
cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
cv2.imwrite('./cam_image.png', cam_image[:, :, [2, 1, 0]])
print("9999999999999999999999")













# import cv2

# from ultralytics import solutions

# cap = cv2.VideoCapture("images/output.mp4")
# assert cap.isOpened(), "错误读取视频文件"
# w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# # 视频写入器
# video_writer = cv2.VideoWriter("heatmap_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# # 初始化热图
# heatmap = solutions.Heatmap(
#     show=True,
#     model="runs/detect/yolo12x-4heads-900epoches-visdrone-20250310/weights/best.pt",
#     colormap=cv2.COLORMAP_PARULA,
# )

# lop_cnt = 0
# while cap.isOpened():
#     print('Deal {} images'.format(lop_cnt))
#     lop_cnt = lop_cnt + 1
#     success, im0 = cap.read()
#     if not success:
#         print("视频帧为空或视频处理已成功完成。")
#         break
#     im0 = heatmap.generate_heatmap(im0)
#     video_writer.write(im0)

# cap.release()
# video_writer.release()
# cv2.destroyAllWindows()



# import cv2

# from ultralytics import solutions

# cap = cv2.VideoCapture("images/output.mp4")
# heatmap = solutions.Heatmap(show=True, model="runs/detect/yolo12x-4heads-900epoches-visdrone-20250310/weights/best.pt", classes=[0, 2])

# lop_cnt = 0
# while cap.isOpened():
#     print('Deal {} images'.format(lop_cnt))
#     lop_cnt = lop_cnt + 1
#     success, im0 = cap.read()
#     if not success:
#         break
#     im0 = heatmap.generate_heatmap(im0)
#     cv2.imshow("Heatmap", im0)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()




# import torch
# import torch.nn as nn
# import cv2
# import numpy as np
# from ultralytics import YOLO


# def generate_heatmap(model, img_tensor, layer_name):
#     """
#     Generate a heatmap for a specific layer in the YOLOv8 model.
#     :param model: The YOLOv8 model (PyTorch model).
#     :param img_tensor: Input image tensor (preprocessed, shape [1, C, H, W]).
#     :param layer_name: Name of the layer to extract feature maps from.
#     :return: Heatmap overlaid on the original image.
#     """
#     # Set the model to evaluation mode
#     model.model.eval()

#     # Register a hook to capture the output of the specified layer
#     activation = {}

#     def hook_fn(module, input, output):
#         activation[layer_name] = output.detach()

#     # Find the target layer by name
#     try:
#         target_layer = dict([*model.model.named_modules()])[layer_name]
#     except KeyError:
#         print(f"Layer '{layer_name}' not found in the model.")
#         print("Available layers:")
#         for name, _ in model.model.named_modules():
#             print(name)
#         raise

#     hook = target_layer.register_forward_hook(hook_fn)

#     # Forward pass through the model
#     with torch.no_grad():
#         _ = model.model(img_tensor)

#     # Remove the hook
#     hook.remove()

#     # Get the feature map from the target layer
#     feature_map = activation[layer_name].squeeze(0)  # Shape: [C, H, W]

#     # Aggregate feature maps across channels (e.g., by taking the mean)
#     heatmap = torch.mean(feature_map, dim=0).cpu().numpy()

#     # Normalize the heatmap to [0, 1]
#     heatmap = np.maximum(heatmap, 0)  # ReLU-like operation
#     heatmap /= np.max(heatmap) + 1e-8  # Avoid division by zero

#     # Convert heatmap to RGB (for visualization)
#     heatmap = cv2.resize(heatmap, (img_tensor.shape[3], img_tensor.shape[2]))
#     heatmap = np.uint8(255 * heatmap)  # Scale to [0, 255]
#     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

#     # Convert the input image tensor back to a numpy array
#     img = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
#     img = np.uint8(255 * img)  # Scale to [0, 255]

#     # Overlay the heatmap on the original image
#     superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

#     return superimposed_img


# # Example Usage
# if __name__ == "__main__":
#     # Step 1: Load the local YOLOv8 model
#     # model_path = "runs/detect/yolo12x-4heads-900epoches-visdrone-20250310/weights/best.pt"  # Path to your local .pt file
#     model_path = "yolov8n.pt"  # Path to your local .pt file
#     model = YOLO(model_path)

#     # Inspect the last layer of the model
#     last_layer = model.model.model[-1]
#     print(last_layer)

#     # Step 2: Load and preprocess an image
#     img_path = "/home/david/dataset/drone/VisDroneOrigin/VisDrone2019-DET-val/images/0000296_00601_d_0000038.jpg"
#     img = cv2.imread(img_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
#     img_tensor = torch.tensor(img / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # Preprocess

#     # Step 3: Specify the layer name to extract feature maps
#     layer_name = "model.9"  # Replace with the actual layer name

#     # Step 4: Generate the heatmap
#     heatmap_img = generate_heatmap(model, img_tensor, layer_name)

#     # Step 5: Save or display the heatmap
#     cv2.imwrite("heatmap.jpg", heatmap_img)
#     cv2.imshow("Heatmap", heatmap_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()