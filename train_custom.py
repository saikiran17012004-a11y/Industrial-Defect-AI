from ultralytics import YOLO

# 1. Load the classification model 'nano' 
model = YOLO("yolov8n-cls.pt")

# 2. Train it on your images
#'data' points to the folder we just created
model.train(data='train_data', epochs=10, imgsz=224)

print("Finished! Your custom is in: 'runs/classify/train/weights/best.pt'python train_custom.py")