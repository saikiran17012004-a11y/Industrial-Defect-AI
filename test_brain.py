from ultralytics import YOLO
model = YOLO("yolov8n.pt")
results = model.predict("https://ultralytics.com/images/bus.jpg", save=True)
print("Sucess! check the 'runs/detect/predict' folder to see what the AI saw.")