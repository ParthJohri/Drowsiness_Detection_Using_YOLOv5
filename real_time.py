import os
import time
import torch
import cv2
from torchvision import transforms
import numpy as np

weights_path = "./best.pt"

model = torch.hub.load('yolov5_', 'custom', path=weights_path, source='local')

def process_frame(frame):
    results = model(frame)

    labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    for i in range(n):
        row = cords[i]
        if row[4] >= 0.1: 
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
            bgr = (0, 255, 0) 
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            cv2.putText(frame, f'{model.names[int(labels[i])]} {row[4]:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
    return frame

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = process_frame(frame)

    cv2.imshow('YOLOv5 Real-time Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end_time = time.time()
inference_time = end_time - start_time
print(f'Inference Time: {inference_time} seconds')

cap.release()
cv2.destroyAllWindows()
