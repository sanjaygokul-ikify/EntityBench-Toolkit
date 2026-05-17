import cv2
import torch
import torchvision

class EntityDetector:
    def __init__(self):
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
    def detect(self, frames):
        # Detect entities in each frame
        entities = []
        for frame in frames:
            # Preprocess frame
            frame = cv2.resize(frame, (640, 480))
            # Detect entities
            outputs = self.model(frame)
            entities.append(outputs)
        return entities