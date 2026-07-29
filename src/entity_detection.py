import cv2
import torch
import torchvision
import numpy as np

class EntityDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"EntityDetector initialized on divice: {self.device}")

        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True).to(self.device)
    def detect(self, frames):
        # Detect entities in each frame
        entities = []
        
        # Agar bahar se ek single frame bina list ke pass ho jaye, toh use list mein wrap karo
        if not isinstance(frames, (list, tuple)):
            frames = [frames]
            
        self.model.eval()

        for frame in frames:
            # 1. Check format: Agar input torch Tensor hai
            if isinstance(frame, torch.Tensor):
                frame_dev = frame.to(self.device)
                with torch.no_grad():
                    outputs = self.model(frame_dev)['out']
                entities.append(outputs)
            else:
                # 2. Agar standard NumPy array (OpenCV frame) hai
                resized_frame = cv2.resize(frame, (640, 480))
                
                # [THE FIX]: Model input ke liye normalized Tensor banana
                f_normalized = resized_frame.astype(np.float32) / 255.0
                t = torch.from_numpy(f_normalized).permute(2, 0, 1).float()
                input_tensor = t.unsqueeze(0)  # Add batch dimension
                
                input_tensor = input_tensor.to(self.device)
                # DeepLabV3 internally torchvision model hai, use output dict milti hai
                with torch.no_grad():
                    outputs = self.model(input_tensor)['out']
                entities.append(outputs)
                
        return entities