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
        # Detect entities in a batch safely without breaking tracking
        if not isinstance(frames, (list, tuple)):
            frames = [frames]
            
        self.model.eval()
        
        processed_tensors = []
        for frame in frames:
            if isinstance(frame, torch.Tensor):
                processed_tensors.append(frame.to(self.device))
            else:
                # OpenCV frame preprocessing
                resized_frame = cv2.resize(frame, (640, 480))
                f_normalized = resized_frame.astype(np.float32) / 255.0
                t = torch.from_numpy(f_normalized).permute(2, 0, 1).float()
                processed_tensors.append(t)
                
        # Stack into batch
        input_batch = torch.stack(processed_tensors).to(self.device)
        
        # Super fast forward pass
        with torch.no_grad():
            outputs = self.model(input_batch)['out']
            
        # [THE FIX]: Tracker ko khush rakhne ke liye exact dictionary format return karo
        # Purana code har frame ke liye ek dict={'out': tensor} bhejta tha loop ke andar
        entities = []
        for i in range(outputs.size(0)):
            single_output_4d = outputs[i].unsqueeze(0)
            entities.append({'out': single_output_4d})
            
        return entities