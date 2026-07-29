import cv2
import torch
import numpy as np

class VideoGenerator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 3, kernel_size=1) # Channels back to 3 (RGB)
        )
        self.model.to(self.device)
        self.model.eval()

    def generate(self, tracked_entities):
        if not isinstance(tracked_entities, (list, tuple)):
            tracked_entities = [tracked_entities]

        for entity in tracked_entities:
            if isinstance(entity, list) and len(entity) > 0:
                entity = entity[0]
            if isinstance(entity, dict) and 'out' in entity:
                entity = entity['out']

            if hasattr(entity, 'shape'):
                entity = entity.to(self.device)
                if entity.shape[1] != 3:
                    entity = entity[:, :3, :, :]

                # Model inference safely without gradients
                with torch.no_grad():
                    output_tensor = self.model(entity)
                
                # [THE ULTIMATE FIX]: Convert 4D PyTorch Tensor [1, 3, H, W] to 3D OpenCV NumPy Array [H, W, 3]
                output_tensor = output_tensor.squeeze(0)  # Remove batch dimension -> [3, H, W]
                output_np = output_tensor.permute(1, 2, 0).cpu().numpy()  # Channels last -> [H, W, 3]
                
                # Scale back to 0-255 range and convert to uint8 format for OpenCV
                final_frame = np.clip(output_np * 255.0, 0, 255).astype(np.uint8)
                
                # Return direct single frame for main.py loop consumption
                return final_frame
                
        # Fallback agar kuch process na ho paye
        return np.zeros((480, 640, 3), dtype=np.uint8)