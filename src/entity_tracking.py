import cv2
import torch

class EntityTracker:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.ReLU()
        )
        self.model.to(self.device)
        self.model.eval()
        
    def track(self, self_or_entities, entities=None):
        # Kuch cases mein arguments bypass ho jaate hain, safely resolve karo
        actual_entities = entities if entities is not None else self_or_entities
        if not isinstance(actual_entities, (list, tuple)):
            actual_entities = [actual_entities]

        tracked_entities = []
        for entity in actual_entities:
            # [CRITICAL FIX]: Agar list ke andar list/dict aa jaye
            if isinstance(entity, list) and len(entity) > 0:
                entity = entity[0]
            if isinstance(entity, dict) and 'out' in entity:
                entity = entity['out']

            # Agar abhi bhi list hai toh loop skip karo crash se bachne ke liye
            if hasattr(entity, 'shape'):
                entity = entity.to(self.device)
                if entity.shape[1] == 21:
                    entity = entity[:, :3, :, :]
                with torch.no_grad():
                    features = self.model(entity)
                tracked_entities.append(features)
                
        return tracked_entities