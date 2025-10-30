# Import All the Required Libraries
import torch
import torchvision.transforms as transforms
import cv2
# from torchvision import models
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np

class CourtLineDetector:

    def __init__(self, model_path):
        # Download pretrained Resnet50 model.
        # self.model = models.resnet50(pretrained = True)
        # self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model = resnet50(weights=None) # Not needed, as we are replacing the weights.
        # Replace the last layer with a smaller one.
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14 * 2)
        # Load our fine-tuned saved weights into the model.
        self.model.load_state_dict(torch.load(model_path, map_location = "cpu"))

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image_rgb).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(image_tensor)
        keypoints = outputs.squeeze().cpu().numpy()
        original_height, original_width = image.shape[:2]
        keypoints[::2] *= original_width / 224.0
        keypoints[1::2] *= original_height / 224.0
        return keypoints
    
    # NOTE: The camera is fixed, so we can just use one frame to plot the keypoints for the entire video.

    # Plot / Draw Keypoints on the image
    def draw_keypoints (self, image, keypoints):
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i + 1])
            cv2.putText(image, str(i//2), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        return image

    # Plot / Draw keypoints on the video
    def draw_keypoints_on_video (self, video_frames, keypoints):
        output_video_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame)
        return output_video_frames
