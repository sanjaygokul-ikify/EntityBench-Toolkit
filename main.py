import argparse
import cv2
import torch
import torchvision
import numpy as np
from src.entity_detection import EntityDetector
from src.entity_tracking import EntityTracker
from src.video_generation import VideoGenerator

def main():
    parser = argparse.ArgumentParser(description='EntityBench-Toolkit')
    parser.add_argument('--demo', action='store_true', help='Run the demo')
    args = parser.parse_args()
    if args.demo:
        # Run the demo
        entity_detector = EntityDetector()
        entity_tracker = EntityTracker()
        video_generator = VideoGenerator()
        # Load video input safely
    video_input = cv2.VideoCapture('input.mp4')
    
    # Video properties nikaalna output file setup ke liye
    width = int(video_input.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_input.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_input.get(cv2.CAP_PROP_FPS)) if video_input.get(cv2.CAP_PROP_FPS) > 0 else 20
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
    
    success_flag = False

    print("Processing frames one-by-one to save RAM...")
    frame_count = 0
    while video_input.isOpened():
        ret, frame = video_input.read()
        if not ret:
            break
        frame_count += 1
        print(f"Working on Frame: {frame_count} | Status: Processing...", flush=True)
        with torch.no_grad(): # RAM aur computational speed badhaane ke liye
            entities = entity_detector.detect(frame)
            
        tracked_entities = entity_tracker.track(entities)
        
        # 3. Output frame generation aur video saving
        generated_frame = video_generator.generate(tracked_entities)
        final_output = cv2.resize(generated_frame, (width, height))
        video_writer.write(final_output)
        success_flag = True

    # Resources ko clean up karna
    video_input.release()
    video_writer.release()
    
    if success_flag:
        print("SUCCESS: Video output.mp4 perfectly save ho gayi hai!")
    else:
        print("ERROR: Frame processing execution incomplete.")
if __name__ == '__main__':
    main()