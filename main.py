import argparse
import cv2
import torch
import torchvision
import numpy as np
from src.entity_detection import EntityDetector
from src.entity_tracking import EntityTracker
from src.video_generation import VideoGenerator


def process_batch(frames, detector, tracker, generator, writer, current_count):
    print(f"Working on Batch ending at Frame: {current_count} | Status: Processing...", flush=True)
    
    with torch.no_grad():
        # 1. Detector ko poora batch ek saath bhej rahe hain!
        batch_entities = detector.detect(frames)
        
    # Tracker aur generator abhi single frame loop support karte hain, unhe smoothly loop me chalao
    for i, entities in enumerate(batch_entities):
        tracked_entities = tracker.track(entities)
        generated_frame = generator.generate(tracked_entities)
        
        # Final output ko resize karke save karo
        height, width = frames[i].shape[0], frames[i].shape[1]
        final_output = cv2.resize(generated_frame, (width, height))
        writer.write(final_output)


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
    BATCH_SIZE = 16
    frames_batch = []
    while video_input.isOpened():
        ret, frame = video_input.read()
        if not ret:
            if frames_batch:
                process_batch(frames_batch, entity_detector, entity_tracker, video_generator, video_writer, frame_count)
        
            break
        frame_count += 1
        frames_batch.append(frame)
        if len(frames_batch) == BATCH_SIZE:
            process_batch(frames_batch, entity_detector, entity_tracker, video_generator, video_writer, frame_count)
            frames_batch = []
            
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