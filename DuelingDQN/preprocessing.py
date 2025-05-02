import numpy as np
import cv2
from collections import deque


class AtariFrameProcessor:
 
    def __init__(self, frame_history=4, frame_dimensions=(84, 84)):
       
        self.frame_history = frame_history
        self.frame_dimensions = frame_dimensions
        self.frame_buffer = deque(maxlen=frame_history)
        
    def initialize(self, initial_observation):
       
        
        self.frame_buffer.clear()
        
        processed_frame = self._transform_frame(initial_observation)
        for _ in range(self.frame_history):
            self.frame_buffer.append(processed_frame)
            
        return self._stack_frames()
    
    def update(self, new_observation):
        
        processed_frame = self._transform_frame(new_observation)
        self.frame_buffer.append(processed_frame)
        return self._stack_frames()
    
    def _transform_frame(self, raw_frame):
        
       
        grayscale = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2GRAY)
        
     
        resized = cv2.resize(grayscale, self.frame_dimensions, 
                             interpolation=cv2.INTER_AREA)
        
       
        normalized = resized / 255.0
        
        return normalized
    
    def _stack_frames(self):
        return np.stack(self.frame_buffer, axis=0)