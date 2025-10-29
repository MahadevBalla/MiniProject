"""
Integration example showing how to combine AudioDirector with YOLO camera system.

This example demonstrates:
- Real-time audio processing with camera control
- Speech-triggered camera focus changes
- Feature-based speaker classification
- Smooth transitions between audio and visual cues
"""

import cv2
import numpy as np
import time
import threading
from ultralytics import YOLO
from auto_frame import draw_bounding_box, create_zoomed_view
from audio_director import AudioDirector


class AudioCameraDirector:
    """
    Integrated audio-visual director that combines audio cues with camera control.
    """
    
    def __init__(self, model_path="models/yolov8n.pt", camera_id=0):
        """
        Initialize the integrated audio-visual director.
        
        Args:
            model_path: Path to YOLO model
            camera_id: Camera device ID
        """
        # Camera setup
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # YOLO model
        self.model = YOLO(model_path)
        
        # Camera state
        self.prev_box = None
        self.alpha = 0.2  # Smoothing factor
        self.auto_focus_enabled = True
        
        # Audio state
        self.speech_active = False
        self.last_speech_time = 0
        self.speech_timeout = 2.0  # seconds
        
        # Feature collection for training
        self.feature_buffer = []
        self.label_buffer = []
        self.collecting_training_data = False
        
        # Setup audio director with callbacks
        self.audio_director = AudioDirector(
            samplerate=16000,
            channels=1,
            enable_doa=False,  # Set to True if you have a mic array
            callbacks={
                'on_speech_start': self._on_speech_start,
                'on_speech_end': self._on_speech_end,
                'on_feature': self._on_feature,
                'on_direction': self._on_direction
            }
        )
        
        print("AudioCameraDirector initialized")
    
    def _on_speech_start(self):
        """Handle speech start event."""
        self.speech_active = True
        self.last_speech_time = time.time()
        print("[Integration] Speech detected - enabling auto-focus")
    
    def _on_speech_end(self):
        """Handle speech end event."""
        self.speech_active = False
        print("[Integration] Speech ended")
    
    def _on_feature(self, frame_idx, features):
        """Handle audio feature extraction."""
        # Collect features for training if in training mode
        if self.collecting_training_data:
            self.feature_buffer.append(features)
        
        # Optional: Use features for real-time classification
        if self.audio_director.clf_ready:
            role, confidence = self.audio_director.predict_role(features)
            if role is not None and confidence > 0.7:
                role_name = "lecturer" if role == 1 else "audience"
                print(f"[Integration] Detected {role_name} (confidence: {confidence:.2f})")
    
    def _on_direction(self, angle):
        """Handle direction of arrival estimation."""
        print(f"[Integration] Sound direction: {angle:.1f}°")
        # Could use this to pan camera or adjust focus area
    
    def start_training_mode(self, label):
        """
        Start collecting training data for speaker classification.
        
        Args:
            label: 0 for audience, 1 for lecturer
        """
        self.collecting_training_data = True
        self.feature_buffer = []
        self.label_buffer = []
        print(f"[Integration] Training mode started for label: {label}")
    
    def stop_training_mode(self):
        """Stop collecting training data and train classifier."""
        if not self.collecting_training_data:
            return
        
        self.collecting_training_data = False
        
        if len(self.feature_buffer) > 10:  # Minimum samples for training
            # Convert features to training format
            X = []
            for feats in self.feature_buffer:
                x = np.hstack([
                    feats['mfcc_mean'],
                    feats['mfcc_delta_mean'],
                    feats['lpc_mean']
                ])
                X.append(x)
            
            X = np.array(X)
            y = np.array(self.label_buffer)
            
            # Train classifier
            self.audio_director.fit_classifier(X, y)
            print(f"[Integration] Classifier trained on {len(X)} samples")
        else:
            print("[Integration] Not enough training data collected")
    
    def add_training_sample(self, label):
        """
        Add current features to training data.
        
        Args:
            label: 0 for audience, 1 for lecturer
        """
        if self.collecting_training_data and self.feature_buffer:
            self.label_buffer.append(label)
            print(f"[Integration] Added training sample with label: {label}")
    
    def process_frame(self, frame):
        """
        Process a single camera frame with audio-aware focusing.
        
        Args:
            frame: Input camera frame
            
        Returns:
            Processed frame with bounding boxes
        """
        # Run YOLO detection
        results = self.model(frame, stream=True)
        
        person_boxes = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if self.model.names[cls] == "person":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    person_boxes.append((x1, y1, x2, y2))
        
        if person_boxes:
            # Pick largest person
            person_boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
            x1, y1, x2, y2 = person_boxes[0]
            
            new_box = np.array([x1, y1, x2, y2], dtype=float)
            
            # Apply smoothing
            if self.prev_box is None:
                self.prev_box = new_box
            else:
                # Adjust smoothing based on speech activity
                if self.speech_active:
                    # Faster response during speech
                    smooth_alpha = min(0.5, self.alpha * 2)
                else:
                    # Slower response when no speech
                    smooth_alpha = self.alpha
                
                self.prev_box = smooth_alpha * new_box + (1 - smooth_alpha) * self.prev_box
            
            x1, y1, x2, y2 = map(int, self.prev_box)
            
            # Draw bounding box with different colors based on speech state
            color = (0, 255, 0) if self.speech_active else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Add speech indicator
            if self.speech_active:
                cv2.putText(frame, "SPEECH ACTIVE", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add status information
        status_text = f"Speech: {'ON' if self.speech_active else 'OFF'}"
        cv2.putText(frame, status_text, (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def run(self):
        """Main processing loop."""
        # Start audio processing
        self.audio_director.start()
        
        # Create window
        cv2.namedWindow("Audio-Visual Director", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Audio-Visual Director", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        print("Audio-Visual Director running. Controls:")
        print("  'q' - Quit")
        print("  't' - Toggle training mode")
        print("  '0' - Add audience sample (in training mode)")
        print("  '1' - Add lecturer sample (in training mode)")
        print("  's' - Stop training and train classifier")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Display frame
                cv2.imshow("Audio-Visual Director", processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('t'):
                    if not self.collecting_training_data:
                        label = int(input("Enter label (0=audience, 1=lecturer): "))
                        self.start_training_mode(label)
                    else:
                        self.stop_training_mode()
                elif key == ord('0') and self.collecting_training_data:
                    self.add_training_sample(0)
                elif key == ord('1') and self.collecting_training_data:
                    self.add_training_sample(1)
                elif key == ord('s') and self.collecting_training_data:
                    self.stop_training_mode()
                
        except KeyboardInterrupt:
            print("\nShutdown requested...")
        finally:
            # Cleanup
            self.audio_director.stop()
            self.cap.release()
            cv2.destroyAllWindows()
            print("Audio-Visual Director stopped")


def main():
    """Main function to run the integrated system."""
    try:
        director = AudioCameraDirector()
        director.run()
    except Exception as e:
        print(f"Error running AudioCameraDirector: {e}")


if __name__ == "__main__":
    main()
