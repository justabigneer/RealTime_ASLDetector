
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2 as cv 
from ultralytics import YOLO
import time  
import tensorflow as tf 
from tensorflow import keras
import numpy as np
import traceback

print("Loading models...", flush=True)

# Load YOLO
yolo_model_path = "D:\\GitHub\ASL(new)\\best.pt"
if not os.path.exists(yolo_model_path):
    print(f"YOLO model not found, using pre-trained YOLOv8...")
    yolo_model = YOLO('yolov8n.pt')  
    use_person_detection = True
else:
    print(f"Loading custom YOLO model: {yolo_model_path}")
    yolo_model = YOLO(yolo_model_path)
    use_person_detection = False

# Load CNN with error handling
cnn_model_path = "D:\\GitHub\\ASL(new)\\cnn.keras"
try:
    cnn_model = keras.models.load_model(cnn_model_path)
    print(f"CNN Model loaded successfully!")
    
    
    print(f"CNN input shape: {cnn_model.input_shape}")
    print(f"CNN output shape: {cnn_model.output_shape}")
    
    num_classes = cnn_model.output_shape[-1]
    print(f"Model has {num_classes} output classes")
    
except Exception as e:
    print(f"Error loading CNN model: {e}")
    print("Creating a dummy model for testing...")
    # Create a simple dummy model
    cnn_model = keras.Sequential([
        keras.layers.Input(shape=(64, 64, 1)),
        keras.layers.Flatten(),
        keras.layers.Dense(26, activation='softmax')
    ])
    cnn_model.compile(optimizer='adam', loss='categorical_crossentropy')
    num_classes = 26


if num_classes == 26:
    class_names = ['A','B','C','D','E','F','G','H','I','K','L','M',
                   'N','O','P','Q','R','S','T','U','V','W','X','Y','del','space']
elif num_classes == 29:
    class_names = ['A','B','C','D','E','F','G','H','I','J','K','L','M',
                   'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
                   'del','nothing','space']
else:
    print(f"⚠ Model has {num_classes} classes, using generic names")
    class_names = [f'Class_{i}' for i in range(num_classes)]

print(f"Using class names: {class_names}")

# Webcam
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

print("\nControls: Q=Quit, S=Save, D=Debug, C=Clear detection history\n")

prev_time = time.time()
frame_count = 0
debug_mode = True  
detection_history = []

def process_hand_region(hand_img):
    try:
        # Convert to grayscale if needed
        if len(hand_img.shape) == 3:
            gray = cv.cvtColor(hand_img, cv.COLOR_BGR2GRAY)
        else:
            gray = hand_img
        
        
        resized = cv.resize(gray, (64, 64))
        
        
        normalized = resized.astype('float32') / 255.0
        
       
        expected_channels = cnn_model.input_shape[-1]
        
        if expected_channels == 1:
            
            batched = np.expand_dims(normalized, axis=0)
            batched = np.expand_dims(batched, axis=-1)
        elif expected_channels == 3:
            
            rgb = cv.cvtColor(resized, cv.COLOR_GRAY2RGB)
            normalized_rgb = rgb.astype('float32') / 255.0
            batched = np.expand_dims(normalized_rgb, axis=0)
        else:
           
            batched = np.expand_dims(normalized, axis=0)
            batched = np.expand_dims(batched, axis=-1)
        
        return batched
        
    except Exception as e:
        print(f"Error processing hand region: {e}")
        
        return np.random.randn(1, 64, 64, 1).astype('float32')


try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        
        frame = cv.flip(frame, 1)
        display = frame.copy()
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if current_time > prev_time else 0
        prev_time = current_time
        
        
        try:
            if use_person_detection:
                results = yolo_model(frame, conf=0.3, classes=[0], verbose=False)
            else:
                results = yolo_model(frame, conf=0.15, verbose=False)  # Lower confidence
        except Exception as e:
            print(f"YOLO error: {e}")
            continue
        
        
        hand_detected = False
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    try:
                        conf = float(box.conf[0])
                        
                        if conf < 0.15:  
                            continue
                        
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Ensure valid coordinates
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                        
                        if x2 <= x1 or y2 <= y1:
                            continue
                        
                        
                        if use_person_detection:
                            person_height = y2 - y1
                            hand_y1 = y1 + int(person_height * 0.6)
                            hand_y2 = y2
                            hand_region = frame[hand_y1:hand_y2, x1:x2]
                            draw_y1, draw_y2 = hand_y1, hand_y2
                        else:
                            hand_region = frame[y1:y2, x1:x2]
                            draw_y1, draw_y2 = y1, y2
                        
                        if hand_region.size == 0:
                            continue
                        
                        # Process and predict
                        processed = process_hand_region(hand_region)
                        
                        try:
                            predictions = cnn_model.predict(processed, verbose=0)
                            
                            # Safety check
                            if len(predictions[0]) == 0:
                                print("Empty predictions")
                                continue
                            
                            pred_idx = np.argmax(predictions[0])
                            confidence = predictions[0][pred_idx]
                            
                            # Ensure index is valid
                            if pred_idx >= len(class_names):
                                print(f"Warning: Index {pred_idx} out of range for {len(class_names)} classes")
                                pred_idx = pred_idx % len(class_names)  # Wrap around
                            
                            letter = class_names[pred_idx]
                            
                            # Draw
                            color = (0, 255, 0) if confidence > 0.5 else (0, 165, 255)
                            cv.rectangle(display, (x1, draw_y1), (x2, draw_y2), color, 2)
                            
                            label = f"{letter}: {confidence:.2f}"
                            label_size = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                            label_y = max(draw_y1 - label_size[1] - 10, 0)
                            
                            cv.rectangle(display, (x1, label_y), 
                                       (x1 + label_size[0], draw_y1), color, -1)
                            cv.putText(display, label, (x1, draw_y1 - 5),
                                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            
                            hand_detected = True
                            detection_history.append(letter)
                            if len(detection_history) > 10:
                                detection_history.pop(0)
                                
                            if debug_mode:
                                print(f"Detected: {letter} (confidence: {confidence:.2f})")
                                
                        except Exception as e:
                            if debug_mode:
                                print(f"Prediction error: {e}")
                            continue
                            
                    except Exception as e:
                        if debug_mode:
                            print(f"Box processing error: {e}")
                        continue
        
        #  display
        status = "HAND DETECTED" if hand_detected else "NO HAND - Show hand clearly"
        status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
        cv.putText(display, status, (20, 30), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        cv.putText(display, f"FPS: {int(fps)}", (20, 60), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv.putText(display, f"Frame: {frame_count}", (20, 90), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if detection_history:
            recent = "".join(detection_history[-5:])
            cv.putText(display, f"Recent: {recent}", (20, 120), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Show detection tips if no hand
        if not hand_detected:
            cv.putText(display, "TIP: Move hand closer", 
                      (frame.shape[1] - 250, 30),
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            cv.putText(display, "Good lighting helps", 
                      (frame.shape[1] - 250, 60),
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        # Show
        cv.imshow("ASL Recognition - Press Q to quit", display)
        
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
        elif key == ord('c'):
            detection_history.clear()
            print("Cleared detection history")
        elif key == ord('s'):
            filename = f"capture_{frame_count}.jpg"
            cv.imwrite(filename, frame)
            print(f"Saved: {filename}")
        
        frame_count += 1
        
except KeyboardInterrupt:
    print("\nInterrupted by user")
except Exception as e:
    print(f"\nFatal error in main loop: {e}")
    traceback.print_exc()
finally:
    print("\nCleaning up...")
    cap.release()
    cv.destroyAllWindows()
    print(f"Processed {frame_count} frames")
    print(f"Final detection history: {''.join(detection_history)}")
    print("Done!")