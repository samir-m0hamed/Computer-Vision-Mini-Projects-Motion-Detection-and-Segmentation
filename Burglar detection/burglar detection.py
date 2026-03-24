import cv2
import numpy as np
import os
import threading
import subprocess
from pathlib import Path
import time
import winsound

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
videos_dir = os.path.join(script_dir, "videos")

# Setup output directory
output_dir = os.path.join(script_dir, "burglar_detection_results")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get emergency alarm sound from folder
print("Loading emergency alarm sound...")
alarm_file = os.path.join(script_dir, "emergency sound effect", "Sound Effects Emergency Alarm.mp3")
if os.path.exists(alarm_file):
    print(f"✓ Emergency alarm found: {alarm_file}")
    print("✓ Windows sound system ready!")
    alarm_ready = True
else:
    print(f"⚠️ WARNING: Alarm file not found at {alarm_file}")
    alarm_ready = False

# Global variable for alarm control
alarm_playing = False
alarm_start_time = None  # Track when alarm started
ALARM_DURATION = 30  # Alarm will play for 30 seconds after first detection

# Motion detection parameters - Increased sensitivity for small movements
MIN_CONTOUR_AREA = 50  # Much more sensitive - detects very small movements
THRESHOLD_SENSITIVITY = 5  # Very sensitive threshold for motion detection

# Select device (not needed for motion detection)
print("Using motion detection for burglar tracking...")

print("Motion detection ready!")

# Alarm sound function - plays the alarm file continuously
def play_alarm():
    """Play emergency alarm sound continuously"""
    global alarm_playing, alarm_start_time
    if alarm_ready and not alarm_playing:
        alarm_playing = True
        alarm_start_time = time.time()  # Record when alarm started
        print(f"🚨 BURGLAR ALERT! Sound activated for {ALARM_DURATION} seconds!")
        
        def alarm_thread():
            global alarm_playing
            try:
                # Try pygame first (if available)
                try:
                    import pygame
                    pygame.mixer.init()
                    pygame.mixer.music.load(alarm_file)
                    pygame.mixer.music.play(-1)  # -1 for loop
                    while alarm_playing and pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                    pygame.mixer.music.stop()
                except ImportError:
                    # Fallback: Use winsound for continuous beep
                    while alarm_playing:
                        winsound.Beep(800, 500)  # 800Hz beep for 500ms
                        time.sleep(0.3)  # Short pause between beeps
                        
            except Exception as e:
                print(f"⚠️ Could not play alarm: {e}")
                alarm_playing = False
        
        # Play in background thread immediately
        thread = threading.Thread(target=alarm_thread, daemon=True)
        thread.start()

def stop_alarm():
    """Stop the alarm sound"""
    global alarm_playing, alarm_start_time
    alarm_playing = False
    alarm_start_time = None  # Reset timer
    try:
        # Try to stop pygame if it's playing
        try:
            import pygame
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
        except:
            pass
        
        # Kill all Windows Media Player processes (ignore errors if none exist)
        subprocess.run(['taskkill', '/f', '/im', 'wmplayer.exe', '/t'], 
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, 
                      capture_output=True)
        
        # Kill any PowerShell processes (ignore errors if none exist)
        subprocess.run(['taskkill', '/f', '/im', 'powershell.exe', '/t'], 
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, 
                      capture_output=True)
        
        print("🔇 Alarm sound stopped!")
    except Exception as e:
        # Don't print error for normal cases (no processes to kill)
        if "No tasks" not in str(e):
            print(f"⚠️ Could not stop alarm: {e}")

# Get video files from videos folder
video_files = list(Path(videos_dir).glob("*.mp4")) + list(Path(videos_dir).glob("*.mkv")) + list(Path(videos_dir).glob("*.avi"))

if not video_files:
    print("ERROR: No video files found in videos folder!")
    exit()

print(f"Found {len(video_files)} video file(s)")

# Process each video
for video_file in video_files:
    print(f"\n{'='*60}")
    print(f"Processing: {video_file.name}")
    print(f"Status: Motion Detection Mode")
    print('='*60)
    
    # Open video
    cap = cv2.VideoCapture(str(video_file))
    
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_file.name}")
        continue
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Resolution: {frame_width}x{frame_height}")
    print(f"FPS: {fps}")
    print(f"Total Frames: {total_frames}")
    print(f"📊 Analysis: Every frame with motion detection")
    print(f"⚡ Playback: ULTRA FAST mode")
    print(f"🚨 Sound: {'Ready (PowerShell Media Player)' if alarm_ready else 'Not available'}")
    
    # Setup VideoWriter - normal speed playback
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_name = f"burglar_detection_{video_file.stem}.mp4"
    output_video_path = os.path.join(output_dir, output_video_name)
    # Write at normal speed
    output_fps = fps if fps > 0 else 30
    out = cv2.VideoWriter(output_video_path, fourcc, output_fps, (frame_width, frame_height))
    
    frame_count = 0
    burglar_detected = False
    burglar_frames = []
    # alarm_playing = False  # Now global
    prev_frame = None  # For motion detection
    
    print(f"Processing frames with motion detection...")
    print("🔍 Monitoring for motion... (Press Q to stop)")
    
    # Start with a warning sound at the beginning
    if alarm_ready:
        try:
            # Quick beep at start
            winsound.Beep(800, 200)  # 800Hz for 200ms
            print("🔊 System ready - monitoring started!")
        except:
            pass
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Convert to grayscale for motion detection (smaller size for speed)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (320, 180))  # Smaller size for faster processing
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            # Initialize previous frame
            if prev_frame is None:
                prev_frame = gray
                continue
            
            # Compute frame difference
            frame_diff = cv2.absdiff(prev_frame, gray)
            prev_frame = gray
            
            # Threshold the difference
            thresh = cv2.threshold(frame_diff, THRESHOLD_SENSITIVITY, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            motion_detected = False
            display_frame = frame.copy()
            
            # Calculate scale factors for accurate bounding box positioning
            scale_x = frame.shape[1] / 320  # Original width / resized width
            scale_y = frame.shape[0] / 180  # Original height / resized height
            
            for contour in contours:
                if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
                    continue
                
                motion_detected = True
                # Draw bounding box around motion (scaled to original frame)
                (x, y, w, h) = cv2.boundingRect(contour)
                # Scale coordinates back to original frame size
                x_orig = int(x * scale_x)
                y_orig = int(y * scale_y)
                w_orig = int(w * scale_x)
                h_orig = int(h * scale_y)
                
                cv2.rectangle(display_frame, (x_orig, y_orig), (x_orig + w_orig, y_orig + h_orig), (0, 0, 255), 3)
                cv2.putText(display_frame, "MOTION DETECTED", (x_orig, y_orig - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # If motion detected
            if motion_detected:
                if not burglar_detected:
                    print(f"\n!!! MOTION DETECTED at frame {frame_count} !!!")
                    burglar_detected = True
                    # Start continuous alarm sound immediately (will play for ALARM_DURATION seconds)
                    play_alarm()
                
                burglar_frames.append(frame_count)
            else:
                # No motion detected - but alarm continues for ALARM_DURATION seconds
                if burglar_detected:
                    print(f"Motion stopped at frame {frame_count} (alarm continues...)")
                    burglar_detected = False
                    # Don't stop alarm here - let it play for the full duration
            
            # Check if alarm should be stopped (after ALARM_DURATION seconds)
            if alarm_playing and alarm_start_time and (time.time() - alarm_start_time) > ALARM_DURATION:
                print(f"Alarm duration ({ALARM_DURATION}s) completed - stopping sound")
                stop_alarm()
            
            # Add frame counter
            cv2.putText(display_frame, f"Frame: {frame_count}/{total_frames}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add detection info
            cv2.putText(display_frame, f"Motion Detected: {motion_detected}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            if burglar_detected:
                cv2.putText(display_frame, "!!! MOTION ALERT !!!", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            
            # Write processed frame
            out.write(display_frame)
            
            # Display frame in real-time (normal speed)
            cv2.imshow('Motion Detection - Processing (Press Q to Stop)', display_frame)
            
            # Press 'q' to quit early (normal delay for smooth playback)
            key = cv2.waitKey(30) & 0xFF  # Normal delay for proper video speed
            if key == ord('q') or key == ord('Q'):
                print("\n⛔ Stopping processing...")
                break
            
            # Progress - show every 50 frames for better monitoring
            if frame_count % 50 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames) - Motion: {motion_detected}")
    
    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user (Ctrl+C or window close)")
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
    finally:
        # Release resources
        cap.release()
        out.release()
        
        # Stop any playing alarm
        if alarm_playing:
            stop_alarm()
        
        print(f"\n--- Video Summary ---")
        print(f"Total Frames Processed: {frame_count}")
        if len(burglar_frames) > 0:
            print(f"BURGLAR DETECTED: YES")
            print(f"Frames with Burglar: {len(burglar_frames)}")
            print(f"First Detection: Frame {burglar_frames[0]}")
            print(f"Last Detection: Frame {burglar_frames[-1]}")
        else:
            print(f"BURGLAR DETECTED: NO (Safe)")
        
        print(f"Output Video: {output_video_path}\n")

cv2.destroyAllWindows()

# Clean up sound processes
try:
    if alarm_ready:
        stop_alarm()
        print("✓ Sound processes cleaned up successfully!")
except:
    pass

print("\n" + "="*40)
print("All videos processed successfully!")
print("="*40)
