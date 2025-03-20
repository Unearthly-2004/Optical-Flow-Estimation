import streamlit as st
import cv2
import numpy as np
import tempfile
import os

# Streamlit UI
st.title("Lucas-Kanade Optical Flow Experiment")
st.write("Upload a video file to analyze optical flow using the Lucas-Kanade method.")

# File uploader
uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save uploaded video to a temporary file
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video.write(uploaded_file.read())
    temp_video_path = temp_video.name

    # Open video file
    video = cv2.VideoCapture(temp_video_path)
    
    ret, prev_frame = video.read()
    if not ret:
        st.error("Error reading the video file. Please try another one.")
        video.release()
        os.remove(temp_video_path)
    else:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Lucas-Kanade parameters
        lk_params = dict(winSize=(14, 14),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        # Detect good features to track
        prev_corners = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        
        # Create mask for drawing
        mask = np.zeros_like(prev_frame)
        
        # Display the processed video
        stframe = st.empty()
        
        while video.isOpened():
            ret, curr_frame = video.read()
            if not ret:
                break
            
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate Optical Flow
            curr_corners, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_corners, None, **lk_params)
            
            if curr_corners is not None and status is not None:
                good_old = prev_corners[status == 1]
                good_new = curr_corners[status == 1]
                
                for (old, new) in zip(good_old, good_new):
                    x_old, y_old = old.ravel()
                    x_new, y_new = new.ravel()
                    mask = cv2.line(mask, (int(x_new), int(y_new)), (int(x_old), int(y_old)), (0, 255, 0), 2)
                    curr_frame = cv2.circle(curr_frame, (int(x_new), int(y_new)), 5, (0, 0, 255), -1)
                
                output = cv2.add(curr_frame, mask)
                
                # Convert frame to RGB
                output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                
                # Display video
                stframe.image(output, channels="RGB")
                
                # Update previous frame and corners
                prev_gray = curr_gray.copy()
                prev_corners = good_new.reshape(-1, 1, 2)
            
        # Release video
        video.release()
        os.remove(temp_video_path)

st.write("Press **Q** to exit the video preview in case of issues.")
