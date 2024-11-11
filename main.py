from  templeRunHandler import *
from frameCapture import *
import time


startMacro()

# Define the monitor region (customize this for your game window)
monitor = {"top": 50, "left": 0, "width": 500, "height": 900}

# Set up video writer to save the video
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 'XVID' codec for .avi format, 'mp4v' for .mp4
out = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (800, 600))  # Adjust frame rate and resolution

time = 0
timeEnd = 100
# Capture and write frames to video
try:
    while True:
        # Capture the screen
        frame = capture_screen(monitor)
        
        # Write the captured frame to the video
        out.write(frame)
        
        # Optional: Display the frame while capturing (to see it in real time)
        cv2.imshow("Recording", frame)
        
        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        

        # NEURAL NETWORK CODE HERE ALONG WITH WHAT BUTTON TO PRESS
        if time > timeEnd:
            tryReset() # if this returns true, YOU ARE DEAD! (in game), otherwise, you are alive
            time = 0
        else:
            time += 1
finally:
    # Release the video writer and close OpenCV windows
    out.release()
    cv2.destroyAllWindows()
    