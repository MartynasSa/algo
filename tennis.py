import serial
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import deque

PORT = "COM5"
BAUD = 115200
DURATION = 5      # seconds to capture per batch
TRAIL_LENGTH = 50 # previous points to show in trail
INTERVAL = 1      # seconds to wait before next batch

# Setup interactive 3D plot
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-2000, 2000)
ax.set_ylim(-2000, 2000)
ax.set_zlim(-2000, 2000)
ax.set_xlabel('X Acc')
ax.set_ylabel('Y Acc')
ax.set_zlabel('Z Acc')
ax.set_title('Tennis Racket Swing')

ball_plot = None
trail_plot = None

# Open serial port
with serial.Serial(PORT, BAUD, timeout=1) as ser:
    time.sleep(2)  # wait for ESP32 reset

    while True:  # continuous capture and redraw
        # Buffers for this batch
        x_trail = deque(maxlen=TRAIL_LENGTH)
        y_trail = deque(maxlen=TRAIL_LENGTH)
        z_trail = deque(maxlen=TRAIL_LENGTH)

        start_time = time.time()
        while time.time() - start_time < DURATION:
            line = ser.readline().decode(errors='ignore').strip()
            if line.startswith("Accel:"):
                try:
                    parts = line.split("|")
                    acc_vals = [int(v) for v in parts[0].replace("Accel:", "").split()]

                    x_trail.append(acc_vals[0])
                    y_trail.append(acc_vals[1])
                    z_trail.append(acc_vals[2])

                    # Remove previous plots
                    if ball_plot:
                        ball_plot.remove()
                    if trail_plot:
                        trail_plot.remove()

                    # Draw trail and ball
                    trail_plot, = ax.plot(list(x_trail), list(y_trail), list(z_trail), c='blue', alpha=0.7)
                    ball_plot = ax.scatter(x_trail[-1], y_trail[-1], z_trail[-1], c='red', s=100)

                    plt.pause(0.01)
                except:
                    pass

        # After 5 seconds, wait interval before next batch
        time.sleep(INTERVAL)
