import cv2
from tracker1 import ObjectCounter  # Make sure this points to the correct module where ObjectCounter is defined

# Define the mouse callback function
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse moved to: [{x}, {y}]")

# Open the video file
cap = cv2.VideoCapture('vid1.mp4')

# Define region points (line or polygon)
region_points = [(278, 1), (278, 599)]  # Example: Line region for counting

# Initialize the ObjectCounter
counter = ObjectCounter(
    region=region_points,
    model="yolo12n.pt",           # YOLO model file
#    classes=[0],               # Detect only class 0 (e.g., person)
    show_in=True,
    show_out=True,
    line_width=2
)

# Create a named OpenCV window and set the mouse callback
cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", RGB)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 2 != 0:
        continue  # Skip odd frames to reduce processing load

    # Resize frame to a fixed size
    frame = cv2.resize(frame, (1020, 500))

    # Process the frame using ObjectCounter
    results = counter.process(frame)

    # Display the frame with object counting overlays
    cv2.imshow("RGB", results.plot_im)

    # Print stats to console
    print(f"IN: {results.in_count}, OUT: {results.out_count}, Total Tracks: {results.total_tracks}")

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
