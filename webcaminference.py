import torch
import cv2

# Load the model
model = torch.hub.load('ultralytics/yolov3', 'custom', path='C:/Users/Asus/yolov3/best.pt', force_reload=True)
model.eval()

# Open webcam (0 for the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Perform inference
    results = model(frame)

    # Get detections (bounding boxes and labels)
    detections = results.xyxy[0].cpu().numpy()  # xyxy format: [x_min, y_min, x_max, y_max, confidence, class]

    for det in detections:
        # Unpack the detection: coordinates, confidence, and class label
        x_min, y_min, x_max, y_max, conf, cls = det

        # Draw bounding box
        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

        # Add label and confidence
        label = f'Class: {int(cls)}, Conf: {conf:.2f}'
        cv2.putText(frame, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Print coordinates to the console
        print(f'Bounding box: [{x_min}, {y_min}, {x_max}, {y_max}] | Confidence: {conf:.2f} | Class: {int(cls)}')

    # Display the resulting frame with bounding boxes and labels
    cv2.imshow('Webcam Inference', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()