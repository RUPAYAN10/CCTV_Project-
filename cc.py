import cv2
import face_recognition



# Set the URL using the phone's IP address and port with the MJPEG stream
# url = "http://192.168.1.6:4747/video"
# video_capture.open(url)

# Use the standard USB camera index (0)
# for webcam (0)

video_capture=cv2.VideoCapture(1)



# Load the reference face images for comparison
reference_image_path1 = 'C:\\CCTV_Project\\env\\img\\rd.jpg'
reference_image_path2 = 'C:\\CCTV_Project\\env\\img\\guria.jpg'

reference_image1 = face_recognition.load_image_file(reference_image_path1)
reference_face_encoding1 = face_recognition.face_encodings(reference_image1)[0]

reference_image2 = face_recognition.load_image_file(reference_image_path2)
reference_face_encoding2 = face_recognition.face_encodings(reference_image2)[0]

while True:
    # Read a frame from the camera feed
    ret, frame = video_capture.read()

    # Check if the frame was read successfully
    if not ret:
        print("Error: Could not read frame.")
        break

    # Detect face locations in the frame
    face_locations = face_recognition.face_locations(frame)

    for face_location in face_locations:
        # Extract face encodings once per frame
        top, right, bottom, left = face_location
        face_encoding = face_recognition.face_encodings(frame, [face_location])[0]

        # Compare the face encoding with the reference faces
        matches1 = face_recognition.compare_faces([reference_face_encoding1], face_encoding)
        matches2 = face_recognition.compare_faces([reference_face_encoding2], face_encoding)

        # Draw a rectangle and label on the face
        # cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        if matches1[0]:
            name = "Rupayan"
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        elif matches2[0]:
            name = "Traidha"
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    
        else:
            name = "Unknown"
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            
            
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 2, bottom + 18), font, 0.6, (255, 255, 0), 1)

    # Display the frame with recognized faces
    cv2.imshow('Face Recognition', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video stream and close windows
video_capture.release()
cv2.destroyAllWindows()