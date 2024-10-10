import cv2
import mediapipe as mp


def detect_face(frame,face_detection):

    H,W,_=frame.shape

    out = face_detection.process(frame)

    if out.detections is not None:
        for detection in out.detections:
            location_data=detection.location_data
            bbox=location_data.relative_bounding_box

            x1,y1,w,h=bbox.xmin,bbox.ymin,bbox.width,bbox.height

            x1=int(x1*W)
            y1=int(y1*H)
            w=int(w*W)
            h=int(h*H)

            #draw box
            frame=cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (0, 0, 255), 5)

    return frame


mp_face_detection=mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection=0,min_detection_confidence=0.8) as face_detection:
    cam = cv2.VideoCapture(0)

    while True:
        ret, frame = cam.read()
        frame=detect_face(frame,face_detection)

        cv2.imshow('webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and destroy all OpenCV windows
    cam.release()
    cv2.destroyAllWindows()