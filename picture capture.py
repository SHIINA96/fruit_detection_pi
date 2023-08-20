import cv2, os

cap = cv2.VideoCapture(0)
path = 'images'

while(True):
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)  # flip the frame horizontally

    cv2.imshow('photo capture', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite(os.path.join(path,'captured.jpg'),frame)
        break

cap.release()
cv2.destroyWindow('photo capture')

# process the captured image
os.system('python3 infer.py --model_dir=yolov3_mobilenet_v3_large_voc --image_file=images/captured.jpg')