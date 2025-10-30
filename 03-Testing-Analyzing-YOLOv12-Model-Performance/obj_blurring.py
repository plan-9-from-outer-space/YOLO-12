#----------Steps-------------------#
#Read an Image using OpenCV
#Load the YOLOv12 Model and Perform Object Detection
#Add the Confidence Value 'conf'
#Add NMS IOU 'iou'
#Add the classes parameter 'classes'
#Add the maximum detection parameter 'max_det
#Add show image parameter 'show = True'
#Add 'save_txt = True', save detection results in a text file
#Add 'save_crop = True' parameter
#Object Detection on Image
#Object Detection on Video and FPS Calculation
#----------------------------------#
#Import All the Required Libraries
import cv2
import math
import time
from ultralytics import YOLO
#Read the Image/Video/Live WebcamFeed using OpenCV
cap = cv2.VideoCapture("Resources/Videos/video.mp4")
#cap = cv2.VideoCapture(0)
#Get Video Properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))
output_video = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
#Load the YOLOv12 Model
model = YOLO("yolo12n.pt")
#Classes in the MS COCO Dataset
cocoClassNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                  "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                  "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                  "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                  "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"]
blur_ratio = 50
ctime = 0
ptime = 0
count = 0
while True:
    ret, frame = cap.read()
    if ret:
        count += 1
        print(f"Frame Number: {count}")
        #Object Detection using YOLOv12
        results = model.predict(frame, conf=0.15, iou = 0.1, classes=[0])
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                blur = frame[y1:y2, x1:x2]
                blur_obj = cv2.blur(blur, (blur_ratio, blur_ratio))
                frame[y1:y2, x1:x2] = blur_obj
                cv2.rectangle(frame, (x1, y1), (x2, y2), [255,0,0], 2)
                classNameInt = int(box.cls[0])
                className = cocoClassNames[classNameInt]
                conf = math.ceil(box.conf[0] * 100)/100
                label = className + ":" + str(conf)
                text_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
                c2 = x1 + text_size[0], y1 - text_size[1] - 3
                cv2.rectangle(frame, (x1, y1), c2, [255, 0, 0], -1)
                cv2.putText(frame, label, (x1, y1 -  2), 0, 0.5, [255,255,255], thickness=1, lineType = cv2.LINE_AA)
        #Display the Video using OpenCV
        ctime = time.time()
        fps = 1/(ctime - ptime)
        ptime = ctime
        cv2.putText(frame, "FPS" + ":" + str(int(fps)), (30, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        output_video.write(frame)
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('1'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()