import cv2
from ultralytics import YOLO

model = YOLO("results/best.pt") 

# IP Webcam 앱 화면 하단에 뜨는 주소 뒤에 /video 를 반드시 붙여주세요.
# (예시 주소이므로, 실제 스마트폰에 뜬 IP로 숫자를 변경해야 합니다.)
cam_url = "http://put-your-ip-here/video"

cap = cv2.VideoCapture(cam_url)

print("인식 프로그램 시작. 'q'를 누르면 종료됩니다.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: 
        print("프레임을 불러올 수 없습니다. 주소를 다시 확인해 주세요.")
        break

    results = model(frame, stream=True, conf=0.2, augment=True)
    
    current_object = "Searching..." 

    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            current_object = model.names[class_id] 
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) 
            
            conf = box.conf[0]
            cv2.putText(frame, f"{current_object} {conf:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    display_text = f"Status: {current_object.upper()}"
    cv2.putText(frame, display_text, (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3) 

    cv2.imshow("YOLOv8 Waste Detection Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()