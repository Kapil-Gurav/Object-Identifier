import cv2
from gui_buttons import Buttons n

#initialize button
button = Buttons()
button.add_button("person", 20, 20)
button.add_button("cell phone", 20, 90)
button.add_button("scissors", 20, 160)
button.add_button("bottle", 20, 230)
button.add_button("mouse", 20, 300)
button.add_button("keyboard", 20, 370)
button.add_button("book", 20, 440)

# Opencv DNN
net = cv2.dnn.readNet("C://Users//kartik//objectidentifier//dnn_model//yolov4-tiny.weights",
                      "C://Users//kartik//objectidentifier//dnn_model//yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1 / 255)

# Load class lists
classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

print("Object list")
print(classes)

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# FULL HD 1920 x 1080


def click_button(event, x, y, flags, params):
    global button_person
    if event == cv2.EVENT_LBUTTONDOWN:
        button.button_click(x, y)

# Create window
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", click_button)

while True:
    # Get frames
    ret, frame = cap.read()

    # get active button list
    active_buttons = button.active_buttons_list()
    print("active", active_buttons)

    # Object Detection
    (class_ids, scores, bboxes) = model.detect(frame)
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox
        class_name = classes[class_id]

        if class_name in active_buttons:
            cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (200, 0, 50), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 0, 50), 3)

    #display buttons
    button.display_buttons(frame)

    #print("class ids", class_ids)
    #print("scores", scores)
    #print("bboxes", bboxes)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(2)
    if key == 2:
        break

cap.release()
cv2.destroyAllWindows()