import cv2
from ultralytics import YOLO
import easyocr
import re
import time


def check_string(s):
    pattern = r'^[A-Za-z\u0410-\u044F]\d{3}[A-Za-z\u0410-\u044F]{2}\d{2}(\d)?$'

    if re.match(pattern, s) and (len(s) == 8 or len(s) == 9):
        return True

def transliterate(text):
    translit_dict = {
        'А': 'A', 'В': 'B', 'Е': 'E', 'К': 'K', 'М': 'M',
        'Н': 'H', 'О': 'O', 'Р': 'P', 'С': 'C', 'Т': 'T',
        'У': 'Y', 'Х': 'X'
    }

    translit_dict.update({str(i): str(i) for i in range(10)})

    transliterated_text = ''.join(translit_dict.get(char, char) for char in text)

    return transliterated_text

def reconnect_stream(video_url):
    cap = cv2.VideoCapture(video_url)
    while not cap.isOpened():
        print("Try to reconnect to the rtsp url...")
        cap.release()
        time.sleep(10)
        cap = cv2.VideoCapture(video_url)
    print("Cap got succesful")
    return cap

model = YOLO('/home/user/Загрузки/license_plate_detector.pt')
reader = easyocr.Reader(['ru'], gpu=True)

rtsp_url = 'your_url'
cap = cv2.VideoCapture(rtsp_url)
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)
number_plates = []
n = 0

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale1 = 3
font_scale2 = 2.5
font_color = (0, 0, 0)
thickness1 = 8
thickness2 = 6

plates_container = []
start_time = time.time()
save_interval = 22
last_frame_with_vehicle = None
time_recognition = None
list_dicts = [{} for _ in range(9)]
while True:
    current_time = time.time()
    empty_plate = cv2.imread('num_plate_clear.jpg')
    ret, frame = cap.read()
    if not ret:
        print('Trouble with cap')
        cap = reconnect_stream(rtsp_url)
    else:
        results = model(frame, conf=0.5, verbose=False)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cropped_image = frame[y1:y2, x1:x2]
                cropped_image_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                text = reader.readtext(cropped_image, allowlist='АВEКМНОРСТУХ0123456789',
                                       contrast_ths=8, adjust_contrast=0.85, add_margin=0.015, width_ths=20,
                                       decoder='beamsearch', text_threshold=0.1, batch_size=8, beamWidth=32)
                for t in text:
                    bbox, texter, score = t
                    if check_string(texter):
                        time_recognition = current_time
                        last_frame_with_vehicle = frame.copy()
                        texter = transliterate(texter)
                        plates_container.append(texter)
        if time_recognition is not None:
            if current_time - time_recognition >= save_interval:
                for plate in plates_container:
                    for idx, c in enumerate(plate):
                        if list_dicts[idx].get(c):
                            list_dicts[idx][c] += 1
                        else:
                            list_dicts[idx][c] = 1
                    if len(plate) == 8:
                        if list_dicts[8].get(''):
                            list_dicts[8][''] += 1
                        else:
                            list_dicts[8][''] = 1
                final_plate = ''
                for i in list_dicts:
                    final_plate += max(i, key=i.get)
                if final_plate:
                    part_first = final_plate[:6]
                    part_region = final_plate[6:]
                    text_size = cv2.getTextSize(final_plate, font, font_scale1, thickness1)[0]
                    text_x = (empty_plate.shape[1] - text_size[0]) // 2
                    text_y = (empty_plate.shape[0] + text_size[1]) // 2
                    cv2.putText(empty_plate, part_first, (text_x, text_y), font, font_scale1, font_color, thickness1)
                    cv2.putText(empty_plate, part_region, (text_x + 450, text_y - 25), font, font_scale2, font_color,
                                thickness2)

                    if last_frame_with_vehicle is not None:
                        plate_height, plate_width = empty_plate.shape[:2]
                        last_frame_with_vehicle[0:plate_height, 0:plate_width] = empty_plate
                        cv2.imwrite(
                            f'./screens/screen{n}_{final_plate}.jpg',
                            last_frame_with_vehicle)
                        n += 1
                        number_plates.append(final_plate)
                        plates_container.clear()
                        list_dicts = [{} for _ in range(9)]
                        time_recognition = None

        cv2.namedWindow('YOLOv8 Detection', cv2.WINDOW_NORMAL)
        cv2.imshow('YOLOv8 Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

file_name = './reports/number_plates.txt'
with open(file_name, 'w') as file:
    for item in number_plates:
        file.write(item + '\n')

print(f'Number plates saved to {file_name}')
cap.release()
cv2.destroyAllWindows()

