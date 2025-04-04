# Проект: Распознавание автомобильных номеров (Number Plate Recognition)

Данный проект предназначен для детекции и распознавания автомобильных номеров из видеопотока (RTSP). В основе лежит модель YOLO (из библиотеки [Ultralytics](https://github.com/ultralytics/ultralytics)) для детекции номерных знаков и библиотека [easyocr](https://github.com/JaidedAI/EasyOCR) для OCR‑распознавания символов.

## Основные файлы

- **app.py**  
  - Загружает модель `license_plate_detector.pt` (YOLO), предназначенную для детекции автомобильных номеров.  
  - Подключается к RTSP‑потоку.  
  - Обнаруживает номера на кадрах, вырезает их и распознаёт с помощью easyocr.  
  - Проводит дополнительную валидацию (функция `check_string`) и транслитерацию (функция `transliterate`) для приведения символов к стандартному формату.  
  - Сохраняет результаты распознавания (номерные знаки) и скриншоты в соответствующие файлы.

- **class_load_rtsp.py**  
  - Модифицированная версия класса `LoadStreams` из репозитория Ultralytics.  
  - Позволяет работать с несколькими RTSP‑потоками, организуя буфер кадров и перезапуск в случае потери соединения.

- **license_plate_detector.pt**  
  - Предобученная модель YOLO для детекции автомобильных номеров.

- **requirements.txt**  
  - Список необходимых библиотек и версий для работы проекта (YOLO, easyocr, OpenCV и др.).

## Ключевые особенности

1. **Детекция**  
   Используется YOLO для быстрой и точной детекции области номерного знака на кадре.

2. **Распознавание**  
   Для OCR‑анализа применяется easyocr, обученная на кириллических и латинских символах (режим `ru`).

3. **Валидация и транслитерация**  
   - Функция `check_string` проверяет соответствие распознанного текста формату номерных знаков (регулярное выражение).  
   - Функция `transliterate` заменяет кириллические буквы на латинские аналоги (А->A, В->B, и т.д.), упрощая дальнейшую обработку и хранение.

4. **Переподключение к RTSP**  
   Если поток недоступен, скрипт пытается автоматически переподключиться (функция `reconnect_stream`).

5. **Сохранение результатов**  
   - Распознанные номера сохраняются в файл `number_plates.txt`.  
   - Скриншоты с обнаруженными номерами сохраняются в папку `screens/`.

## Применение

Подобный подход может использоваться в системах видеонаблюдения и контроля доступа, где требуется автоматическое распознавание автомобильных номеров в режиме реального времени.  
