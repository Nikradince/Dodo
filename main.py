import cv2
import pandas as pd
import argparse
import os
from ultralytics import YOLO

CONF_THRESHOLD = 0.25

# сколько времени человек может "пропасть", но стол все еще считаем занятым
EMPTY_TIMEOUT = 2.0

# минимальная длительность состояния (анти-дребезг)
MIN_EVENT_GAP = 1.0


# ---------------- ROI CHECK ---------------- #
def foot_point_in_roi(box, roi):
    x1, y1, x2, y2 = box
    tx, ty, tw, th = roi

    foot_x = (x1 + x2) // 2
    foot_y = y2

    margin = 10

    return (tx - margin < foot_x < tx + tw + margin and
            ty - margin < foot_y < ty + th + margin)


# ---------------- DETECTION ---------------- #
def detect_people(model, frame, roi):
    results = model(frame, classes=[0], verbose=False)[0]

    person_in_zone = False

    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # foot point
        foot_x = (x1 + x2) // 2
        foot_y = y2
        cv2.circle(frame, (foot_x, foot_y), 5, (0, 255, 255), -1)

        if foot_point_in_roi((x1, y1, x2, y2), roi):
            person_in_zone = True

    return person_in_zone


# ---------------- STATE MACHINE ---------------- #
class StateMachine:
    def __init__(self):
        self.is_occupied = False
        self.last_seen_person_time = 0
        self.last_event_time = -999
        self.events = []

    def update(self, person_in_zone, current_time):
        # обновляем время последнего обнаружения человека
        if person_in_zone:
            self.last_seen_person_time = current_time

        # определяем состояние через время
        time_since_seen = current_time - self.last_seen_person_time

        if time_since_seen < EMPTY_TIMEOUT:
            new_state = True
        else:
            new_state = False

        # если состояние не изменилось — ничего не делаем
        if new_state == self.is_occupied:
            return

        # анти-дребезг (защита от быстрых переключений)
        if current_time - self.last_event_time < MIN_EVENT_GAP:
            return

        prev = self.is_occupied
        self.is_occupied = new_state
        self.last_event_time = current_time

        # фиксируем события
        if not prev and self.is_occupied:
            event = "approach"
        elif prev and not self.is_occupied:
            event = "empty"
        else:
            return

        self.events.append({
            "event": event,
            "time": current_time
        })

        print(f"[{current_time:.2f}] EVENT: {event}")


# ---------------- METRICS ---------------- #
def compute_metrics(events):
    df = pd.DataFrame(events)

    delays = []

    for i in range(len(df) - 1):
        if df.iloc[i]['event'] == 'empty' and df.iloc[i + 1]['event'] == 'approach':
            delays.append(df.iloc[i + 1]['time'] - df.iloc[i]['time'])

    if delays:
        avg = sum(delays) / len(delays)
        print(f"\n✅ Avg delay: {avg:.2f} sec")
    else:
        print("\n⚠️ Недостаточно данных")

    return df


# ---------------- MAIN ---------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print("Видео не найдено")
        return

    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(args.video)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps < 10 or fps > 60:
        print("⚠️ FPS некорректный, используем 25")
        fps = 25

    ret, first_frame = cap.read()
    if not ret:
        print("Ошибка чтения видео")
        return

    h, w = first_frame.shape[:2]

    # выбор ROI
    cv2.namedWindow("Select Table", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select Table", 1280, 720)

    roi = cv2.selectROI("Select Table", first_frame, fromCenter=False)
    cv2.destroyAllWindows()

    roi = tuple(map(int, roi))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    out = cv2.VideoWriter(
        'output.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'),
        float(fps),
        (w, h)
    )

    sm = StateMachine()
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        time_sec = frame_id / fps

        person_in_zone = detect_people(model, frame, roi)
        sm.update(person_in_zone, time_sec)

        # визуализация
        color = (0, 0, 255) if sm.is_occupied else (0, 255, 0)
        label = "OCCUPIED" if sm.is_occupied else "FREE"

        tx, ty, tw, th = roi
        cv2.rectangle(frame, (tx, ty), (tx + tw, ty + th), color, 3)
        cv2.putText(frame, label, (tx, ty - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        out.write(frame)

        try:
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except:
            pass

        frame_id += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    df = compute_metrics(sm.events)
    df.to_csv("events_log.csv", index=False)


if __name__ == "__main__":
    main()