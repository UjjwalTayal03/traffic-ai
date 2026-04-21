import os
import cv2
from flask import Flask, render_template, request, send_from_directory
from ultralytics import YOLO

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
EVIDENCE_FOLDER = "evidence"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(EVIDENCE_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = YOLO("yolov8n.pt")

vehicle_classes = ["car", "motorcycle", "bus", "truck"]


def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if fps == 0:
        fps = 25

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    violations = 0

    counted = set()
    previous_pos = {}

    stop_line_y = int(height * 0.65)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Signal changes automatically
        signal_red = (frame_count // 100) % 2 == 0
        signal_text = "RED" if signal_red else "GREEN"

        results = model(frame)[0]

        for box in results.boxes:

            cls_id = int(box.cls[0])
            name = model.names[cls_id]

            if name in vehicle_classes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.putText(
                    frame,
                    name,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                center_y = (y1 + y2) // 2

                # rough vehicle tracking id
                vehicle_id = (x1 // 60, y1 // 60)

                # check crossing only once
                if vehicle_id in previous_pos:

                    prev_y = previous_pos[vehicle_id]

                    if (
                        signal_red
                        and prev_y <= stop_line_y
                        and center_y > stop_line_y
                        and vehicle_id not in counted
                    ):

                        counted.add(vehicle_id)
                        violations += 1

                        cv2.putText(
                            frame,
                            "VIOLATION!",
                            (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2
                        )

                        cv2.imwrite(
                            f"evidence/violation_{violations}.jpg",
                            frame
                        )

                previous_pos[vehicle_id] = center_y

        # Stop line
        cv2.line(
            frame,
            (0, stop_line_y),
            (width, stop_line_y),
            (0, 0, 255),
            3
        )

        # Signal display
        color = (0, 0, 255) if signal_red else (0, 255, 0)

        cv2.putText(
            frame,
            signal_text,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            3
        )

        # Violation count
        cv2.putText(
            frame,
            f"Violations: {violations}",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        out.write(frame)

    cap.release()
    out.release()

@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        file = request.files["video"]

        if file:

            # clear old evidence images
            for img in os.listdir(EVIDENCE_FOLDER):
                os.remove(os.path.join(EVIDENCE_FOLDER, img))

            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)

            output_path = os.path.join(OUTPUT_FOLDER, "result.avi")

            process_video(path, output_path)

            images = os.listdir(EVIDENCE_FOLDER)

            return render_template(
                "index.html",
                done=True,
                images=images
            )

    return render_template("index.html", done=False, images=[])

@app.route("/outputs/<filename>")
def output_file(filename):
    return send_from_directory("outputs", filename)


@app.route("/evidence/<filename>")
def evidence_file(filename):
    return send_from_directory("evidence", filename)


if __name__ == "__main__":
    app.run(debug=True)