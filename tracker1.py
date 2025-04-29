
import cv2
import os
import base64
import threading
from datetime import datetime
from shapely.geometry import Point, Polygon, LineString
from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Set up Google API Key
GOOGLE_API_KEY = ""
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

class ObjectCounter(BaseSolution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.in_count = 0
        self.out_count = 0
        self.counted_ids = []
        self.classwise_counts = {}
        self.region_initialized = False
        self.show_in = self.CFG["show_in"]
        self.show_out = self.CFG["show_out"]
        self.margin = self.line_width * 2

        # Output folder
        today = datetime.now().strftime("%Y-%m-%d")
        self.output_dir = os.path.join("count_logs", today)
        os.makedirs(self.output_dir, exist_ok=True)
        self.log_file = os.path.join(self.output_dir, "object_counts.txt")

        # Gemini model
        self.gemini_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

        self.Polygon = Polygon
        self.Point = Point
        self.LineString = LineString

    def analyze_image_with_gemini(self, image_path, track_id, direction):
        """Analyze cropped person image using Gemini AI."""
        try:
            with open(image_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode("utf-8")

            prompt = """
Analyze the person shown in this image carefully.

Return these two fields in a clean table format:

- Dress Code (example answers: Shirt, T-shirt, Jeans, Dress, Skirt, Jacket, Suit, etc.)
- Gender (example answers: Man, Woman)

Respond ONLY in the exact table format like this:

| Dress Code | Gender |
|------------|--------|
|            |        |
"""

            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            )
            response = self.gemini_model.invoke([message])
            result = response.content.strip()

            # Save Gemini result
            gemini_file = os.path.join(self.output_dir, f"{track_id}_gemini_result.txt")
            with open(gemini_file, "w", encoding="utf-8") as f:
                f.write(f"Track ID: {track_id}\n")
                f.write(f"Direction: {direction}\n\n")
                f.write(result)

        except Exception as e:
            print(f"[Gemini Error]: {e}")

    def count_objects(self, current_centroid, track_id, prev_position, cls, im0, box):
        if prev_position is None or track_id in self.counted_ids:
            return

        crossed = False
        direction = None

        if len(self.region) == 2:
            # Line
            line = self.LineString(self.region)
            if line.intersects(self.LineString([prev_position, current_centroid])):
                if abs(self.region[0][0] - self.region[1][0]) < abs(self.region[0][1] - self.region[1][1]):
                    direction = "IN" if current_centroid[0] > prev_position[0] else "OUT"
                else:
                    direction = "IN" if current_centroid[1] > prev_position[1] else "OUT"
                crossed = True

        elif len(self.region) > 2:
            # Polygon
            polygon = self.Polygon(self.region)
            if polygon.contains(self.Point(current_centroid)):
                region_width = max(p[0] for p in self.region) - min(p[0] for p in self.region)
                region_height = max(p[1] for p in self.region) - min(p[1] for p in self.region)
                direction = "IN" if (
                    region_width < region_height and current_centroid[0] > prev_position[0]
                    or region_width >= region_height and current_centroid[1] > prev_position[1]
                ) else "OUT"
                crossed = True

        if crossed and direction:
            self.classwise_counts[self.names[cls]][direction] += 1
            self.counted_ids.append(track_id)

            # Save cropped image
            x1, y1, x2, y2 = map(int, box)
            crop = im0[y1:y2, x1:x2]
            timestamp_filename = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_name = f"{track_id}_{self.names[cls]}_{direction}_{timestamp_filename}.jpg"
            image_path = os.path.join(self.output_dir, image_name)
            cv2.imwrite(image_path, crop)

            # Save count info
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"{timestamp}, TrackID: {track_id}, Class: {self.names[cls]}, Direction: {direction}\n")

            if direction == "IN":
                self.in_count += 1
            elif direction == "OUT":
                self.out_count += 1

            # Start Gemini analysis thread
            threading.Thread(target=self.analyze_image_with_gemini, args=(image_path, track_id, direction)).start()

    def store_classwise_counts(self, cls):
        if self.names[cls] not in self.classwise_counts:
            self.classwise_counts[self.names[cls]] = {"IN": 0, "OUT": 0}

    def display_counts(self, plot_im):
        labels = {
            str.capitalize(cls): f"{'IN ' + str(val['IN']) if self.show_in else ''} {'OUT ' + str(val['OUT']) if self.show_out else ''}".strip()
            for cls, val in self.classwise_counts.items()
            if val['IN'] or val['OUT']
        }
        if labels:
            self.annotator.display_analytics(plot_im, labels, (104, 31, 17), (255, 255, 255), self.margin)

    def process(self, im0):
        if not self.region_initialized:
            self.initialize_region()
            self.region_initialized = True

        self.extract_tracks(im0)
        self.annotator = SolutionAnnotator(im0, line_width=self.line_width)
        self.annotator.draw_region(self.region, color=(104, 0, 123), thickness=self.line_width * 2)

        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            if self.names[cls].lower() == "person":
                self.annotator.box_label(box, label=self.names[cls], color=colors(cls, True))
                self.store_tracking_history(track_id, box)
                self.store_classwise_counts(cls)

                current_centroid = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                prev_position = self.track_history[track_id][-2] if len(self.track_history[track_id]) > 1 else None

                self.count_objects(current_centroid, track_id, prev_position, cls, im0, box)

        plot_im = self.annotator.result()
        self.display_counts(plot_im)
        self.display_output(plot_im)

        return SolutionResults(
            plot_im=plot_im,
            in_count=self.in_count,
            out_count=self.out_count,
            classwise_count=self.classwise_counts,
            total_tracks=len(self.track_ids),
        )
