import os
import re
import cv2
import numpy as np
import pytesseract
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


class DataExtractorType(Enum):
    instagram = "instagram"
    tiktok = "tiktok"
    xhs = "xhs"


class DataExtractor:
    def __init__(self, file_path: str, type: DataExtractorType):
        self.file_path = file_path
        self.type = type
        # Key mapping for assets (images)
        self.ASSET_KEY_MAP_INSTAGRAM = {
            "instagram_play.jpeg": "views",
            "instagram_heart.png": "like",
            "instagram_comment.jpeg": "comment",
            "instagram_share.jpeg": "share",
            "instagram_bookmark.jpeg": "save"
        }
        self.ASSET_KEY_MAP_TIKTOK = {
            "tik_heart.jpeg": "heart",
            "tik_comment.jpeg": "comment",
            "tik_share.jpeg": "share",
            "tik_play.jpeg": "play",
            "tik_save.jpeg": "save"
        }
        # Key mapping for target texts
        self.TEXT_KEY_MAP_INSTAGRAM = {
            "Accounts reached": "reach",
            "Accounts engaged": "engaged",
            "Replies": "reply",
            "Sticker taps": "sticker"
        }

    def json_serializable(self, obj: Any) -> Any:
        """Convert non-serializable objects to serializable ones."""
        if isinstance(obj, np.int64):
            return int(obj)
        raise TypeError(f"Type {type(obj)} not serializable")

    def load_and_convert_image(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and convert the image to grayscale."""
        large_image = cv2.imread(self.file_path)
        if large_image is None:
            raise FileNotFoundError(
                f"Unable to load image at {self.file_path}")
        large_gray = cv2.cvtColor(large_image, cv2.COLOR_BGR2GRAY)
        return large_image, large_gray

    def find_image_locations(self, large_gray: np.ndarray, small_gray: np.ndarray, threshold: float = 0.85) -> Tuple[np.ndarray, np.ndarray]:
        """Find locations of small image in large image."""
        result = cv2.matchTemplate(
            large_gray, small_gray, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= threshold)
        return locations, result

    def process_roi(self, roi: np.ndarray) -> str:
        """Process a region of interest (ROI) to extract text."""
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_denoised = cv2.fastNlMeansDenoising(roi_gray, None, 30, 7, 21)
        text = pytesseract.image_to_string(
            roi_denoised, config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789KM.')
        text = re.sub(r'[^0-9.KM]', '', text)
        return str(int(self.convert_to_number(text))) if re.match(r'^\d+(\.\d+)?[KM]?$|^\d+(\.\d+)?$', text) else ''

    def convert_to_number(self, text: str) -> float:
        """Convert text with 'K' or 'M' suffix to number."""
        if 'K' in text:
            return float(text.replace('K', '').strip()) * 1000
        elif 'M' in text:
            return float(text.replace('M', '').strip()) * 1000000
        return float(text)

    def non_max_suppression(self, boxes: List[List[int]], overlapThresh: float = 0.3) -> Tuple[List[List[int]], List[str]]:
        """Apply non-maximum suppression to reduce overlapping boxes."""
        if not boxes:
            return [], []

        boxes_array = np.array([box[:4] for box in boxes])
        x1, y1, x2, y2 = boxes_array[:, 0], boxes_array[:,
                                                        1], boxes_array[:, 2], boxes_array[:, 3]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort([box[4] for box in boxes])
        pick = []

        while idxs.size > 0:
            i = idxs[-1]
            pick.append(i)
            xx1, yy1 = np.maximum(x1[i], x1[idxs[:-1]]
                                  ), np.maximum(y1[i], y1[idxs[:-1]])
            xx2, yy2 = np.minimum(x2[i], x2[idxs[:-1]]
                                  ), np.minimum(y2[i], y2[idxs[:-1]])
            w, h = np.maximum(0, xx2 - xx1 + 1), np.maximum(0, yy2 - yy1 + 1)
            overlap = (w * h) / area[idxs[:-1]]
            idxs = np.delete(idxs, np.concatenate(
                ([len(idxs) - 1], np.where(overlap > overlapThresh)[0])))

        return [boxes[i] for i in pick], [boxes[i][5] for i in pick]

    def process_locations(self, locations: Tuple[np.ndarray, np.ndarray], result: np.ndarray, small_image: np.ndarray, asset: str) -> List[List[int]]:
        """Process locations to create bounding boxes."""
        boxes = []
        if self.type == DataExtractorType.instagram:
            for pt in zip(*locations[::-1]):
                x_start, y_start = pt[0] - 15, pt[1] + small_image.shape[0] + 2
                x_end, y_end = pt[0] + small_image.shape[1] + \
                    15, pt[1] + small_image.shape[0] + 35
                boxes.append([x_start, y_start, x_end, y_end,
                              result[pt[1], pt[0]], asset])
            return boxes
        elif self.type == DataExtractorType.tiktok:
            for pt in zip(*locations[::-1]):
                x_start, y_start = pt[0] - 15, pt[1] + \
                    small_image.shape[0] + 10
                x_end, y_end = pt[0] + small_image.shape[1] + \
                    15, pt[1] + small_image.shape[0] + 40
                boxes.append([x_start, y_start, x_end, y_end,
                              result[pt[1], pt[0]], asset])
            return boxes

    def process_detected_boxes(self, large_image: np.ndarray, boxes: List[List[int]], asset_names: List[str]) -> Dict[str, Any]:
        """Process detected boxes to extract text and draw rectangles."""
        image_result = {"img_name": os.path.basename(
            self.file_path), "objects": []}

        for idx, box in enumerate(boxes):
            x_start, y_start, x_end, y_end = box[:4]
            asset_name = asset_names[idx]
            roi = large_image[y_start:y_end, x_start:x_end]
            text = self.process_roi(roi)
            key = self.ASSET_KEY_MAP_INSTAGRAM.get(asset_name, asset_name)
            image_result["objects"].append({key: text})
            cv2.rectangle(large_image, (x_start, y_start),
                          (x_end, y_end), (255, 0, 0), 2)
            # cv2.imshow('Detected Text', large_image)
            # cv2.waitKey(0)
        return image_result

    def detect_text_by_image(self, large_image: np.ndarray, large_gray: np.ndarray) -> Dict[str, Any]:
        """Detect text based on template images."""
        assets_folder = "assets"
        if self.type == DataExtractorType.instagram:
            assets_images = [f for f in os.listdir(
                assets_folder) if f.endswith(('.png', '.jpg', '.jpeg')) and f.startswith('instagram')]
        elif self.type == DataExtractorType.tiktok:
            assets_images = [f for f in os.listdir(
                assets_folder) if f.endswith(('.png', '.jpg', '.jpeg')) and f.startswith('tik')]
        boxes = []

        for asset in assets_images:
            small_image = cv2.imread(os.path.join(assets_folder, asset))
            if small_image is None:
                logging.warning(f"Unable to load asset: {asset}")
                continue
            small_gray = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
            locations, result = self.find_image_locations(
                large_gray, small_gray)
            boxes.extend(self.process_locations(
                locations, result, small_image, asset))

        boxes, asset_names = self.non_max_suppression(boxes, overlapThresh=0.5)
        return self.process_detected_boxes(large_image, boxes, asset_names)

    def detect_text_by_text(self, large_gray: np.ndarray) -> Dict[str, int]:
        """Detect text directly from the image using Tesseract."""
        target_text = self.TEXT_KEY_MAP_INSTAGRAM.keys()
        data = pytesseract.image_to_data(
            large_gray, output_type=pytesseract.Output.DICT, config='--psm 6')
        n_boxes = len(data['text'])

        for i in range(n_boxes):
            if data['text'][i].strip():
                data['text'][i] = re.sub(r'[^a-zA-Z0-9,]', '', data['text'][i])

        text_results = {}
        for target in target_text:
            target_words = target.lower().split()
            target_len = len(target_words)

            for i in range(n_boxes - target_len + 1):
                if all(target_words[j] in data['text'][i + j].lower() for j in range(target_len)):
                    row_texts = [data['text'][j] for j in range(n_boxes) if data['text'][j].strip(
                    ) and abs(data['top'][j] - data['top'][i]) < 10]
                    final = " ".join(row_texts)
                    value = int(re.search(r'\d+', final).group()
                                ) if re.search(r'\d+', final) else None
                    if value is not None:
                        key = self.TEXT_KEY_MAP_INSTAGRAM.get(
                            target, target.lower())
                        text_results[key] = value

        return text_results

    def save_results_to_json(self, results: Dict[str, Any], output_file: str = 'result.json'):
        """Save results to a JSON file."""
        with open(output_file, 'w') as json_file:
            json.dump(results, json_file, indent=4,
                      default=self.json_serializable)
        logging.info(f"Results saved to '{output_file}'")

    def extract(self):
        """Main function to extract data from the image."""
        large_image, large_gray = self.load_and_convert_image()
        image_result = self.detect_text_by_image(large_image, large_gray)
        text_results = self.detect_text_by_text(large_gray)
        image_result["objects"].extend(
            [{k: v} for k, v in text_results.items()])
        self.save_results_to_json(image_result)
