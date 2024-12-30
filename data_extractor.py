import os
import re
import sys
import cv2
import numpy as np
import pytesseract
import json

class DataExtractor:
    def __init__(self, file_path):
        self.file_path = file_path

    def json_serializable(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        raise TypeError(f"Type {type(obj)} not serializable")

    def find_image_locations(self, large_gray, small_gray, threshold=0.85):
        result = cv2.matchTemplate(large_gray, small_gray, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= threshold)
        return locations, result

    def convert_to_number(self, text):
        if 'K' in text:
            return float(text.replace('K', '').strip()) * 1000
        elif 'M' in text:
            return float(text.replace('M', '').strip()) * 1000000
        else:
            return float(text)

    def extract_text_from_roi(self, roi):
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_denoised = cv2.fastNlMeansDenoising(roi_gray, None, 30, 7, 21)

        text = pytesseract.image_to_string(roi_denoised,
                                           config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789KM.')

        text = re.sub(r'[^0-9.KM]', '', text)

        if re.match(r'^\d+(\.\d+)?[KM]?$|^\d+(\.\d+)?$', text):
            return int(self.convert_to_number(text))
        return ''

    def non_max_suppression(self, boxes, overlapThresh=0.3):
        if len(boxes) == 0:
            return [], []

        boxes_array = np.array([box[:4] for box in boxes])
        x1 = boxes_array[:, 0]
        y1 = boxes_array[:, 1]
        x2 = boxes_array[:, 2]
        y2 = boxes_array[:, 3]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort([box[4] for box in boxes])
        pick = []
        while len(idxs) > 0:
            i = idxs[-1]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:-1]])
            yy1 = np.maximum(y1[i], y1[idxs[:-1]])
            xx2 = np.minimum(x2[i], x2[idxs[:-1]])
            yy2 = np.minimum(y2[i], y2[idxs[:-1]])
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            overlap = (w * h) / area[idxs[:-1]]
            idxs = np.delete(idxs, np.concatenate(([len(idxs) - 1], np.where(overlap > overlapThresh)[0])))
        return [boxes[i] for i in pick], [boxes[i][5] for i in pick]

    def extract_number(self, text):
        match = re.search(r'\d+', text)
        return int(match.group()) if match else None


    def extract(self):

        x_start = 0
        y_start = 0
        x_end = 0
        y_end = 0
        value = None

        assets_folder = "assets"
        imgs_folder = "imgs"
        assets_images = [f for f in os.listdir(assets_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        img = self.file_path

        result_data = []
        large_image = cv2.imread(os.path.join(self.file_path))
        large_gray = cv2.cvtColor(large_image, cv2.COLOR_BGR2GRAY)

        # Detect text by image
        boxes = []
        for asset in assets_images:
            small_image = cv2.imread(os.path.join(assets_folder, asset))
            small_gray = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
            locations, result = self.find_image_locations(large_gray, small_gray)

            for pt in zip(*locations[::-1]):
                x_start = pt[0] - 15
                y_start = pt[1] + small_image.shape[0] + 2
                x_end = pt[0] + small_image.shape[1] + 15
                y_end = pt[1] + small_image.shape[0] + 35
                boxes.append([x_start, y_start, x_end, y_end, result[pt[1], pt[0]], asset])

        boxes, asset_names = self.non_max_suppression(boxes, overlapThresh=0.5)

        image_result = {
            "img_name": img,
            "objects": []
        }

        for idx, box in enumerate(boxes):
            x_start, y_start, x_end, y_end = box[:4]
            asset_name = asset_names[idx]
            roi = large_image[y_start:y_end, x_start:x_end]

            text = self.extract_text_from_roi(roi)

            if asset_name == "play.jpeg":
                key = "views"
            elif asset_name == "heart.png":
                key = "like"
            elif asset_name == "comment.jpeg":
                key = "comment"
            elif asset_name == "share.jpeg":
                key = "share"
            elif asset_name == "bookmark.jpeg":
                key = "save"

            image_result["objects"].append({
                key: text
            })

        cv2.rectangle(large_image, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)

        result_data.append(image_result)

        # Detect text by text
        target_text = ["Accounts reached", "Accounts engaged", "Likes", "Replies", "Shares", "Sticker taps"]
        data = pytesseract.image_to_data(large_gray, output_type=pytesseract.Output.DICT, config='--psm 6')
        n_boxes = len(data['text'])

        for i in range(n_boxes):
            if data['text'][i].strip():
                data['text'][i] = re.sub(r'[^a-zA-Z0-9,]', '', data['text'][i])

        for target in target_text:
            target_words = target.lower().split()
            target_len = len(target_words)

            for i in range(n_boxes - target_len + 1):
                text_1 = data['text'][i].lower()

                found = True
                for j in range(target_len):

                    if target_words[j] not in data['text'][i + j].lower():
                        found = False
                        break

                if found:
                    x_start = data['left'][i]
                    y_start = data['top'][i]
                    row_texts = []

                    for j in range(n_boxes):
                        if data['text'][j].strip() == "":
                            continue

                        if abs(data['top'][j] - y_start) < 10:
                            row_texts.append(data['text'][j])

                    final = " ".join(row_texts)
                    value = self.extract_number(final)

            if value is not None:
                if target == "Accounts reached":
                    key2 = "reach"
                elif target == "Accounts engaged":
                    key2 = "engaged"
                elif target == "Replies":
                    key2 = "reply"
                elif target == "Sticker taps":
                    key2 = "sticker"

                image_result["objects"].append({
                    key2: value
                })

        with open('result.json', 'w') as json_file:
            json.dump(result_data, json_file, indent=4, default=self.json_serializable)
            print("Results saved to 'result.json'")