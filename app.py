from flask import Flask, render_template, request
import keras_ocr
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load the keras-ocr pipeline
pipeline = keras_ocr.pipeline.Pipeline()

def drawBoxes(image, boxes, color=(255, 0, 0), thickness=5, boxes_format="boxes"):
    """Draw boxes onto an image."""
    if len(boxes) == 0:
        return image
    canvas = image.copy()
    if boxes_format == "lines":
        revised_boxes = []
        for line in boxes:
            for box, _ in line:
                revised_boxes.append(box)
        boxes = revised_boxes
    if boxes_format == "predictions":
        revised_boxes = []
        for _, box in boxes:
            revised_boxes.append(box)
        boxes = revised_boxes
    for box in boxes:
        cv2.polylines(
            img=canvas,
            pts=box[np.newaxis].astype("int32"),
            color=color,
            thickness=thickness,
            isClosed=True,
        )
    return canvas


def drawAnnotations(image, predictions):
    _, ax = plt.subplots()

    # Draw boxes on the image
    image_with_boxes = drawBoxes(image=image, boxes=predictions, boxes_format="predictions")

    # Display the image with boxes
    ax.imshow(image_with_boxes)

    # Sort predictions based on the y-coordinate of the boxes
    predictions = sorted(predictions, key=lambda p: p[1][:, 1].min())

    left = []
    right = []

    for word, box in predictions:
        # Determine whether the box is on the left or right side
        if box[:, 0].min() < image.shape[1] / 2:
            left.append((word, box))
        else:
            right.append((word, box))

    recognized_words = []

    for side, group in zip(["left", "right"], [left, right]):
        for index, (text, box) in enumerate(group):
            y = 1 - (index / len(group))
            xy = box[0] / np.array([image.shape[1], image.shape[0]])
            xy[1] = 1 - xy[1]

            # Annotate each text on the image
            ax.annotate(
                text=text,
                xy=xy,
                xycoords="axes fraction",
                fontsize=14,
                color="r",
                horizontalalignment="right" if side == "left" else "left",
            )

            # Collect recognized words
            recognized_words.append(text)

    return image_with_boxes, recognized_words

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded image
        image_file = request.files['image']
        if image_file:
            # Save the image to a temporary file
            temp_image_path = 'temp_image.jpg'
            image_file.save(temp_image_path)

            # Process the image using keras-ocr
            image = keras_ocr.tools.read(temp_image_path)
            predictions = pipeline.recognize([image])[0]

            # Draw annotations on the image and get recognized words
            annotated_image, recognized_words = drawAnnotations(image, predictions)

            # Encode the annotated image to base64 for displaying in HTML
            _, buffer = cv2.imencode('.png', annotated_image)
            annotated_image_base64 = base64.b64encode(buffer).decode('utf-8')

            return render_template('index.html', image=annotated_image_base64, recognized_words=recognized_words)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

