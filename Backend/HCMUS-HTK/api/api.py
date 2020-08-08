import numpy as np
import cv2
from flask import Flask, request, Response
import darknet
import os
from PIL import Image
from io import BytesIO
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.config["DEBUG"] = True
app.config["IMAGE_UPLOADS"] = "./upload/"


network, class_names, class_colors = darknet.load_network(
        "./yolov4.cfg",
        "./coco.data",
        "./yolov4.weights",
        batch_size=1
    )

def image_detection(image_path, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    image = darknet.draw_boxes(detections, image_resized, class_colors)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections


def image_to_byte_array(image:Image):
  imgByteArr = BytesIO()
  image.save(imgByteArr, format='JPEG')
  imgByteArr = imgByteArr.getvalue()
  return imgByteArr


@app.route('/api/predict', methods=['POST'])
def predict():
    img = request.files["image"]
    url = os.path.join(app.config["IMAGE_UPLOADS"], img.filename)
    img.save(url)

    image, detections = image_detection(
            url, network, class_names, class_colors, 0.25
            )
    cv2.imwrite(url, image)
    img_encoded=image_to_byte_array(Image.open(url))
    os.remove(url)
    return Response(response=img_encoded, status=200,mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0',use_reloader=False)

#  use_reloader=False