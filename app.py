from flask import Flask, request, render_template
import numpy as np
import cv2
from PIL import Image
import base64
import io
import keras
from keras.models import load_model

app = Flask(__name__)

model = keras.models.load_model("cotton_weed_segmentation.keras")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'

    uploaded_file = request.files['file']

    if uploaded_file.filename == '':
        return 'No selected file'

    if uploaded_file and allowed_file(uploaded_file.filename):
        file_content = uploaded_file.read()
        # Process the file content
        # print(file_content)

        # Convert file content to numpy array
        nparr = np.frombuffer(file_content, np.uint8)

        # Read the image using cv2
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Process the image
        if image is not None:
            # Do something with the image, e.g., display dimensions
            height, width, _ = image.shape


            image = cv2.resize(image, ( 224 , 128 ))
            image = image.astype(np.float32)


            image[:,:,0] -= 103.939
            image[:,:,1] -= 116.779
            image[:,:,2] -= 123.68

            image = image[ : , : , ::-1 ]
            image = image[np.newaxis, :, :, :]


            pr = model.predict(image)[0]
            pr = pr.reshape((64, 112, 4)).argmax(axis=2)
            pr = np.array(pr, dtype='uint8')
            pr = cv2.resize(pr, (width, height))

            data = io.BytesIO()
            res = Image.fromarray(pr)
            res.save(data, "PNG")
            encoded_img_data = base64.b64encode(data.getvalue())



            # print(f"Image dimensions: {width}x{height}")
            # return render_template("index.html", img_data=encoded_img_data.decode('utf-8'))
            return encoded_img_data
        else:
            return 'Invalid image file'
    else:
        return 'Invalid file format or no file selected'
