from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps #Install pillow instead of PIL
import numpy as np
from keras.models import load_model
import io
import base64
 
app = Flask(__name__)

@app.route('/')
def upload_file():
   return render_template('upload.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def uploader_file():
   if request.method == 'POST':
      f = request.files['file']
      f.filename = 'image.jpg'
      f.save(secure_filename(f.filename))
   # Disable scientific notation for clarity
   np.set_printoptions(suppress=True)
   
   #open image from path
   IMAGE_PATH = "image.jpg"

   # Load the model
   model = load_model('keras_model.h5', compile=False)

   # Load the labels
   class_names = open('labels.txt', 'r').readlines()

   # Create the array of the right shape to feed into the keras model
   # The 'length' or number of images you can put into the array is
   # determined by the first position in the shape tuple, in this case 1.
   data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

   # Replace this with the path to your image
   image = Image.open(IMAGE_PATH).convert('RGB')

   #resize the image to a 224x224 with the same strategy as in TM2:
   #resizing the image to be at least 224x224 and then cropping from the center
   size = (224, 224)
   image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

   #turn the image into a numpy array
   image_array = np.asarray(image)

   # Normalize the image
   normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

   # Load the image into the array
   data[0] = normalized_image_array

   # run the inference
   prediction = model.predict(data)
   index = np.argmax(prediction)
   class_name = class_names[index]
   confidence_score = prediction[0][index]

   print('Class:', class_name, end='')
   print('Confidence score:', confidence_score)
   
   im = Image.open("image.jpg")
   data = io.BytesIO()
   im.save(data, "JPEG")
   encoded_img_data = base64.b64encode(data.getvalue())
   return render_template('display.html', img_data=encoded_img_data.decode('utf-8'), score = confidence_score)



if __name__ == '__main__':
   app.run(debug = True)