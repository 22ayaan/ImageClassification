import streamlit as st
import numpy as np
import PIL
import tensorflow as tf
import pathlib
import shutil
from datetime import datetime
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

def imageCheck(model,img):
    #image_path = tf.keras.utils.get_file(origin=img)
    #image_path = pathlib.Path(img)
    img = tf.keras.utils.load_img(
        img, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    st.markdown(
        "**This image most likely belongs to {} with a {:.2f} percent confidence.**"
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

def uploadOption():
    if upload_method == "Upload Image from device":
        img = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
        
        upload_button = st.button("Upload")
        if upload_button:
            img = Image.open(img)
            
            if img.format == 'PNG':
                img = img.convert('RGB')
                img.save('image.jpg')
            st.image(img,width=250)

            date = datetime.now().strftime("%Y_%m_%d-%I-%M-%S_%p")
            directory = 'uploaded_images/'
            filepath = directory+'image_' + date + '.jpg'
            img.save(filepath)
    elif upload_method == "Enter the URL of image":
        img_url = st.text_input("Enter the URL of the image you want to test: ")
        upload_button = st.button("Upload")
        if upload_button:
            st.image(img_url,width=250)
        img_path = tf.keras.utils.get_file(origin=img_url)
        filepath = pathlib.Path(img_path)
    imageCheck(loaded_model,filepath)

st.header("Image Classifier")
st.subheader("This app will help you identify the object(s) in an image.")
upload_method = st.radio("Select upload optiom", ("Upload Image from device", "Enter the URL of image"))

#tab1, tab2 = st.tabs(['Upload Image from device', 'Enter the URL of image'])


batch_size = 32
img_height = 180
img_width = 180  
class_names = []

loaded_model = tf.keras.models.load_model('model#2')
data_dir = pathlib.Path('flower_photos')
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
    
class_names = val_ds.class_names
uploadOption()