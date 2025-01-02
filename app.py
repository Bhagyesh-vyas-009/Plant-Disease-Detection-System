import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from PyPDF2 import PdfReader

st.sidebar.title("Plant Disease Detection System For Sustainable Agriculture")
st.sidebar.write("Use this application to upload a plant leaf image and predict the disease affecting the plant.")
mode=st.sidebar.radio("Select",["Home","Prediction","Limitations"])

st.title("Plant Disease Detection System For Sustainable Agriculture")

def model_predict(image_path):
    model=tf.keras.models.load_model(r"D:\Project\Internship\CNN_plantdiseases_model.keras")
    img=cv2.imread(image_path)
    H,W,C=224,224,3
    img=cv2.resize(img,(H,W))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=np.array(img)
    img=img.astype("float32")
    img=img/255.0
    img=img.reshape(1,H,W,C)

    prediction=np.argmax(model.predict(img),axis=1)[0]

    return prediction

from  PIL import Image

image=Image.open(r"D:\Project\Internship\Header3.jpg")


st.image(image)
if mode=="Home":
    try:
        # Replace 'example.pdf' with your PDF file path
        with open("Plant Disease Detection System Information.pdf", "rb") as pdf_file:
            reader = PdfReader(pdf_file)
            pdf_text = ""
            for page in reader.pages:
                pdf_text += page.extract_text() + "\n"

            st.write(pdf_text)
    except FileNotFoundError:
        st.error("The PDF file does not exist. Please check the file path.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

elif mode=="Prediction":
    st.header("Plant Disease Detection System For Sustainable Agriculture")
    st.warning("Please upload image of following plants (Apple,Blueberry,Cherry,Corn_(maize),Orange,Peach,Pepper,_bell,Potato,Raspberry,Soybean,  Squash ,Strawberry,Tomato)")

    cap_mode=st.radio("Select Input",["Upload Photo","Take Photo"])

    if cap_mode=='Take Photo':
        st.header("Capture Image Using Your Camera")
        captured_image = st.camera_input("Take a photo")

        if captured_image is not None:
        
            st.image(captured_image, caption="Captured Image", width=4,use_column_width=True)

            
            # with st.spinner("Analyzing the image..."):
            #     time.sleep(2)  # Simulate a delay for processing
            save_path=os.path.join(os.getcwd(),captured_image.name)
            print(save_path)
            with open(save_path,"wb")as f:
                f.write(captured_image.getbuffer())         

    else:
        test_image=st.file_uploader("Choose an Image")

        if test_image is not None:
            save_path=os.path.join(os.getcwd(),test_image.name)
            print(save_path)
            with open(save_path,"wb")as f:
                f.write(test_image.getbuffer())    

        if st.button("Show Image"):
            st.image(test_image,width=4,use_column_width=True)

    if st.button('Predict'):
        st.snow()
        st.write("Our Prediction")
        result=model_predict(save_path)

        class_name=['Apple___Apple_scab',
        'Apple___Black_rot',
        'Apple___Cedar_apple_rust',
        'Apple___healthy',
        'Blueberry___healthy',
        'Cherry_(including_sour)___Powdery_mildew',
        'Cherry_(including_sour)___healthy',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
        'Corn_(maize)___Common_rust_',
        'Corn_(maize)___Northern_Leaf_Blight',
        'Corn_(maize)___healthy',
        'Grape___Black_rot',
        'Grape___Esca_(Black_Measles)',
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
        'Grape___healthy',
        'Orange___Haunglongbing_(Citrus_greening)',
        'Peach___Bacterial_spot',
        'Peach___healthy',
        'Pepper,_bell___Bacterial_spot',
        'Pepper,_bell___healthy',
        'Potato___Early_blight',
        'Potato___Late_blight',
        'Potato___healthy',
        'Raspberry___healthy',
        'Soybean___healthy',
        'Squash___Powdery_mildew',
        'Strawberry___Leaf_scorch',
        'Strawberry___healthy',
        'Tomato___Bacterial_spot',
        'Tomato___Early_blight',
        'Tomato___Late_blight',
        'Tomato___Leaf_Mold',
        'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites Two-spotted_spider_mite',
        'Tomato___Target_Spot',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy']
        
        st.success("Predicted Output {}".format(class_name[result]))