<<<<<<< HEAD
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from PyPDF2 import PdfReader
from pathlib import Path
from datetime import datetime

from  PIL import Image

image=Image.open(r"D:\Project\Internship\Header1.jpg")

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


# Initialize history storage
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Set up Streamlit app
st.set_page_config(page_title="Plant Disease Detection System", layout="wide")

# Navigation Sidebar
st.sidebar.title("Navigation")
mode = st.sidebar.radio("Select", ["Home", "Detection", "History", "About"])

# Home mode
if mode == "Home":
    st.title("Plant Disease Detection System")
    st.image(image,width=400)
    st.markdown(
        """
        ### What is Plant Disease Detection System?
        The Plant Disease Detection System is an advanced tool designed to identify diseases in plants 
        by analyzing images of plant leaves. By leveraging machine learning and computer vision techniques, 
        the system can quickly and accurately diagnose plant health issues, enabling timely intervention.

        ### Why is it Required?
        - **Early Detection**: Identifying diseases early can save crops and prevent widespread damage.
        - **Sustainable Agriculture**: Reducing the use of harmful chemicals by targeting specific issues.
        - **Improved Yield**: Healthy plants lead to better productivity and profitability for farmers.
        - **Global Food Security**: Minimizing crop losses helps to ensure a stable food supply.

        This system supports detection for the following plants:

        - Apple
        - Blueberry
        - Cherry
        - Corn (Maize)
        - Orange
        - Peach
        - Pepper (Bell)
        - Potato
        - Raspberry
        - Soybean
        - Squash
        - Strawberry
        - Tomato

        """
    )
    st.warning("**Note:** This system is trained on specific data and may not recognize all plants or diseases.")

# Detection mode
elif mode == "Detection":
    st.title("Plant Disease Detection System at Your Service")
    st.warning("Please upload image of following plants (Apple,Blueberry,Cherry,Corn_(maize),Orange,Peach,Pepper,_bell,Potato,Raspberry,Soybean,  Squash ,Strawberry,Tomato)")

    cap_mode=st.radio("Select Input",["Upload Photo","Take Photo"])

    if cap_mode=='Take Photo':
        st.header("Capture Image Using Your Camera")
        captured_image = st.camera_input("Take a photo")

        if captured_image is not None:
        
            st.image(captured_image, caption="Captured Image", width=300)

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
            st.image(test_image,width=200)

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
        
        image = test_image or captured_image
        # Save history
        st.session_state['history'].append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image": image,
            "prediction":class_name[result]
        })

# History mode
elif mode == "History":
    st.title("Detection History")

    if not st.session_state['history']:
        st.info("No history available.")
    else:
        for record in st.session_state['history']:
            st.markdown(f"**Timestamp:** {record['timestamp']}")
            st.image(record['image'],  width=100)
            st.write(f"**Prediction:** {record['prediction']}")
            
            st.markdown("---")

# About mode
elif mode == "About":
    st.title("About Plant Disease Detection System for Sustainable Agriculture")


    st.image(image,width=600)

    try:
        # Replace 'example.pdf' with your PDF file path
        with open("Plant Disease Detection System Information.pdf", "rb") as pdf_file:
            reader = PdfReader(pdf_file)
            pdf_text = ""
            for page in reader.pages:
                pdf_text += page.extract_text() + "\n"

            # st.write(pdf_text)
    except FileNotFoundError:
        st.error("The PDF file does not exist. Please check the file path.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

    st.markdown(
        """
        
        ### Problem Statement :

        The agricultural sector faces significant challenges due to plant diseases, which can severely
        reduce crop yield and quality. Traditional methods of disease detection rely on manual
        observation by farmers or agronomists, which can be time-consuming, subjective, and prone
        to errors. Early and accurate detection of plant diseases is crucial to ensure sustainable
        agriculture and food security. The challenge is to develop an automated system that can
        efficiently and accurately detect diseases in plants, minimizing crop losses and reducing the
        excessive use of pesticides that harm the environment.

        So we need to develop CNN based Machine Learning model which can take image of leaf
        process on it as per requirement and give output as healthy or unhealthy with disease name
        and condition. So we can develop a sustainable agriculture system.
        
        The system will enable early detection, promote sustainable farming practices by reducing
        pesticide overuse, and support real-time decision-making for crop management.

        ### Proposed Problem Solution :

        A Plant Disease Detection System leveraging computer vision and machine learning can
        automate the detection process by analyzing images of plant leaves to identify potential
        diseases. The proposed solution involves the following components:

        1. Image Acquisition: Use cameras or mobile devices to capture images of plant leaves
        in real-time.
        
        2. Preprocessing: Enhance and normalize the images to standard formats, improving the
        quality of the input for analysis.
        
        3. Machine Learning Model: Utilize a convolutional neural network (CNN) trained on
        a labeled dataset of healthy and diseased plant leaves to classify and diagnose plant
        conditions.
        
        4. User Interface: Develop a user-friendly web or mobile application for farmers to
        upload images and receive instant feedback on plant health and recommended actions.
        
        5. Continuous Improvement: Use feedback loops where farmers validate results to
        continuously improve the model through retraining.

        ### Technolgies used :
        ##### CNN:
        • A Convolutional Neural Network (CNN) is a specialized deep learning model
        designed to process and analyze structured grid data, such as images. Its architecture
        mimics the way the human brain processes visual information, making it
        exceptionally effective for tasks like image classification, object detection, and
        segmentation.

        ##### Why Use CNNs for Plant Disease Detection?
        1. Automated Feature Extraction : CNNs automatically learn hierarchical features directly from images (e.g., spots,
        patterns, discoloration) without requiring manual feature engineering.
        
        2. Handles Complex Data : Can process large-scale image datasets with varying backgrounds, lighting conditions,
        and perspectives.
        3. Spatial Awareness : Captures spatial relationships in images, ensuring that patterns in specific regions of a
        leaf are identified.
        4. High Accuracy : CNNs achieve state-of-the-art results in image-based tasks, outperforming traditional
        machine learning methods.
        5. Scalability : Models can be retrained or fine-tuned for different crops, diseases, or new datasets,
        ensuring long-term usability.
        6. Pretrained Models (Transfer Learning) : Leverage pretrained CNNs like ResNet, VGG, or Inception to save time and
        resources, as these models already understand general image features.

        ##### Reuirements :
        ###### 1. Minimum Hardware Requirements
        ###### Development :
        • Processor: Dual-core CPU (e.g., Intel i5 or AMD Ryzen 3).\n
        • RAM: 8 GB.\n
        • Storage:\n
        o SSD: At least 128 GB (preferably SSD for faster operations).\n
        o HDD: 500 GB for dataset storage.\n
        • GPU (Optional): NVIDIA GTX 1050 or a similar entry-level GPU (if available, for
        faster training).\n
        • Camera: A smartphone camera with a resolution of 8 MP or higher for capturing
        images.\n
        
        ###### 2. Minimum Software Requirements
        ###### Operating System\n
        • Windows 10 (64-bit), macOS, or Linux (Ubuntu 18.04 or newer).
        Programming Environment\n
        • Programming Language: Python 3.8 or newer.\n
        • IDE/Text Editor: Jupyter Notebook, VS Code.\n
        Libraries and Frameworks\n
        • **Machine Learning Framework**: TensorFlow Lite or PyTorch Mobile (lightweight
        versions of TensorFlow/PyTorch).\n
        • **Database**: \n
        **Kaggle :** Data Set Link https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
        ##### Libraries Used:
        - **numpy**: For numerical operations
        - **pandas**: For data manipulation
        - **matplotlib**: For data visualization
        - **pathlib**: For filesystem operations
        - **os**: For interacting with the operating system
        - **glob**: For file pattern matching
        - **opencv**: For image processing
        - **tensorflow**: For deep learning model
        - **seaborn**: For advanced data visualization

        ##### Work Flow :
        ###### Data Preparation:
        • **Dataset Loading:** The datasets, consisting of leaf images from different plants, are
        loaded into the environment for training and testing purposes. The images are likely
        stored in separate directories for healthy and diseased leaves.\n
        • **Class Identification:** The dataset includes multiple classes corresponding to healthy
        and different types of plant diseases, allowing the model to distinguish between them.
        ###### Model Training:
        • **CNN Model:** A Convolutional Neural Network (CNN) model is trained on the
        labeled leaf images to learn visual patterns associated with healthy and diseased
        conditions.
        ##### Prediction Function:
        • **Image Preprocessing:** The prediction function preprocesses input images by resizing
        them to a fixed size (e.g., 224x224 pixels) and reshaping them to be compatible with
        the CNN model.\n
        • **Model Prediction:** The trained model predicts the disease type or identifies the leaf
        as healthy based on the extracted features.\n
        • **Output:** The predicted disease type or healthy condition is displayed, helping to
        assess the health of the plant.\n
        ###### Evaluation:
        • **Testing:** The model's performance is evaluated on a separate testing dataset to
        determine its ability to generalize to unseen data, ensuring it can accurately classify
        plant diseases in realworld scenarios.

        
        
        By leveraging these technologies, we aim to promote sustainable agriculture and reduce 
        the impact of plant diseases on global food production.
        """
    )
=======
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
>>>>>>> 7e1d3ac9309a3533a4093e347cc2773447cb2204
