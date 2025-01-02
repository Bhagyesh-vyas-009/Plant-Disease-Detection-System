# Plant Disease Detection System for Sustainable Agriculture Week-1 Submission


## Plant Disease Detection System for Sustainable Agriculture

### Problem Statement

The agricultural sector faces significant challenges due to plant diseases, which can severely reduce crop yield and quality. Traditional methods of disease detection rely on manual observation by farmers or agronomists, which can be time-consuming, subjective, and prone to errors. Early and accurate detection of plant diseases is crucial to ensure sustainable agriculture and food security. The challenge is to develop an automated system that can efficiently and accurately detect diseases in plants, minimizing crop losses and reducing the excessive use of pesticides that harm the environment.

To address this, we need to develop a CNN-based Machine Learning model that can process leaf images and provide output indicating whether the plant is healthy or unhealthy, along with the disease name and condition. This system can promote sustainable agriculture by enabling early detection, reducing pesticide overuse, and supporting real-time decision-making for crop management.

---

### Proposed Problem Solution

A Plant Disease Detection System leveraging computer vision and machine learning can automate the detection process by analyzing images of plant leaves to identify potential diseases. The proposed solution involves the following components:

1. **Image Acquisition**: Use cameras or mobile devices to capture images of plant leaves in real-time.
2. **Preprocessing**: Enhance and normalize the images to standard formats, improving the quality of the input for analysis.
3. **Machine Learning Model**: Utilize a convolutional neural network (CNN) trained on a labeled dataset of healthy and diseased plant leaves to classify and diagnose plant conditions.
4. **User Interface**: Develop a user-friendly web or mobile application for farmers to upload images and receive instant feedback on plant health and recommended actions.
5. **Continuous Improvement**: Use feedback loops where farmers validate results to continuously improve the model through retraining.

---

### Technologies Used

**CNN (Convolutional Neural Network)**

CNNs are specialized deep learning models designed to process and analyze structured grid data, such as images. They mimic the way the human brain processes visual information, making them effective for tasks like image classification, object detection, and segmentation.

#### Why Use CNNs for Plant Disease Detection?

1. **Automated Feature Extraction**: Learns hierarchical features directly from images (e.g., spots, patterns, discoloration) without manual engineering.
2. **Handles Complex Data**: Processes large-scale image datasets with varying backgrounds, lighting conditions, and perspectives.
3. **Spatial Awareness**: Captures spatial relationships in images, ensuring patterns in specific regions of a leaf are identified.
4. **High Accuracy**: Achieves state-of-the-art results in image-based tasks.
5. **Scalability**: Retrainable or fine-tunable for different crops, diseases, or datasets.
6. **Pretrained Models**: Utilize pretrained models like ResNet, VGG, or Inception for faster development.

---

### Requirements

#### Minimum Hardware Requirements

- **Processor**: Dual-core CPU (e.g., Intel i5 or AMD Ryzen 3).  
- **RAM**: 8 GB.  
- **Storage**: SSD (128 GB minimum) and HDD (500 GB for dataset storage).  
- **GPU (Optional)**: NVIDIA GTX 1050 or similar.  
- **Camera**: Smartphone camera with a resolution of 8 MP or higher.  

#### Minimum Software Requirements

- **Operating System**: Windows 10 (64-bit), macOS, or Linux (Ubuntu 18.04 or newer).  
- **Programming Environment**: Python 3.8 or newer, Jupyter Notebook, VS Code.  
- **Libraries/Frameworks**: TensorFlow Lite, PyTorch Mobile.  
- **Deployment**: Streamlit.  

---

### Tools Used

- **Programming Languages**: Python.  
- **Machine Learning Frameworks**: TensorFlow for building and training CNN models.  
- **Web Framework**: Streamlit.  
- **Dataset**: Kaggle.  
  - Dataset link: [New Plant Diseases Dataset (Augmented)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseasesdataset/data?select=New+Plant+Diseases+Dataset%28Augmented%29).

---

### Workflow

1. **Data Preparation**
   - **Dataset Loading**: Load datasets containing images of healthy and diseased leaves for training and testing.
   - **Class Identification**: Include classes for healthy and various plant diseases.

2. **Model Training**
   - Train a CNN model on labeled leaf images to learn patterns associated with healthy and diseased conditions.

3. **Prediction Function**
   - **Image Preprocessing**: Resize input images to a fixed size (e.g., 224x224 pixels) and reshape them for the CNN model.
   - **Model Prediction**: Predict disease type or healthy condition based on extracted features.
   - **Output**: Display the predicted disease type or health status.

4. **Evaluation**
   - **Testing**: Evaluate the model's performance on a testing dataset to ensure it generalizes well to unseen data.


