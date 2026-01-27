# SakuraScan – Predictive Analytics Portfolio Project
## Project Overview

SakuraScan is a predictive analytics application developed for Farmy & Foods to support early detection of powdery mildew in cherry leaves. The project combines image-based data analysis, machine learning, and an interactive Streamlit dashboard to reduce the time and cost associated with manual crop inspection.

--------------------------------------------------------------------------------------------------------------------------------

## Business Understanding (CRISP-DM)

Cherry plantations operated by Farmy & Foods are affected by powdery mildew, a fungal disease that reduces crop quality. The current manual inspection process is time-consuming and not scalable across thousands of trees.

The goal of this project is to:

* Support visual understanding of differences between healthy and infected cherry leaves.

* Provide instant predictions using a trained ML model based on leaf images.

* Enable scalability and future reuse of the solution for other crops.

---------------------------------------------------------------------------------------------------------------------------------

## Stakeholders & Benefits

### Primary stakeholder:

* Marianne McGuiney, Head of IT & Innovation, Farmy & Foods

### Benefits:

* Significant reduction in inspection time (from ~30 minutes to seconds).

* Faster application of treatment when mildew is detected.

* Foundation for a reusable ML-based inspection system across multiple crops.

----------------------------------------------------------------------------------------------------------------------------------

## Business Requirements

* The client wants to visually differentiate healthy cherry leaves from leaves affected by powdery mildew.

* The client wants to predict whether a cherry leaf is healthy or infected using an ML system.

* The client requires an interactive dashboard that supports both analysis and prediction.

----------------------------------------------------------------------------------------------------------------------------------

## Mapping Business Requirements to ML and Data Visualisation Tasks

| Business Requirement                                  | Dashboard Page         | ML / Data Task                                   | Rationale                                                     |
|------------------------------------------------------|------------------------|--------------------------------------------------|----------------------------------------------------------------|
| Visual differentiation of healthy vs infected leaves | Explore Dataset        | Image exploration, class distribution, sampling  | Helps stakeholders understand visual patterns and data quality |
| Predict leaf health status                           | Leaf Health Prediction | Image classification using CNN (ResNet18)        | Automates disease detection and supports rapid decisions       |
| Scalable decision support                            | Full Dashboard         | Embedded ML pipeline in Streamlit application    | Demonstrates real business use of ML predictions               |

-------------------------------------------------------------------------------------------------------------------------------------

## Machine Learning Business Case

* Task: Binary image classification (Healthy vs Powdery Mildew)

* Learning method: Supervised learning using transfer learning (ResNet18)

* Input: RGB images of cherry leaves

* Output: Predicted class with confidence score

* Success criteria: High validation accuracy and reliable confidence scores suitable for decision support

* Relevance to user: Enables instant assessment of leaf health directly in the dashboard

* Training data: Image dataset provided by Farmy & Foods

* Heuristics & techniques: Image augmentation, frozen backbone, validation split, accuracy monitoring

-------------------------------------------------------------------------------------------------------------------------------------

## Data Analysis & Insights

Exploratory Data Analysis (EDA) was conducted to:

* Inspect dataset structure and class balance

* Visually examine sample images per class

* Validate image quality and variability

-----------------------------------------------------------------------------------------------------------------------------------------

## Dashboard Design
### Pages Overview

* #### Explore Dataset

* Image counts per class

* Interactive selection of sample images

* Supports the visual analysis business requirement

* #### Leaf Health Prediction

* Image upload widget

* ML-based prediction with confidence score

* Visual comparison between confidence and user-defined threshold

* Supports the predictive business requirement

----------------------------------------------------------------------------------------------------------------------------------------

## Machine Learning Pipeline

* Data loading from image folders

* Image preprocessing and augmentation

* Model training using PyTorch in Jupyter Notebook

* Model evaluation on validation data

* Model persistence and reuse in Streamlit inference pipeline

-----------------------------------------------------------------------------------------------------------------------------------------

## Deployment

The application is built using Streamlit and structured for deployment with:

* Procfile (Upcoming)

* requirements.txt

* runtime.txt (Upcoming)

* setup.sh (Upcoming)

----------------------------------------------------------------------------------------------------------------------------------------

## References and Acknowledgements

This project was developed using a combination of course materials, official documentation, and external technical resources. The following sources supported different stages of the project:

### Dataset Preparation
- GeeksforGeeks: How to Create a Dataset  
  https://www.geeksforgeeks.org/data-science/how-to-create-a-dataset/

- Unidata: How to Prepare an ML Dataset  
  https://unidata.pro/blog/how-to-prepare-ml-dataset/

- Dataset Builder (GitHub)  
  https://github.com/karan3691/dataset-builder

- Stack Overflow: Loading Images from Folders  
  https://stackoverflow.com/questions/56848253/

### Jupyter Notebook Workflow
- AWS Sample Image Classification Notebook  
  https://github.com/aws-samples/aws-sar-sagemaker-image-classification

### Model Training with PyTorch
- PyTorch CIFAR-10 Tutorial  
  https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

- PyTorch Transfer Learning Tutorial  
  https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

- PyTorch Documentation  
  https://docs.pytorch.org/docs/stable/index.html

- Torchvision Models  
  https://docs.pytorch.org/vision/stable/models.html

- PyTorch ImageNet Examples  
  https://github.com/pytorch/examples/tree/main/imagenet

### Streamlit Development
- File Uploader API  
  https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader

- Caching and State  
  https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource

- Multipage Applications  
  https://docs.streamlit.io/develop/concepts/multipage-apps

- Slider Widget  
  https://docs.streamlit.io/develop/api-reference/widgets/st.slider

----------------------------------------------------------------------------------------------------------------------------------------------

## Acknowledgements - AI
### ChatGPT for:
* Structuring and folder build
* Bug fixes and solutions
* Swedish translation and explinations
* Docstrings
* Assistance with README file
* Aid in finding the right information on the web

---------------------------------------------------------------------------------------------------------------------------------------------
