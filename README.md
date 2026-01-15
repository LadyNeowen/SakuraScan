# SakuraScan
Mildew detection program that allows the user to scan a cherry leaf that quickly can tell if the leaf is infested or not.


"""
Boilerplate documentation for README, Business requirements and Dashboard expectations
"""

# What are the business requirements?
 The client is interested in conducting a study to visually differentiate a cherry leaf that is healthy from one that contains powdery mildew.
 The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.
# Is there any business requirement that can be answered with conventional data analysis?
 Yes, we can use conventional data analysis to conduct a study to visually differentiate a cherry leaf that is healthy from one that contains powdery mildew.
# Does the client need a dashboard or an API endpoint?
The client needs a dashboard.
# What does the client consider as a successful project outcome?
A study showing how to visually differentiate a cherry leaf that is healthy from one that contains powdery mildew.
Also, the capability to predict if a cherry leaf is healthy or contains powdery mildew.
# Can you break down the project into Epics and User Stories?
Information gathering and data collection.
Data visualization, cleaning, and preparation.
Model training, optimization and validation.
Dashboard planning, designing, and development.
Dashboard deployment and release.
# Ethical or Privacy concerns?
The client provided the data under an NDA (non-disclosure agreement), therefore the data should only be shared with professionals that are officially involved in the project.
# Does the data suggest a particular model?
The data suggests a binary classifier, indicating whether a particular cherry leaf is healthy or contains powdery mildew.
# What are the model's inputs and intended outputs?
The input is a cherry leaf image and the output is a prediction of whether the cherry leaf is healthy or contains powdery mildew.
What are the criteria for the performance goal of the predictions?
We agreed with the client a degree of 97% accuracy.
# How will the client benefit?
The client will not supply the market with a product of compromised quality.

# Business Requirement 1
Your study should include at least analysis on:
average images and variability images for each class (healthy or powdery mildew),
the differences between average healthy and average powdery mildew cherry leaves,
an image montage for each class.

# Business Requirement 2
You may deliver an ML system that is capable of predicting whether a cherry leaf is healthy or contains powdery mildew. In this case, we suggest to use Neural Networks to map the relationships between the features and the labels.
You will notice when exploring the dataset that the images are 256 pixels × 256 pixels. When defining your image shape to load the images to memory for training the model, you may choose 256 × 256 as your image shape. However, that will lead to a trained model that will likely be larger than 100Mb. This is fine as long as the model meets the project requirement, the caveat is that you may need to use Git LFS (Large File Storage) to push files larger than 100Mb to GitHub. As a result, we suggest you consider using an image shape that is smaller, like 100 × 100 or 50 × 50, with the expectation that the model would still meet the performance requirement and will be smaller than 100Mb for a smoother push to GitHub.

# Dashboard Expectations
Your dashboard should contain:

A project summary page, showing the project dataset summary and the client's requirements.
A page listing your findings related to a study to visually differentiate a cherry leaf that is healthy from one that contains powdery mildew
A page containing:
A link to download a set of cherry leaf images for live prediction (you may use the Kaggle repository that was provided to you).
A User Interface with a file uploader widget. The user should have the capacity to upload multiple images. For each image, it will display the image and a prediction statement, indicating if a cherry leaf is healthy or contains powdery mildew and the probability associated with this statement.
A table with the image name and prediction results, and a download button to download the table.
A page indicating your project hypothesis and how you validated it across the project.
A technical page displaying your model performance.




Dataset build
https://www.geeksforgeeks.org/data-science/how-to-create-a-dataset/?utm_source=chatgpt.com

https://unidata.pro/blog/how-to-prepare-ml-dataset/?utm_source=chatgpt.com

https://github.com/karan3691/dataset-builder/blob/master/dataset_builder/dataset/dataset_builder.py

https://stackoverflow.com/questions/56848253/scikit-learn-loading-images-from-folder-to-create-a-labelled-dataset-for-knn-cl?utm_source=chatgpt.com


Jupyter notebook workflow for image classification with dataset preparation and training

https://github.com/aws-samples/aws-sar-sagemaker-image-classification/blob/master/jupyter-notebook/image-classification-example.ipynb?utm_source=chatgpt.com


Modelling training with PyTorch 

https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

https://docs.pytorch.org/docs/stable/index.html

https://docs.pytorch.org/vision/stable/models.html

https://github.com/pytorch/examples/tree/main/imagenet

