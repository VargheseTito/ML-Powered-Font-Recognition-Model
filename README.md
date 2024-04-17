# ML-Powered-Font-Recognition-Model
Font Recognition Model using CNN

Dataset Link- https://www.kaggle.com/datasets/muhammadardiputra/font-recognition-data/data

Project Overview:

The goal of this project is to develop a font recognition model powered by machine
learning. The model should be capable of analyzing images, detecting text regions, and
identifying the fonts used within these regions. It aims to accurately recognize different
fonts from an image input, even when the image contains text in multiple fonts. The
desired output of the model is to provide the detected text alongside the identified font
for each text box within the image.

Abstract:

In alignment with the project's objectives, I've crafted an exemplary CNN deep learning
model that excels in font recognition from uploaded textual-image data within a
Streamlit application, boasting an impressive accuracy rate of 97%. Trained on a
meticulously curated dataset comprising 3000 instances for each of the four fonts –
Arial, Bodoni, Franklin_Gothic, and Times_New_Roman – the model adeptly predicts
these fonts with remarkable precision post-validation. Furthermore, I've harnessed the
power of the pytesseract library to extract crucial text information as a pivotal
component of this endeavor. Our evaluation methodology revolves around the validation
accuracy score, providing a robust metric for assessing the model's performance and
ensuring its efficacy in real-world scenarios. This breakthrough project stands as a
testament to cutting-edge innovation and signifies a monumental leap forward in font
recognition technology.

![image](https://github.com/VargheseTito/ML-Powered-Font-Recognition-Model/assets/110298267/20cf447b-ff45-450b-8828-2f237bbb799a)

Project Workflow:

1. Data Collection and Image Preprocessing:
● Load the font dataset containing images of text in various fonts.
● Preprocess the images:
● Resize the images to a standardized size.
● Crop the images to remove unnecessary borders or whitespace.
● Apply denoising techniques to reduce image noise and improve
clarity.
● Normalize the pixel values of the images to enhance model
performance.

2. Feature Engineering and Data Preparation:
   
● Extract relevant features from the preprocessed images.
● Subsample the dataset to select 3000 records from each font category
due to computational limitations.
● Initialize the Convolutional Neural Network (CNN) model:
● Configure the model architecture with 6 convolutional layers, batch
normalization, maxpooling, dropout, and ReLU activation function.
● Perform data augmentation to increase the diversity of the training data
and prevent overfitting.

3. Model Training:
● Split the data into training and validation sets to assess model
performance.
● Train the CNN model on the training set using the defined architecture and
augmented data.

4. Model Evaluation:
   
● Evaluate the trained model's performance on the validation set using
metrics such as validation accuracy.
● Adjust hyperparameters as needed based on evaluation results and retrain
the model if necessary to improve performance.

5. Model Deployment:
   
● Save the trained model for deployment in a Streamlit application, ensuring
compatibility and efficiency.
● Export necessary preprocessing steps, such as image resizing and
normalization, for real-time inference during deployment.

Conclusion:

This workflow outlines the systematic process undertaken to develop a font recognition
model, encompassing data collection, preprocessing, model initialization, training,
evaluation, and deployment stages. By carefully curating and preprocessing the font
dataset, initializing a robust CNN model architecture, and training it on subsampled data
with data augmentation, we achieved significant advancements in font recognition
accuracy. Through rigorous evaluation and optimization, we ensured the model's
reliability and effectiveness in real-world scenarios. The deployment of the trained
model in a Streamlit application facilitates user-friendly access and seamless
integration, thereby contributing to the democratization of font recognition technology.
Overall, this project underscores the importance of methodical approach and innovation
in advancing machine learning applications for practical use cases.
