import cv2
import numpy as np
import streamlit as st
import mysql.connector
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import yagmail.error
from tensorflow.keras.applications.vgg16 import preprocess_input

model = load_model("/Users/jensontmathew/Documents/My_Projects/Pneumonia_detection/hai.h5")
db_config = {'host': 'localhost', 'user': 'root', 'password': 'Password@123', 'database': 'DL_Projects'}
email_config = {'sender_email': 'jensontmathew020@gmail.com', 'sender_password': 'jensonjenson', 'smtp_server': 'smtp.gmail.com', 'smtp_port': '587', 'sender_app_password':'whwo hiiw cwhj ttpa'}

def predict_pneumonia(img, confidence_threshold=0.5):
   input_shape = (224, 224)
   img = cv2.resize(img, input_shape)
   img = image.img_to_array(img)
   img = np.expand_dims(img, axis=0)
   img /= 255.0
   prediction = model.predict(img)
   pneumonia_probability = prediction[0][0]
   pneumonia_result = 'Pneumonia Detected' if pneumonia_probability > confidence_threshold else 'Normal'
   return pneumonia_probability, pneumonia_result

def is_chest_xray(img):
   resized_img = cv2.resize(img, (224, 224))
   img_array = np.expand_dims(resized_img, axis=0)
   preprocessed_img = preprocess_input(img_array)
   features = vgg16.predict(preprocessed_img)  # Load VGG16 model here
   pixel_intensity_distribution = np.histogram(resized_img.ravel(), bins=256)[0]
   return pixel_intensity_distribution.mean() < 100 and pixel_intensity_distribution.std() > 30

def save_to_database(patient_name, patient_email, phone_number, doctor_name, pneumonia_probability):
   try:
       connection = mysql.connector.connect(**db_config)
       cursor = connection.cursor()
       pneumonia_result = 'Pneumonia Detected' if pneumonia_probability > 0.5 else 'Normal'
       insert_query = "INSERT INTO Patient_Records (patient_name, patient_email, phone_number, doctor_name, pneumonia_result) VALUES (%s, %s, %s, %s, %s)"
       data = (patient_name, patient_email, phone_number, doctor_name, pneumonia_result)
       cursor.execute(insert_query, data)
       connection.commit()
       st.success("Patient information saved to the database")
       send_email(patient_name, patient_email, pneumonia_result)
   except mysql.connector.Error as err:
       st.error(f"Error: {err}")
   finally:
       if connection.is_connected():
           cursor.close()
           connection.close()

def send_email(patient_name, recipient_email, pneumonia_result):
   subject = 'Pneumonia Detection Result'
   body = f"Dear {patient_name},\n\nThe pneumonia detection result is: {pneumonia_result}\n\nThank you."
   try:
       yag = yagmail.SMTP(email_config['sender_email'], email_config['sender_app_password'], host=email_config['smtp_server'])
       yag.send(to=recipient_email, subject=subject, contents=body)
       st.success("Email notification sent successfully")
   except yagmail.error.YagConnectionClosed as e:
       st.error(f"Failed to send email notification. Error: {e}")
   except Exception as e:
       st.error()
