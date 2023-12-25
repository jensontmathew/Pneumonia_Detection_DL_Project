import cv2
import yagmail
import numpy as np
import streamlit as st
import mysql.connector
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import yagmail.error 

# Loading the pneumonia detection model
model = load_model("/Users/jensontmathew/Documents/My_Projects/Pneumonia_detection/pneumonia.h5")

# Database connection configuration
db_config = {
    'host' : 'localhost',
    'user' : 'root',
    'password': '********',
    'database':'DL_Projects'
}

email_config = {
    'sender_email': '*********',
    'sender_password': '**********',
    'smtp_server': '*********',
    'smtp_port': '***',
    'sender_app_password':'**************'
}
# Function to predict pneumonia
def predict_pneumonia(img):
    input_shape = (224, 224)
    img = cv2.resize(img, input_shape)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0

    prediction = model.predict(img)
    pneumonia_probability = prediction[0][0]

    return pneumonia_probability

# Function to save patient information to the database
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

# Function to send email notification using yagmail
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
        st.error(f"An unexpected error occurred: {e}")

    finally:
        yag.close()

# Streamlit app
def main():
    st.title("Pneumonia Detection")
    st.sidebar.title("Upload Patient Information")
    patient_name = st.sidebar.text_input("Patient Name")
    patient_email = st.sidebar.text_input("Patient Email")
    phone_number = st.sidebar.text_input("Phone Number")
    doctor_name = st.sidebar.text_input("Doctor Name")

    uploaded_file = st.sidebar.file_uploader("Choose a chest X-ray image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None and st.sidebar.button("Predict and Save"):
        image_bytes = uploaded_file.read()
        frame = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), 1)

        pneumonia_probability = predict_pneumonia(frame)
        st.image(frame, caption='Uploaded Image', use_column_width=True)

        if pneumonia_probability > 0.5:
            prediction_text = 'Pneumonia Detected'
        else:
            prediction_text = 'Normal'
        st.write(f"Prediction: {prediction_text}")
        save_to_database(patient_name, patient_email, phone_number, doctor_name, pneumonia_probability)

if __name__ == "__main__":
    main()