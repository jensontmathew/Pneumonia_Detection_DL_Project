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
    'host': 'localhost',
    'user': 'root',
    'password': 'Password@123',
    'database': 'DL_Projects'
}

email_config = {
    'sender_email': 'jensontmathew020@gmail.com',
    'sender_password': 'jensonjenson',
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': '587',
    'sender_app_password':'whwo hiiw cwhj ttpa'
}

# Function to predict pneumonia with a confidence threshold
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

###

def is_chest_xray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    if lines is not None:
        return True  # It's a chest X-ray
    else:
        return False  # Not a chest X-ray
####

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

        if is_chest_xray(frame):
            pneumonia_probability, prediction_text = predict_pneumonia(frame)
            st.image(frame, caption='Uploaded Image', use_column_width=True)

            st.write(f"Prediction Probability: {pneumonia_probability}")
            st.write(f"Prediction: {prediction_text}")

            save_to_database(patient_name, patient_email, phone_number, doctor_name, pneumonia_probability)
        else:
            st.write("Not an X-ray image")

if __name__ == "__main__":
    main()

