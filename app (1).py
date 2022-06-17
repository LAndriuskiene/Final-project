import numpy as np
import streamlit as st
import cv2 as cv
from PIL import Image
from keras.models import load_model

# Label traffic signs
labels_dict = {
    0: '200m',
    1: '50-100m',
    2: 'Ahead-Left',
    3: 'Ahead-Right',
    4: 'Axle-load-limit',
    5: 'Barrier Ahead',
    6: 'Bullock Cart Prohibited',
    7: 'Cart Prohobited',
    8: 'Cattle',
    9: 'Compulsory Ahead',
    10: 'Compulsory Keep Left',
    11: 'Compulsory Left Turn',
    12: 'Compulsory Right Turn',
    13: 'Cross Road',
    14: 'Cycle Crossing',
    15: 'Compulsory Cycle Track',
    16: 'Cycle Prohibited',
    17: 'Dangerous Dip',
    18: 'Falling Rocks',
    19: 'Ferry',
    20: 'Gap in median',
    21: 'Give way',
    22: 'Hand cart prohibited',
    23: 'Height limit',
    24: 'Horn prohibited',
    25: 'Humpy Road',
    26: 'Left hair pin bend',
    27: 'Left hand curve',
    28: 'Left Reverse Bend',
    29: 'Left turn prohibited',
    30: 'Length limit',
    31: 'Load limit 5T',
    32: 'Loose Gravel',
    33: 'Major road ahead',
    34: 'Men at work',
    35: 'Motor vehicles prohibited',
    36: 'Nrrow bridge',
    37: 'Narrow road ahead',
    38: 'Straight prohibited',
    39: 'No parking',
    40: 'No stoping',
    41: 'One way sign',
    42: 'Overtaking prohibited',
    43: 'Pedestrian crossing',
    44: 'Pedestrian prohibited',
    45: 'Restriction ends sign',
    46: 'Right hair pin bend',
    47: 'Right hand curve',
    48: 'Right Reverse Bend',
    49: 'Right turn prohibited',
    50: 'Road wideness ahead',
    51: 'Roundabout',
    52: 'School ahead',
    53: 'Side road left',
    54: 'Side road right',
    55: 'Slippery road',
    56: 'Compulsory sound horn',
    57: 'Speed limit',
    58: 'Staggred intersection',
    59: 'Steep ascent',
    60: 'Steep descent',
    61: 'Stop',
    62: 'Tonga prohibited',
    63: 'Truck prohibited',
    64: 'Compulsory turn left ahead',
    65: 'Compulsory right turn ahead',
    66: 'T-intersection',
    67: 'U-turn prohibited',
    68: 'Vehicle prohibited in both directions',
    69: 'Width limit',
    70: 'Y-intersection',
    71: 'Sign_C',
    72: 'Sign_T',
    73: 'Sign_S',
    74: 'No entry',
    75: 'Compulsory Keep Right',
    76: 'Parking'

}


@st.cache
def sign_predict(image):
    model = load_model('/content/drive/MyDrive/yolov5s_saved_model/saved_model.pb')
    image = np.array(image, dtype=np.float32)
    image = image / 255
    image = np.reshape(image, (1, 32, 32))
    x = image.astype(np.float32)
    prediction = model.predict(x)
    prediction_max = np.argmax(prediction)
    prediction_label = labels_dict[prediction_max]
    confidence = np.max(prediction)
    return prediction_label, confidence


def main():
    # Set page config and markdowns
    st.set_page_config(page_title='Traffic Signs Classifier', page_icon=':car:')
    st.title('Traffic Signs Classifier')
    st.markdown("""
        This application classifies traffic signs. Upload any photo of a traffic sign 
        and receive its name out of 43 present classes. For getting the correct prediction, 
        try to upload a square picture containing only the sign.
        """)
    with st.expander("See list of classes"):
        st.write(list(labels_dict.values()))
    st.image('/content/drive/MyDrive/Final_project/images_for_YOLO/test/images/00094.png', use_column_width=True)
    image_usr = st.file_uploader('Upload a photo of traffic sign here', type=['jpg', 'jpeg', 'png'])

    if image_usr is not None:
        col1, col2 = st.columns(2)
        col1.markdown('#### Your picture')
        col2.markdown('#### Your picture 32x32 gray')
        image = Image.open(image_usr)
        with col1:
            st.image(image, use_column_width=True)

        image_np = np.array(image.convert('RGB'))
        image_col = cv.cvtColor(image_np, 1)
        image_gray = cv.cvtColor(image_col, cv.COLOR_BGR2GRAY)
        image_32 = cv.resize(image_gray, (32, 32))
        with col2:
            st.image(image_32, use_column_width=True)

        # Make prediction
        prediction_label, confidence = sign_predict(image_32)

        st.write('##### Prediction:', prediction_label)
        st.write('##### Confidence:', str(confidence))
        st.markdown('***')

    # Markdowns
    st.subheader('About this app')
    st.markdown("""
    The app uses an implementation of LeNet-5 Convolutional Neural Network. 
    The model was trained and tested on about 40.000 real photos of 43 types of german traffic signs.

    Data was taken from The German Traffic Sign Recognition Benchmark (GTSRB):
    https://benchmark.ini.rub.de/gtsrb_dataset.html

    Source code on GitHub: https://github.com/AndriiGoz/traffic_signs_classification

    Author: Andrii Gozhulovskyi
    """)


if __name__ == '__main__':
    main()