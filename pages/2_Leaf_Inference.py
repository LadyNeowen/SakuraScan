from pathlib import Path
import torch
from PIL import Image
import streamlit as st
from app_pages.src.inference import load_model, predict_image
import pandas as pd
import altair as alt

"""
Streamlit page for running inference with the SakuraScan model.
"""

@st.cache_resource
def load_app_model():
    """
    Load the trained model once per session.
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / 'app_pages' / 'src' / 'models' / 'sakuramodel_resnet18.pth'
    
    model, class_names, image_size = load_model(model_path, device)
    return model, class_names, image_size, device


def run() -> None:
    """
    Streamlit UI for leaf health prediction
    """
    st.title('SakuraScan - Leaf Health Prediction')
    st.info("""
    Model Performance Summary:
    
    The current model achieved approximately 95% validation accuracy on the test dataset.
    This indicates strong performance in classifying cherry leaf health conditions.
    However, real-world performance may vary based on image quality and environmental factors.        
    """)
    st.write('Upload an image of a cherry leaf to classify it.')
    
    # Confidence threshold slider
    confidence_threshold = st.slider(
        'Confidence Threshold',
        min_value=0.50,
        max_value=1.0,
        value=0.80,
        step=0.01
    )
    
    st.session_state.setdefault('last_pred', None)
    st.session_state.setdefault('last_conf', None)
    st.session_state.setdefault('last_image_name', None)
    
    model, class_names, image_size, device = load_app_model()
    file = st.file_uploader('Upload leaf image', type=['jpg', 'jpeg', 'png'])
    
    if file is not None:
        image = Image.open(file).convert('RGB')
        st.image(image, caption='Uploaded image', use_column_width=True)

        # Reset prediction if a new image is uploaded
        if st.session_state.get('last_image_name') != file.name:
            st.session_state['last_pred'] = None
            st.session_state['last_conf'] = None
            st.session_state['last_image_name'] = file.name

        if st.button('Run prediction'):
            with st.spinner('Predicting...'):
                pred, conf = predict_image(model, image, class_names, device, image_size)

            st.session_state['last_pred'] = pred
            st.session_state['last_conf'] = float(conf)

            st.success('Prediction completed.')

    if st.session_state.get('last_pred') is not None and st.session_state.get('last_conf') is not None:
        pred = st.session_state['last_pred']
        conf = st.session_state['last_conf']

        st.subheader('Prediction')
        st.write(f'**Confidence:** {conf:.2%}')

        if conf >= confidence_threshold:
            st.success(f'Predicted class: **{pred}**')
        else:
            st.warning(
                'The model is not confident enough to make a prediction. '
                f'(threshold: {confidence_threshold:.0%})'
            )

        df = pd.DataFrame({
                'Metric': ['Model confidence', 'Confidence threshold'],
                'Value': [conf, confidence_threshold],
            }
        )

        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x='Metric',
                y=alt.Y('Value', scale=alt.Scale(domain=[0, 1])),
                color='Metric',
            )
        )

        st.subheader('Confidence comparison')
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info('Upload an image and run prediction to see results.')
        
        st.markdown("""
        ### How to Interpret the Prediction
        
        The confidence score respresents the model's estimates of how likely the predicted class is correct.
        
        Predictions above the selected threshold are considered reliable.
        Lower confidence values indicate uncertainty and should be interpreted with caution,
        potentially requiring manual inspection.
        """)

if __name__ == '__main__':
    run()