from pathlib import Path
import torch
from PIL import Image
import streamlit as st
from app_pages.src.inference import load_model, predict_image

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


def run():
    """
    Streamlit UI for leaf health prediction
    """
    st.title('SakuraScan - Leaf Health Prediction')
    st.write('Upload an image of a cherry leaf to classify it.')
    
    model, class_names, image_size, device = load_app_model()
    file = st.file_uploader('Upload leaf image', type=['jpg', 'jpeg', 'png'])
    if file:
        image = Image.open(file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if st.button('Run prediction'):
            with st.spinner('Predicting...'):
                pred, conf = predict_image(model, image, class_names, device, image_size)
                
            st.subheader('Prediction')
            st.write(f'**Class:** {pred}')
            st.write(f'**Confidence:** {conf:.2%}')
            
if __name__ == '__main__':
    run()