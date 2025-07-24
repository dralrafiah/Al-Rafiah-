import streamlit as st
from PIL import Image
import base64
import os
import numpy as np
from reports.generate_breast_report import generate_breast_report
from reports.generate_lungColon_report import generate_vit_report
from modules.lung_model.vit_cancer_detector import predict_vit as predict_vit_model
from modules.breast_model.breast_predictor import predict_breast_model as predict_breast

# Page setup (MUST BE FIRST)
st.set_page_config(page_title="Al-Rafiah Medical Platform", page_icon="🧠", layout="centered")

# Load CSS
def load_local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file {file_name} not found. Using default styling.")

load_local_css("alrafiah_custom.css")

# Load logo
def get_base64_of_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode()
        return encoded
    except FileNotFoundError:
        st.warning(f"Logo file {image_path} not found.")
        return ""

logo_path = "assets/alrafiah_logo.png"
logo_base64 = get_base64_of_image(logo_path)

# Navigation bar
if logo_base64:
    st.markdown(f'''
    <div class="navbar">
        <img class="logo" src="data:image/png;base64,{logo_base64}">
        <div>
            <a href="https://al-rafiah.com/" target="_blank">Home</a>
            <a href="#ai-service">AI Service</a>
        </div>
    </div>
    ''', unsafe_allow_html=True)

# Header
st.markdown("<h1 id='home' style='text-align: center; color: #134734;'>Al-Rafiah Medical Platform</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Smart Cancer Detection Service</h4>", unsafe_allow_html=True)
st.markdown("---")

# Instructions
with st.expander("📌 Instructions"):
    st.write("""
    Before using the AI service, please make sure to:
    1. Select the organ type (Lung/Colon or Breast)
    2. Select the analysis type (CT or Histology)
    3. Upload a medical image in JPG format only
    """)

# Organ selection
st.markdown("### 🧬 Select the organ to analyze:")
col1, col2 = st.columns(2)
if col1.button("Lung and Colon"):
    st.session_state["model_choice"] = "Lung and Colon"
if col2.button("Breast"):
    st.session_state["model_choice"] = "Breast"

selected = st.session_state.get("model_choice")
if selected:
    st.markdown(f'<div id="model-{selected}" style="display:none;"></div>', unsafe_allow_html=True)
    st.info(f"Selected: {selected.capitalize()} model")

# Analysis type selection
if "model_choice" in st.session_state:
    st.markdown("### 🧪 Select the type of analysis:")
    analysis_type = st.radio("Choose one:", ["CT", "Histology"], horizontal=True)
    st.session_state["analysis_type"] = analysis_type

# Image upload and analysis
uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
if uploaded_image:
    if not selected or "analysis_type" not in st.session_state:
        st.warning("⚠️ Please select both the organ and the type of analysis before proceeding.")
    else:
        st.image(uploaded_image, caption="Uploaded Image")

        model_choice = st.session_state.get("model_choice")
        analysis_type = st.session_state.get("analysis_type")

        if model_choice == "Breast" and analysis_type == "CT":
            st.error("⚠️ CT analysis is not available for breast cancer. Please select Histology analysis.")

        elif st.button("🔍 Analyze"):
            try:
                image = Image.open(uploaded_image).convert("RGB")

                if model_choice == "Breast":
                    st.info("Running Breast Histology Model...")

                    img_resized = image.resize((128, 128))
                    img_array = np.array(img_resized) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    result, subtype, confidence = predict_breast(img_array)

                    if result == "Error":
                        st.error(f"Analysis failed: {subtype}")
                    else:
                        # Display results with enhanced information
                        st.success(f"✔ **Primary Classification**: {result}")
                        st.info(f"🔬 **Specific Subtype**: {subtype}")
                        st.metric("🎯 Confidence Score", f"{confidence * 100:.1f}%")
                        
                        # Add confidence interpretation
                        if confidence >= 0.95:
                            st.success("🟢 **Very High Confidence** - Strong indication based on histopathological features")
                        elif confidence >= 0.85:
                            st.info("🔵 **High Confidence** - Good confidence in classification result")
                        elif confidence >= 0.70:
                            st.warning("🟡 **Moderate Confidence** - Clinical correlation advised")
                        else:
                            st.error("🔴 **Low Confidence** - Expert review recommended")

                        try:
                            # Generate PDF report using BREAST report generator
                            pdf_path = generate_breast_report(
                                analysis_type=analysis_type,
                                result=f"{result} ({subtype})",
                                confidence=confidence * 100,
                                image_name=uploaded_image.name,
                                output_path="reports/Alrafiah_AI_Report_Breast.pdf"
                            )

                            with open(pdf_path, "rb") as file:
                                st.download_button(
                                    "📄 Download Breast Report", 
                                    file.read(), 
                                    file_name="Alrafiah_AI_Report_Breast.pdf", 
                                    mime="application/pdf"
                                )
                                
                        except Exception as pdf_error:
                            st.warning(f"Report generation failed: {pdf_error}")

                else:
                    try:
                        result = predict_vit_model(image, tissue_type="colon")
                    except Exception as lung_error:
                        st.error(f"Lung/Colon model error: {lung_error}")
                        result = None

                    if result and result["predicted_class"] != "Error":
                        predicted_class = result["predicted_class"]
                        confidence = result["confidence"]
                        
                        # Display using SAME FORMAT as breast model, but with ViT's actual data
                        st.success(f"✔ **Primary Classification**: {predicted_class}")
                        st.info(f"🔬 **Model Prediction**: {predicted_class}")
                        st.metric("🎯 Confidence Score", f"{confidence:.1f}%")
                        
                        # Add confidence interpretation (same logic as breast)
                        if confidence >= 95:
                            st.success("🟢 **Very High Confidence**")
                        elif confidence >= 85:
                            st.info("🔵 **High Confidence**")
                        elif confidence >= 70:
                            st.warning("🟡 **Moderate Confidence**")
                        else:
                            st.error("🔴 **Low Confidence**")

                        try:
                            # Generate PDF report using VIT report generator
                            pdf_path = generate_vit_report(
                                analysis_type=analysis_type,
                                result=predicted_class,  # Raw ViT output like "colon_aca"
                                confidence=confidence,
                                image_name=uploaded_image.name,
                                output_path="reports/Alrafiah_AI_Report_LungColon.pdf"
                            )

                            with open(pdf_path, "rb") as file:
                                st.download_button(
                                    "📄 Download Lung/Colon Report", 
                                    file.read(), 
                                    file_name="Alrafiah_AI_Report_LungColon.pdf", 
                                    mime="application/pdf"
                                )
                        except Exception as pdf_error:
                            st.warning(f"Report generation failed: {pdf_error}")
                    else:
                        error_msg = result.get("error", "Unknown error") if result else "Model failed to load"
                        st.error("Analysis failed. Reason: " + error_msg)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>© 2025 Al-Rafiah | AI-Powered Medical Analysis Platform</div>", unsafe_allow_html=True)