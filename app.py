import streamlit as st
from PIL import Image
import base64
import os
import numpy as np
import torch
from reports.generate_breast_report import generate_breast_report
from reports.generate_lung_colon_report import generate_lung_colon_report
from breast_model.breast_predictor import predict_breast_model as predict_breast
from lung_model.run_lung_colon_model import load_models_from_hf, analyze_wsi

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
    1. Select the organ type (Lung, Colon, Breast)
    2. Select the analysis type (CT or Histology)
    3. Upload a medical image in JPG format. You can also upload SVS whole slide images for lung or colon cancer analysis.
    """)

# Organ selection using three separate buttons
st.markdown("### 🧬 Select the organ to analyze:")
col1, col2, col3 = st.columns(3)

if col1.button("Lung"):
    st.session_state["model_choice"] = "Lung"
if col2.button("Colon"):
    st.session_state["model_choice"] = "Colon"
if col3.button("Breast"):
    st.session_state["model_choice"] = "Breast"

selected = st.session_state.get("model_choice")
if selected:
    st.markdown(f'<div id="model-{selected}" style="display:none;"></div>', unsafe_allow_html=True)
    st.info(f"Selected: {selected} model")

# Analysis type selection
if "model_choice" in st.session_state:
    st.markdown("### 🧪 Select the type of analysis:")
    analysis_type = st.radio("Choose one:", ["CT", "Histology"], horizontal=True)
    st.session_state["analysis_type"] = analysis_type

# Load lung/colon models once and cache them
@st.cache_resource
def get_lung_colon_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_models_from_hf(device, repo_id="draziza/lung-colon-model")

lung_colon_models = None
try:
    lung_colon_models = get_lung_colon_models()
except Exception as e:
    st.warning(f"Could not preload Lung/Colon models now: {e}")


# Image upload and analysis
uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "svs"])
if uploaded_image:
    if not selected or "analysis_type" not in st.session_state:
        st.warning("⚠️ Please select both the organ and the type of analysis before proceeding.")
    else:
        # Show preview for non-WSI files
        try:
            if not uploaded_image.name.lower().endswith(".svs"):
                st.image(uploaded_image, caption="Uploaded Image")
        except Exception:
            pass
        
        model_choice = st.session_state.get("model_choice")
        analysis_type = st.session_state.get("analysis_type")

        if model_choice == "Breast" and analysis_type == "CT":
            st.error("⚠️ CT analysis is not available for breast cancer. Please select Histology analysis.")

        if uploaded_image.name.lower().endswith(".svs") and model_choice == "Breast":
            st.error("⚠️ Whole Slide Image (SVS) analysis is not available for breast cancer. Please upload a JPG/PNG image for breast cancer analysis.")

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

                # LUNG and COLON branches: histology-only logic
                else:
                    # If user selected CT in UI for Lung/Colon, inform that CT is not available yet.
                    if analysis_type == "CT":
                        st.error("CT analysis not implemented yet.")

                    else:
                        st.info(f"Running {model_choice} Histology Model...")

                        # Save uploaded file to temp for openslide/analyze_wsi
                        temp_dir = "temp"
                        os.makedirs(temp_dir, exist_ok=True)
                        temp_path = os.path.join(temp_dir, uploaded_image.name)
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_image.getbuffer())

                        # Ensure models are available
                        if lung_colon_models is None:
                            try:
                                lung_colon_models = get_lung_colon_models()
                            except Exception as load_err:
                                st.error(f"Failed to load Lung/Colon models: {load_err}")
                                lung_colon_models = None

                        else:
                            model = lung_colon_models["lung"] if model_choice == "Lung" else lung_colon_models["colon"]

                            try:
                                highlighted_img, diagnosis, confidence = analyze_wsi(
                                    temp_path, model, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                                )

                                st.image(highlighted_img, caption=f"{diagnosis} ({confidence:.2f}%)")
                                st.success(f"✔ **Primary Classification**: {diagnosis}")
                                st.metric("🎯 Cancer Presence Confidence", f"{confidence:.2f}%")

                                if confidence >= 95:
                                    st.success("🟢 **Very High Confidence**")
                                elif confidence >= 85:
                                    st.info("🔵 **High Confidence**")
                                elif confidence >= 70:
                                    st.warning("🟡 **Moderate Confidence**")
                                else:
                                    st.error("🔴 **Low Confidence**")

                                # PDF generation for lung/colon uses existing generator
                                try:
                                    pdf_path = generate_lung_colon_report(
                                        organ="Lung" if model_choice == "lung" else "Colon",
                                        analysis_type=analysis_type,
                                        diagnosis=diagnosis,
                                        confidence=confidence,
                                        highlighted_image=highlighted_img,
                                        output_path=f"reports/Alrafiah_AI_{model_choice}_Report.pdf"
                                    )
                                    
                                    if os.path.exists(pdf_path):
                                        with open(pdf_path, "rb") as file:
                                            st.download_button(
                                                "📄 Download Lung/Colon Report",
                                                file.read(),
                                                file_name=f"Alrafiah_AI_{model_choice}_Report.pdf",
                                                mime="application/pdf"
                                            )
                                    else:
                                        st.warning("Lung/Colon report generation did not produce a PDF file.")
                                except Exception as pdf_error:
                                    st.warning(f"Report generation failed: {pdf_error}")


                            except Exception as infer_err:
                                st.error(f"Histology analysis failed: {infer_err}")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>© 2025 Al-Rafiah | AI-Powered Medical Analysis Platform</div>", unsafe_allow_html=True)