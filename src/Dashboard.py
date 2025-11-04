import json
import requests
import streamlit as st
from pathlib import Path
from streamlit.logger import get_logger

FASTAPI_BACKEND_ENDPOINT = "http://localhost:8000"
LOGGER = get_logger(__name__)

def run():
    st.set_page_config(
        page_title="ML Model Prediction Demo",
        page_icon="🤖",
        layout="wide"
    )
    
    # Sidebar
    with st.sidebar:
        st.title("🤖 ML Predictor")
        
        # Check backend status
        try:
            backend_request = requests.get(f"{FASTAPI_BACKEND_ENDPOINT}/health")
            if backend_request.status_code == 200:
                health_data = backend_request.json()
                st.success("Backend online ✅")
                st.info(f"Models loaded: {', '.join(health_data['models_loaded'])}")
            else:
                st.warning("Problem connecting 😭")
        except requests.ConnectionError as ce:
            st.error("Backend offline 😱")
        
        st.divider()
        
        # Model selection
        model_type = st.selectbox(
            "Select Model",
            ["Iris Flower", "Wine Classification"]
        )
        
        # Input method selection
        input_method = st.radio(
            "Input Method",
            ["Manual Input", "Upload JSON"]
        )
        
        st.divider()
        
        # Input based on method
        if input_method == "Manual Input":
            if model_type == "Iris Flower":
                st.subheader("🌸 Iris Parameters")
                sepal_length = st.slider("Sepal Length", 4.3, 7.9, 5.1, 0.1)
                sepal_width = st.slider("Sepal Width", 2.0, 4.4, 3.5, 0.1)
                petal_length = st.slider("Petal Length", 1.0, 6.9, 1.4, 0.1)
                petal_width = st.slider("Petal Width", 0.1, 2.5, 0.2, 0.1)
                
                manual_data = {
                    "sepal_length": sepal_length,
                    "sepal_width": sepal_width,
                    "petal_length": petal_length,
                    "petal_width": petal_width
                }
                
            else:  # Wine
                st.subheader("🍷 Wine Parameters")
                col1, col2 = st.columns(2)
                with col1:
                    alcohol = st.number_input("Alcohol", 11.0, 15.0, 13.2)
                    malic_acid = st.number_input("Malic Acid", 0.5, 6.0, 1.78)
                    ash = st.number_input("Ash", 1.0, 4.0, 2.14)
                    alcalinity = st.number_input("Alcalinity", 10.0, 30.0, 11.2)
                    magnesium = st.number_input("Magnesium", 70.0, 160.0, 100.0)
                    total_phenols = st.number_input("Total Phenols", 0.5, 4.0, 2.65)
                
                with col2:
                    flavanoids = st.number_input("Flavanoids", 0.5, 5.0, 2.76)
                    nonflavanoid = st.number_input("Nonflavanoid", 0.1, 0.7, 0.26)
                    proanthocyanins = st.number_input("Proanthocyanins", 0.4, 3.5, 1.28)
                    color_intensity = st.number_input("Color Intensity", 1.0, 13.0, 4.38)
                    hue = st.number_input("Hue", 0.5, 1.8, 1.05)
                    od280 = st.number_input("OD280/OD315", 1.0, 4.0, 3.40)
                    proline = st.number_input("Proline", 200.0, 2000.0, 1050.0)
                
                manual_data = {
                    "alcohol": alcohol,
                    "malic_acid": malic_acid,
                    "ash": ash,
                    "alcalinity_of_ash": alcalinity,
                    "magnesium": magnesium,
                    "total_phenols": total_phenols,
                    "flavanoids": flavanoids,
                    "nonflavanoid_phenols": nonflavanoid,
                    "proanthocyanins": proanthocyanins,
                    "color_intensity": color_intensity,
                    "hue": hue,
                    "od280_od315_of_diluted_wines": od280,
                    "proline": proline
                }
            
            st.session_state["input_data"] = manual_data
            st.session_state["data_available"] = True
            
        else:  # Upload JSON
            test_input_file = st.file_uploader('Upload test JSON', type=['json'])
            
            if test_input_file:
                st.write('Preview:')
                test_input_data = json.load(test_input_file)
                st.json(test_input_data)
                
                # Extract the actual data
                if "input_test" in test_input_data:
                    st.session_state["input_data"] = test_input_data["input_test"]
                else:
                    st.session_state["input_data"] = test_input_data
                    
                st.session_state["data_available"] = True
            else:
                st.session_state["data_available"] = False
        
        # Load example button
        if st.button("Load Example", help="Load example data"):
            if model_type == "Iris Flower":
                example_response = requests.get(f"{FASTAPI_BACKEND_ENDPOINT}/iris/example")
                if example_response.status_code == 200:
                    example = example_response.json()
                    st.session_state["input_data"] = example["example_input"]
                    st.session_state["data_available"] = True
                    st.success("Example loaded!")
            else:
                example_response = requests.get(f"{FASTAPI_BACKEND_ENDPOINT}/wine/example")
                if example_response.status_code == 200:
                    example = example_response.json()
                    st.session_state["input_data"] = example["example_input"]
                    st.session_state["data_available"] = True
                    st.success("Example loaded!")
        
        # Predict button
        predict_button = st.button('🔮 Predict', type='primary', use_container_width=True)
    
    # Main body
    st.title(f"{'🌸 Iris Flower' if model_type == 'Iris Flower' else '🍷 Wine'} Classification")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input Data")
        if st.session_state.get("data_available", False):
            st.json(st.session_state["input_data"])
        else:
            st.info("Please provide input data using the sidebar")
    
    with col2:
        st.subheader("Prediction Result")
        result_container = st.empty()
        
        if predict_button and st.session_state.get("data_available", False):
            with st.spinner('Predicting...'):
                try:
                    # Determine endpoint
                    if model_type == "Iris Flower":
                        endpoint = f"{FASTAPI_BACKEND_ENDPOINT}/iris/predict"
                    else:
                        endpoint = f"{FASTAPI_BACKEND_ENDPOINT}/wine/predict"
                    
                    # Send request
                    response = requests.post(
                        endpoint,
                        json=st.session_state["input_data"],
                        headers={'Content-Type': 'application/json'}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        if model_type == "Iris Flower":
                            prediction = result.get("prediction", result.get("class_id"))
                            species = result.get("species", result.get("class_name"))
                            confidence = result.get("confidence", "N/A")
                            
                            result_container.success(f"""
                            ### 🌸 Prediction: {species}
                            - **Class ID**: {prediction}
                            - **Confidence**: {confidence:.2%} if isinstance(confidence, (int, float)) else confidence
                            """)
                        else:
                            prediction = result.get("prediction", result.get("class_id"))
                            wine_class = result.get("wine_class", result.get("class_name"))
                            confidence = result.get("confidence", "N/A")
                            
                            result_container.success(f"""
                            ### 🍷 Prediction: {wine_class}
                            - **Class ID**: {prediction}
                            - **Confidence**: {confidence:.2%} if isinstance(confidence, (int, float)) else confidence
                            """)
                    else:
                        result_container.error(f"Error: {response.status_code}\n{response.text}")
                        
                except Exception as e:
                    result_container.error(f"Prediction failed: {str(e)}")
    
    # Footer with API info
    with st.expander("📚 API Documentation"):
        st.markdown("""
        ### Available Endpoints:
        - `GET /` - Welcome message
        - `GET /health` - Health check
        - `POST /iris/predict` - Iris prediction
        - `POST /wine/predict` - Wine prediction
        - `GET /iris/example` - Iris example data
        - `GET /wine/example` - Wine example data
        - `POST /predict` - Legacy iris endpoint
        
        ### FastAPI Interactive Docs:
        Visit http://localhost:8000/docs for interactive API documentation
        """)

if __name__ == "__main__":
    run()