import streamlit as st
import requests
from PIL import Image
import io

# Configuration de la page
st.set_page_config(
    page_title="Is it grass or a dandelion ?",
    page_icon="🌱",
    layout="centered"
)

# Titre
st.title("Is it grass or a dandelion ?")
st.write("Select an image to find out whether it shows grass or a dandelion.")

# URL de l'API (modifiable)
api_url = st.sidebar.text_input(
    "API URL",
    value="http://localhost:8000/analyze-image"
)

# Upload de fichier
uploaded_file = st.file_uploader(
    "Choose an image file",
    type=['png', 'jpg', 'jpeg'],
    help="Formats acceptés : PNG, JPG, JPEG"
)

# Si un fichier est uploadé
if uploaded_file is not None:
    # Afficher l'image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded picture", use_column_width=True)
    
    # Bouton d'analyse
    if st.button("Analysing", type="primary"):
        with st.spinner("Analysis in progress..."):
            try:
                # Préparer les données
                uploaded_file.seek(0)
                files = {
                    'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                }
                
                # Envoyer à l'API
                response = requests.post(api_url, files=files, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Afficher le résultat
                    if result.get('is_grass'):
                        st.success("This image represents grass.")
                    else:
                        st.error("This image represents dandelion.")
                    
                    # Détails supplémentaires
                    col1, col2 = st.columns(2)
                    if 'confidence' in result:
                        col1.metric("Confiance", f"{result['confidence']}%")
                    if 'minio_url' in result:
                        st.info(f"Image saved on MinIO")
                        
                else:
                    st.error(f"Erreur {response.status_code}: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Connexion impossible. Vérifiez que l'API est lancée.")
            except Exception as e:
                st.error(f"Erreur : {str(e)}")
else:
    st.info("Start by uploading an image file.")