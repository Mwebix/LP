 #ligand predictor
pip install scikit-learn

import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from rdkit import Chem
from rdkit.Chem import Draw
from io import BytesIO
import base64

st.header(':red[Webixinol LP] :sunglasses:', divider='blue')

# Define model in the global scope
model = DecisionTreeClassifier()

# Step 1: Data preparation
def main():
    st.sidebar.title('Ligand Predictor')
    uploaded_file = st.sidebar.file_uploader("Upload Excel file", type=['xlsx'])

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        st.write(df.head())

        # Split the data into features (Z-scores) and labels (SMILES)
        features = df['Z-Scores']
        labels = df['SMILES']

        # Encode labels into numeric values using OneHotEncoder
        label_encoder = OneHotEncoder(sparse_output=False)
        labels_encoded = label_encoder.fit_transform(labels.values.reshape(-1, 1))

        # Step 2: Model Training
        X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)
         

        # Train the model
        model.fit(X_train.values.reshape(-1, 1), y_train)

        # Step 3: User Input and Prediction
        st.write("### Prediction")
        z_score = st.number_input('Enter a Z-score:')
        binary_prediction = model.predict([[z_score]])

        # Convert the encoded label back to its original form
        binary_object = label_encoder.inverse_transform(binary_prediction)

        predicted_smiles = binary_object[0][0]  # Extracting the string from numpy array

        st.write("Predicted SMILES:", predicted_smiles)

        # Generate chemical structure image
        mol = Chem.MolFromSmiles(predicted_smiles)
        if mol is not None:
            img = Draw.MolToImage(mol)

            # Convert image to bytes
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()

            # Display image
            st.image(img, caption='Predicted Ligand Structure', width=300, use_column_width=False)

            

            # Offer download link for image
            st.sidebar.markdown("## Download")
            st.sidebar.markdown(get_binary_file_downloader_html(img_byte_arr, 'Ligand_Structure.png', 'Download Ligand Structure'), unsafe_allow_html=True)
        else:
            st.write("Unable to generate chemical structure for the predicted SMILES.")


def get_binary_file_downloader_html(bin_file, file_label='File', button_text='Download'):
    """
    Generates a link to download the given binary file.
    """
    bin_str = bin_file
    bin_file_str = str(bin_str)
    b64 = base64.b64encode(bin_str).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_label}">{button_text}</a>'
    return href


if __name__ == "__main__":
    main()
