import streamlit as st
from disease_info import disease_info
import io

# Optionally, you can add a dictionary of external resources for each disease
external_resources = {
    "Tomato___Early_blight": [
        ("Research Article", "https://www.sciencedirect.com/science/article/pii/S0261219419302072"),
        ("Agricultural Extension", "https://extension.umn.edu/diseases/tomato-early-blight")
    ],
    # Add more as needed
}

def display_disease_info(disease_name):
    """
    Display detailed disease information in a formatted way using Streamlit.
    
    Args:
        disease_name (str): The name of the disease to display information for
    """
    if disease_name not in disease_info:
        st.error(f"No information available for {disease_name}")
        return
    
    info = disease_info[disease_name]
    
    # Display disease name as title
    st.title(f"🌱 {disease_name.replace('___', ' - ')}")
    
    # Collapsible sections for each info block
    with st.expander("🦠 Causal Organism", expanded=True):
        st.info(info["Causal Organism"])
    with st.expander("🔍 Symptoms", expanded=False):
        for symptom in info["Symptoms"]:
            st.write(f"• {symptom}")
    with st.expander("🔄 Disease Cycle", expanded=False):
        for stage in info["Disease Cycle"]:
            st.write(f"• {stage}")
    with st.expander("⚠️ Impact", expanded=False):
        for impact in info["Impact"]:
            st.write(f"• {impact}")
    with st.expander("🛡️ Management Strategies", expanded=False):
        for strategy in info["Management Strategies"]:
            st.write(f"• {strategy}")
    
    # External resources if available
    if disease_name in external_resources:
        st.markdown("**External Resources:**")
        for label, url in external_resources[disease_name]:
            st.markdown(f"- [{label}]({url})")
    
    # Download Info button
    disease_text = f"""
Disease: {disease_name.replace('___', ' - ')}\n\n"""
    disease_text += f"Causal Organism: {info['Causal Organism']}\n\n"
    disease_text += "Symptoms:\n" + "\n".join(f"- {s}" for s in info["Symptoms"]) + "\n\n"
    disease_text += "Disease Cycle:\n" + "\n".join(f"- {s}" for s in info["Disease Cycle"]) + "\n\n"
    disease_text += "Impact:\n" + "\n".join(f"- {s}" for s in info["Impact"]) + "\n\n"
    disease_text += "Management Strategies:\n" + "\n".join(f"- {s}" for s in info["Management Strategies"]) + "\n\n"
    if disease_name in external_resources:
        disease_text += "External Resources:\n" + "\n".join(f"- {label}: {url}" for label, url in external_resources[disease_name]) + "\n\n"
    st.download_button(
        label="Download Info as Text",
        data=disease_text,
        file_name=f"{disease_name.replace('___', '_')}_info.txt",
        mime="text/plain"
    )
    
    # Add some spacing
    st.markdown("---")
    
    # Add a note about the information
    st.info("""
    ℹ️ This information is provided for educational purposes only. 
    Always consult with agricultural experts for specific treatment recommendations.
    """)

# Example usage:
if __name__ == "__main__":
    st.set_page_config(page_title="Plant Disease Information", layout="wide")
    st.title("Plant Disease Information System")
    
    # Create a dropdown to select disease
    disease_names = list(disease_info.keys())
    selected_disease = st.selectbox("Select a disease to view information:", disease_names)
    
    # Display information for selected disease
    display_disease_info(selected_disease) 