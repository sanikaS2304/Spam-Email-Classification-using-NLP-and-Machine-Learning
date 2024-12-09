import streamlit as st
import pickle
from PIL import Image

model = pickle.load(open('spam.pkl', 'rb'))
cv = pickle.load(open('vectorizer.pkl', 'rb'))

def main():
    st.set_page_config(page_title="Spam Classifier", layout="centered")
    st.title("📧 Email Spam Classifier")
    st.markdown("""
    Welcome to the **Email Spam Classification Application**!  
    This tool uses a Machine Learning model to classify emails as **Spam** or **Not Spam**.
    """)

    st.subheader("🔍 Enter Email Content Below:")
    user_input = st.text_area(
        "Type or paste the email content you want to classify:",
        height=200,
        placeholder="Enter email content here..."
    )

    if st.button("📊 Classify Email"):
        if user_input.strip():  
            data = [user_input]
            vec = cv.transform(data).toarray()
            result = model.predict(vec)
            if result[0] == 0:
                st.success("✅ This is **Not A Spam Email**!")
            else:
                st.error("🚨 This is a **Spam Email**!")
        else:
            st.warning("⚠️ Please enter some text to classify.")
   
main()