import streamlit as st
import nltk
import torch
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import google.generativeai as genai
from dotenv import load_dotenv
import os 

nltk.download('punkt')
nltk.download('stopwords')
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
    print("✅ API Key loaded successfully!")
else:
    print("❌ Error: API Key not found.Using GPT-2 instead.")

@st.cache_resource()  # to prevent reloading everytime
def load_gpt2():
    return pipeline("text-generation", model="gpt2", device=0 if torch.cuda.is_available() else -1)

# Load GPT-2 only if Gemini API is unavailable
chatbot_gpt2 = None
if not api_key:
    chatbot_gpt2 = load_gpt2()

def healthcare_chatbot(user_input, model_choice):
    user_input = user_input.lower()
    if "symptom" in user_input:
        return "⚠️ Please consult doctor for an accurate diagnosis"
    elif "appointment" in user_input:
        return "📅 Would you like to schedule appointment with the doctor ?"
    elif "medication" in user_input:
        return "💊 It's important to take prescribed medicine regularly.If you have any concerns, consult your doctor. "
    
    if model_choice == "GPT-2":
        chatbot = load_gpt2()
        response = chatbot(user_input, max_length=150, num_return_sequences=1)
        return response[0]['generated_text']
    
    elif model_choice == "Gemini API":
        try:
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(user_input,generation_config={"max_output_tokens": 100, "temperature": 0.3, "top_p": 0.5})
            return response.text if response else "⚠️ No response received."
        except Exception as e:
            return f"❌ Error: {str(e)}"

    return "⚠️ Invalid model selection."

def main():
    st.title("Healthcare Assistant Chatbot")
    st.write("🤖 **Ask me any medical question!**")

    model_choice = st.selectbox("Choose a Model:", ["GPT-2", "Gemini API"])

    user_input = st.text_input("How can I assist you today?")
    
    if st.button("Submit"):
        if user_input:
            st.write("🧑‍💻 User: ",user_input)
            with st.spinner("Processing your queries Please wait......."):
                response = healthcare_chatbot(user_input,model_choice)
            st.write("🤖 Healthcare Assistant : ",response)
            print(response)
        else:
            st.write("⚠️ Please enter a message to get a response.")

main()