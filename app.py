import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Handle SSL and NLTK data
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = 'intents.json'
with open(file_path, "r") as file:
    intents = json.load(file)

# Create vectorizer and classifier
vectorizer = TfidfVectorizer(ngram_range=(1, 4))
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess and train the model
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Chatbot response function
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

# Initialize counter for unique input keys
counter = 0

# Enhanced interface using Streamlit
def main():
    global counter
    st.set_page_config(page_title="NLP Chatbot", layout="wide")
    st.title("🤖 TalkyBish")
    st.markdown("---")
    
    menu = ["💬 Chat", "📜 Conversation History", "ℹ️ About"]
    choice = st.sidebar.radio("Navigation", menu)

    if choice == "💬 Chat":
        st.header("Welcome to the Chatbot 🤝")
        st.write("Type your message below and let’s have a conversation!")

        # Ensure chat log exists
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        counter += 1
        user_input = st.text_input("Your Message:", key=f"user_input_{counter}", placeholder="Type something here...")

        if user_input:
            response = chatbot(user_input)
            st.success(f"🤖 Chatbot: {response}")
            
            # Log to CSV
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.balloons()
                st.info("Thank you for chatting! Have a great day! 👋")
                st.stop()

    elif choice == "📜 Conversation History":
        st.header("📜 Conversation History")
        if os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip header
                history = list(csv_reader)
            
            if history:
                for entry in history:
                    st.markdown(f"**🗣 User:** {entry[0]}")
                    st.markdown(f"**🤖 Chatbot:** {entry[1]}")
                    st.markdown(f"⏰ **Timestamp:** {entry[2]}")
                    st.markdown("---")
            else:
                st.warning("No conversation history found.")
        else:
            st.warning("No conversation history available yet.")

    elif choice == "ℹ️ About":
        st.header("ℹ️ About the Chatbot")
        st.write("""
        This chatbot is built using **Logistic Regression** and **Natural Language Processing (NLP)** to understand user intents and provide appropriate responses.
        - **Model**: Logistic Regression
        - **Framework**: Streamlit
        - **Features**: Tracks conversation history, supports interactive UI
        """)
        st.markdown("### Project Highlights")
        st.write("""
        - **Training Data**: Intent-based labeled data
        - **Goal**: To demonstrate NLP techniques combined with Logistic Regression for building an efficient chatbot
        - **Enhancements**: This can be extended using advanced machine learning or deep learning techniques.
        """)

        st.markdown("### Future Scope")
        st.write("""
        - Incorporate deep learning models like RNNs, LSTMs, or Transformers
        - Enhance user experience with sentiment analysis or intent recognition improvements
        - Add a richer dataset for more diverse interactions
        """)
        st.markdown("---")
        st.write("**Made with ❤️ by Bishal Sarkar**")
        st.write("📚 3rd Year B.Tech in Electronics and Communication Engineering")
        st.write("🎓 Guru Nanak Institute of Technology")

if __name__ == '__main__':
    main()
