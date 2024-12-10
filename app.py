import os
import json
from datetime import datetime  
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
def chatbot(user_input):
    # Transform the user input and predict the intent
    input_text = vectorizer.transform([user_input])
    tag = clf.predict(input_text)[0]
    
    # Find the response based on the predicted tag
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])  # Pick a random response from the matched intent
            return response
    return "Sorry, I didn't understand that."  # Default response if no match is found

# Ensure NLTK data is downloaded only once
if 'nltk_downloaded' not in st.session_state:
    ssl._create_default_https_context = ssl._create_unverified_context
    nltk.data.path.append(os.path.abspath("nltk_data"))
    nltk.download('punkt')
    st.session_state.nltk_downloaded = True

# Initialize chat history in session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Main application
def main():
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

        # Initialize session state for storing conversation
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display form with clear-on-submit functionality
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input("Your Message:", placeholder="Type something here...")
            submit_button = st.form_submit_button("Send")

        if submit_button and user_input:  # Check for non-empty input and submission
            response = chatbot(user_input)  # Get chatbot response

            # Append to conversation history
            st.session_state.messages.append({"user": user_input, "bot": response})

            # Log conversation to a file
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input, response, datetime.now()])  # Log timestamped interaction

        # Display only the latest chat message
        if st.session_state.messages:
            latest_message = st.session_state.messages[-1]
            st.markdown(f"**🗣 User:** {latest_message['user']}")
            st.markdown(f"**🤖 Chatbot:** {latest_message['bot']}")
            st.markdown("---")

    elif choice == "📜 Conversation History":
        st.header("📜 Conversation History")
        
        # Check if the file exists
        if os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                history = list(csv_reader)
            
            # Check if the CSV file contains any rows
            if len(history) > 0:  # Ensure there is data in the file
                for entry in history:
                    if len(entry) >= 3:  # Ensure each row has enough columns
                        st.markdown(f"**🗣 User:** {entry[0]}")
                        st.markdown(f"**🤖 Chatbot:** {entry[1]}")
                        st.markdown(f"⏰ **Timestamp:** {entry[2]}")
                        st.markdown("---")
            else:
                st.warning("No conversation history found. The file is empty.")
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
