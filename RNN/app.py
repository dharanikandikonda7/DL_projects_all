import streamlit as st
import difflib

st.set_page_config(
    page_title="Smart Reply Assistant",
    page_icon="💬"
)

st.title("💬 Smart Reply Assistant (RNN Concept)")

st.write("""
Generate quick responses for messages using sequence-based text matching.
This demonstrates the idea behind conversational sequence modeling used in RNN applications.
""")

reply_database = {
    "hi": "Hello! How can I help you today?",
    "hello": "Hi! Hope you're having a great day.",
    "how are you": "I'm doing well. Thanks for asking!",
    "are you available tomorrow": "Yes, I am available tomorrow.",
    "can we meet tomorrow": "Sure, let's schedule a meeting.",
    "thank you": "You're welcome!",
    "thanks": "Glad I could help.",
    "good morning": "Good morning! Have a productive day.",
    "good night": "Good night! Take care.",
    "where are you": "I'm currently available online.",
    "what is your name": "I'm your AI assistant.",
    "can you help me": "Of course! Tell me what you need help with.",
    "i need support": "I'd be happy to assist you.",
    "when is the meeting": "Please check the latest schedule for meeting details.",
    "send me details": "Sure, I'll share the details shortly.",
    "what are your working hours": "Our working hours are 9 AM to 6 PM.",
    "how much does it cost": "Please provide more details about the product or service."
}

user_input = st.text_area(
    "Enter a message",
    height=120,
    placeholder="Example: Are you available tomorrow?"
)

if st.button("Generate Reply"):

    if not user_input.strip():
        st.warning("Please enter a message.")
    else:

        message = user_input.lower().strip()

        matches = difflib.get_close_matches(
            message,
            reply_database.keys(),
            n=1,
            cutoff=0.3
        )

        if matches:
            response = reply_database[matches[0]]
        else:
            response = (
                "Thank you for your message. "
                "I will review it and get back to you soon."
            )

        st.subheader("Suggested Reply")
        st.success(response)

st.markdown("---")

st.subheader("Applications")

st.write("""
- Customer Support
- Email Reply Suggestions
- WhatsApp Auto Replies
- Chatbots
- Helpdesk Automation

RNNs are commonly used for sequence processing tasks such as text generation,
conversation modeling, and response prediction.
""")