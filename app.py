import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# ----------------------------
# Dataset
# ----------------------------
emails = [
    "Congratulations! You’ve won a free iPhone",
    "Win cash now!!!",
    "Claim your lottery prize",
    "Exclusive deal just for you",
    "You have been selected for a reward",
    "Earn money from home easily",
    "Limited time offer",
    "Free vacation waiting for you",
    "Meeting scheduled at 10 AM",
    "Please find attached the report",
    "Let’s catch up tomorrow",
    "Project deadline reminder",
    "Lunch at 1 PM?",
    "Invoice for last month",
    "Team meeting agenda",
    "Birthday party invitation",
    "Can you review this document?",
    "Call me when free",
    "Your order has been shipped",
    "Thanks for your help"
]

labels = [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]

# ----------------------------
# Vectorization
# ----------------------------
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    ngram_range=(1,2)
)

X = vectorizer.fit_transform(emails)

# ----------------------------
# Train Model
# ----------------------------
model = LinearSVC()
model.fit(X, labels)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📧 Spam Email Detection App")
st.write("Enter an email message to check whether it is Spam or Not Spam.")

user_input = st.text_area("Enter Email Text Here")

if st.button("Check Email"):
    if user_input.strip() != "":
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)

        if prediction[0] == 1:
            st.error("🚨 This Email is SPAM!")
        else:
            st.success("✅ This Email is NOT Spam.")
    else:
        st.warning("Please enter some text.")



