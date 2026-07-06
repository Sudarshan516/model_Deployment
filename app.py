import streamlit as st
import joblib
from recommendation import recommend_knn

# -------------------------
# Load Data
# -------------------------
df = joblib.load("df.pkl")

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Online Course Recommendation System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Sidebar (ADD HERE)
# =========================

st.sidebar.title("Recommendation Model")

model = st.sidebar.selectbox(
    "Select Model",
    [
        "Content Based",
        "KNN",
        "Collaborative Filtering"
    ]
)

# =========================
# Main Page
# =========================

st.title("Online Course Recommendation System")

st.write("Find similar online courses using different recommendation techniques.")

# Course Selection
course = st.selectbox(
    "Select a Course",
    sorted(df["course_name"].unique())
)

# Recommend Button
if st.button("Recommend"):
    result = recommend_knn(course)

    if result is not None:
        st.success("Recommended Courses")
        st.dataframe(result)
    else:
        st.error("Course not found.")

