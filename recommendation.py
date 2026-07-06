import joblib
import pandas as pd
from scipy.sparse import load_npz

# Load models
df = joblib.load("df.pkl")
knn_model = joblib.load("knn_model.pkl")
course_features = load_npz("course_features.npz")

def recommend_knn(course_name, n=5):

    if course_name not in df["course_name"].values:
        return None

    idx = df[df["course_name"] == course_name].index[0]

    distances, indices = knn_model.kneighbors(
        course_features[idx:idx+1],
        n_neighbors=n+1
    )

    recommendations=[]

    for d,i in zip(distances.flatten()[1:],indices.flatten()[1:]):

        recommendations.append({

            "Course Name":df.iloc[i]["course_name"],
            "Difficulty Level": df.iloc[i]["difficulty_level"],
            "Certification": df.iloc[i]["certification_offered"],
            "Rating": df.iloc[i]["rating"],
            "Similarity Score":f"{(1-d):.3f}"

        })

    return pd.DataFrame(recommendations)