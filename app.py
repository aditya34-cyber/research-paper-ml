import streamlit as st
import pandas as pd
import joblib

from sklearn.base import BaseEstimator, TransformerMixin



print("Model saved")

class ChessFeatureTransformer(BaseEstimator, TransformerMixin):

    def fit(self,X,y=None):
        return self

    def transform(self,X):

        X = X.copy()

        X["rating_diff"] = X["white_rating"] - X["black_rating"]

        X["abs_rating_diff"] = abs(X["rating_diff"])

        X["rating_ratio"] = X["white_rating"]/(X["black_rating"]+1)

        return X


# ===== PAGE =====

st.set_page_config(page_title="Chess Predictor",layout="wide")

st.title("♟ Chess Outcome Predictor")

# ===== LOAD =====

model = joblib.load("chess_model.pkl")


# ===== INPUT =====

st.sidebar.header("Game Features")

white_rating = st.sidebar.slider("White rating",800,2800,1500)

black_rating = st.sidebar.slider("Black rating",800,2800,1500)

turns = st.sidebar.slider("Turns",5,200,40)

opening_ply = st.sidebar.slider("Opening ply",1,10,4)

increment_base = st.sidebar.slider("Base time",1,60,5)

increment_bonus = st.sidebar.slider("Increment",0,30,3)

rated = st.sidebar.selectbox("Rated",[True,False])

victory_status = st.sidebar.selectbox(
"Victory type",
["mate","resign","outoftime"]
)

opening_eco = st.sidebar.text_input("Opening ECO","C20")

opening_name = st.sidebar.text_input(
"Opening name",
"King Pawn Opening"
)

# ===== DATAFRAME =====

data = pd.DataFrame([{

"rated":rated,

"turns":turns,

"victory_status":victory_status,

"white_rating":white_rating,

"black_rating":black_rating,

"opening_eco":opening_eco,

"opening_name":opening_name,

"opening_ply":opening_ply,

"increment_code":str(increment_base)+"+"+str(increment_bonus)

}])


# ===== PREDICT =====

prediction = model.predict(data)

probabilities = model.predict_proba(data)

labels = ["Black win","Draw","White win"]

# ===== OUTPUT =====

st.subheader("Prediction")

st.success(labels[prediction[0]])

st.subheader("Confidence")

c1,c2,c3 = st.columns(3)

c1.metric("Black %",round(probabilities[0][0]*100,2))

c2.metric("Draw %",round(probabilities[0][1]*100,2))

c3.metric("White %",round(probabilities[0][2]*100,2))

# ===== BAR CHART =====

prob_df = pd.DataFrame({

"Result":labels,

"Probability":probabilities[0]

})

st.subheader("Probability Distribution")

st.bar_chart(prob_df.set_index("Result"))

# ===== RATING VISUAL =====

rating_df = pd.DataFrame({

"Player":["White","Black"],

"Rating":[white_rating,black_rating]

})

st.subheader("Rating Comparison")

st.bar_chart(rating_df.set_index("Player"))

# ===== FEATURE INSIGHT =====

diff = white_rating-black_rating

st.subheader("Insight")

if diff>50:

    st.info("White has rating advantage")

elif diff<-50:

    st.info("Black has rating advantage")

else:

    st.info("Players have similar strength")