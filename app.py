from functions import get_results, plot_results, classify_input, show_pie_chart
from CONST import PAGES, CLASSES, MODELS_NAMES
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

df = pd.read_csv("./data/02/emotions_full.csv", index_col=0)
df_1 = pd.read_csv("./data/01/Emotion_final.csv")
df_2 = pd.read_csv('./data/01/text_emotion.csv', usecols=["sentiment", "content"])


st.set_page_config(
    page_title="Sentiment analysis",
    layout='wide',
    initial_sidebar_state='expanded'
)

# ----------------- #
#      SIDEBAR      #
# ----------------- #


page_select = st.sidebar.selectbox("Pages : ", PAGES)

# -------------- #
#      BODY      #
# -------------- #

#======================= PAGE VISUALISATION =======================#

empty = st.empty()

if page_select == PAGES[0]:
    st.title("Visualisation des données.")
    st.header("Analyse de la donnée d'entrainement")
    st.subheader("emotions_full.csv")
    col1, col2, col3 = st.beta_columns(3)

    with col1:
        st.pyplot(show_pie_chart(df))
    
    with col2:
        st.subheader("Emotion_final.csv")
        counts_1 = df_1.Emotion.value_counts()
        st.bar_chart(counts_1)
    
    with col3:
        st.subheader("text_emotion.csv")
        counts_2 = df_2.sentiment.value_counts()
        st.bar_chart(counts_2)

#======================= PAGE RESULTATS =======================#

if page_select == PAGES[1]:
    st.title("Visualisation des résultats.")
    col1, col2 = st.beta_columns([1,1.5])

    with col1:
        model_select = st.selectbox("Modèles : ", MODELS_NAMES)
        y_test, y_probas = get_results(df, model_select)
        select = st.multiselect(
            'Classes à afficher :', CLASSES, help="Représentation des résultats sous la forme AUC-ROC"
        )
        classes_to_plot = select
        if not select:
            classes_to_plot = None

    with col2:
        plots = plot_results(y_test, y_probas, title=model_select, classes_to_plot=classes_to_plot)
        st.pyplot(plots)

#======================= PAGE PRÉDICTION =======================#


if page_select == PAGES[2]:
    col, _ = st.beta_columns(2)
    with col:
        user_input = st.text_input('Input').lower().capitalize()

    if user_input:
        results = classify_input(user_input,df)
        for result in results:
            col1, col2 = st.beta_columns(2)

            with col1:
                st.header(result[0])
                st.subheader(f"Prédiction : {result[1][0]}")
                scores = result[2].tolist()
                with st.beta_expander('Plus de détails'):
                    for i in range(len(CLASSES)):
                        st.text(f"Proabilité d'appartenir à la classe {CLASSES[i]} : {round(scores[0][i],5)}")

            with col2:
                res_df = pd.DataFrame(zip(CLASSES, list(result[2].T.flat)))
                res_df = res_df.set_index(0)
                st.dataframe(res_df)
                st.bar_chart(result[2].T)

            st.markdown("--------------")


