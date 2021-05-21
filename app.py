from functions import get_results, plot_results
from CONST import PAGES, CLASSES, MODELS_NAMES
import pandas as pd
import streamlit as st

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

user_input = st.sidebar.text_input('Input').lower().capitalize()

# -------------- #
#      BODY      #
# -------------- #



#==== PAGE VISUALISATION ====#


empty = st.empty()

if page_select == PAGES[0]:
    empty.empty()
    with empty.beta_container():
        st.title("Visualisation des données.")
        st.header("Analyse de la donnée d'entrainement")
        st.subheader("Représentation des différentes valeurs")
        st.subheader("Emotion_final.csv")

        counts_1 = df_1.Emotion.value_counts()
        st.dataframe(counts_1)
        st.bar_chart(counts_1)
        st.markdown("-----------")

        st.markdown("# text_emotion.csv")
        counts_2 = df_2.sentiment.value_counts()
        st.dataframe(counts_2)
        st.bar_chart(counts_2)

#====== PAGE RESULTATS ======#


if page_select == PAGES[1]:
    empty.empty()
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
