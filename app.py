import streamlit as st
import pandas as pd
import numpy as np
import joblib  
import matplotlib.pyplot as plt
import seaborn as sns 

model_path = "best_model_Avec_Features_Sélectionnées.pkl" 
loaded_model = joblib.load(model_path) 


st.title("Tableau de Bord d’Octroi de Crédit")

st.markdown("""
### Saisie des Caractéristiques Client
Veuillez renseigner les informations du client ci-dessous :
""")

@st.cache_data
def load_data():

    np.random.seed(42)
    data_size = 200
    df_mock = pd.DataFrame({
        "LIMIT_BAL": np.random.randint(5000, 500000, size=data_size),
        "SEX": np.random.choice([1,2], size=data_size),
        "EDUCATION": np.random.choice([1,2,3,4], size=data_size),
        "MARRIAGE": np.random.choice([1,2,3], size=data_size),
        "AGE": np.random.randint(18, 70, size=data_size),
        "PAY_0": np.random.randint(-1, 10, size=data_size),
        "PAY_2": np.random.randint(-1, 10, size=data_size),
        "PAY_3": np.random.randint(-1, 10, size=data_size),
        "BILL_AMT1": np.random.randint(0, 300000, size=data_size),
        "BILL_AMT4": np.random.randint(0, 300000, size=data_size),
        "BILL_AMT5": np.random.randint(0, 300000, size=data_size),
        "PAY_AMT1": np.random.randint(0, 200000, size=data_size),
        "PAY_AMT2": np.random.randint(0, 200000, size=data_size),
        "PAY_AMT3": np.random.randint(0, 200000, size=data_size),
        "PAY_AMT4": np.random.randint(0, 200000, size=data_size),
        "PAY_AMT5": np.random.randint(0, 200000, size=data_size),
        "PAY_AMT6": np.random.randint(0, 200000, size=data_size),
        "DEFAULT_NEXT_MONTH": np.random.choice([0,1], size=data_size, p=[0.7, 0.3])
    })
    return df_mock

df = load_data()


limit_bal = st.number_input("Montant du crédit (LIMIT_BAL)", min_value=0, value=50000, step=1000)


sex = st.selectbox("Sexe (1=Homme, 2=Femme)", [1, 2])


education = st.selectbox("Niveau d'Education (1=Grad School, 2=University, 3=HighSchool, 4=Autres)", [1,2,3,4])


marriage = st.selectbox("Statut Marital (1=Married, 2=Single, 3=Autres)", [1,2,3])


age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)


pay_0 = st.slider("PAY_0 (Historique de remboursement du dernier mois)", min_value=-1, max_value=9, value=0)


pay_2 = st.slider("PAY_2 (Historique de remboursement 2e mois)", min_value=-1, max_value=9, value=0)


pay_3 = st.slider("PAY_3 (Historique de remboursement 3e mois)", min_value=-1, max_value=9, value=0)


bill_amt1 = st.number_input("BILL_AMT1 (Montant de la dernière facture)", min_value=0, value=20000, step=1000)

bill_amt4 = st.number_input("BILL_AMT4 (Montant facture à 4 mois)", min_value=0, value=10000, step=1000)

bill_amt5 = st.number_input("BILL_AMT5 (Montant facture à 5 mois)", min_value=0, value=10000, step=1000)

pay_amt1 = st.number_input("PAY_AMT1 (Montant payé le mois dernier)", min_value=0, value=15000, step=1000)


pay_amt2 = st.number_input("PAY_AMT2 (Montant payé il y a 2 mois)", min_value=0, value=5000, step=500)

pay_amt3 = st.number_input("PAY_AMT3 (Montant payé il y a 3 mois)", min_value=0, value=5000, step=500)


pay_amt4 = st.number_input("PAY_AMT4 (Montant payé il y a 4 mois)", min_value=0, value=3000, step=500)

pay_amt5 = st.number_input("PAY_AMT5 (Montant payé il y a 5 mois)", min_value=0, value=3000, step=500)


pay_amt6 = st.number_input("PAY_AMT6 (Montant payé il y a 6 mois)", min_value=0, value=2000, step=500)


input_data = pd.DataFrame({
    "LIMIT_BAL": [limit_bal],
    "SEX": [sex],
    "EDUCATION": [education],
    "MARRIAGE": [marriage],
    "AGE": [age],
    "PAY_0": [pay_0],
    "PAY_2": [pay_2],
    "PAY_3": [pay_3],
    "BILL_AMT1": [bill_amt1],
    "BILL_AMT4": [bill_amt4],
    "BILL_AMT5": [bill_amt5],
    "PAY_AMT1": [pay_amt1],
    "PAY_AMT2": [pay_amt2],
    "PAY_AMT3": [pay_amt3],
    "PAY_AMT4": [pay_amt4],
    "PAY_AMT5": [pay_amt5],
    "PAY_AMT6": [pay_amt6]
})


st.subheader("**Prédiction du Risque de Défaut**")
if st.button("Prédire"):
    prediction = loaded_model.predict(input_data)
    st.write(f"**Prédiction (DEFAULT = 1) :** {prediction[0]}")

    if hasattr(loaded_model, "predict_proba"):
        prob = loaded_model.predict_proba(input_data)[:,1][0]
        st.write(f"**Probabilité de défaut :** {prob*100:.2f}%")


    st.markdown("---")
    st.markdown("### **Recommandations Personnalisées**")

   
    if hasattr(loaded_model, "predict_proba"):
        if prob > 0.5:
            st.warning("Le risque de défaut est relativement élevé. **Recommandations :**")
            st.write("- Limiter le montant de crédit additionnel.")
            st.write("- Demander des garanties ou un co-signataire.")
            st.write("- Mettre en place un suivi rapproché pour les échéances.")
        else:
            st.success("Le risque de défaut semble faible. **Recommandations :**")
            st.write("- Possibilité d'augmenter légèrement la limite de crédit.")
            st.write("- Proposer des offres de fidélisation (cartes premium, cashback).")

    else:
        
        if prediction[0] == 1:
            st.warning("Le modèle prédit un défaut. **Recommandations :**")
            st.write("- Limiter le montant de crédit additionnel.")
            st.write("- Demander des garanties ou un co-signataire.")
        else:
            st.success("Le modèle ne prédit pas de défaut. **Recommandations :**")
            st.write("- Possibilité d'augmenter légèrement la limite de crédit.")
            st.write("- Proposer des offres de fidélisation (cartes premium, cashback).")



st.markdown("---")
st.subheader("**Tableau de Bord : Visualisation du Dataset**")


variable_to_plot = st.selectbox(
    "Choisissez une variable numérique à visualiser",
    ["LIMIT_BAL", "AGE", "BILL_AMT1", "BILL_AMT4", "BILL_AMT5", "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Histogramme**")
    fig, ax = plt.subplots(figsize=(4,3))
    sns.histplot(df[variable_to_plot], kde=True, ax=ax)
    ax.set_title(f"Distribution de {variable_to_plot}")
    st.pyplot(fig)

with col2:
    st.markdown("**Boxplot**")
    fig2, ax2 = plt.subplots(figsize=(2.5,3))
    sns.boxplot(y=df[variable_to_plot], ax=ax2)
    ax2.set_title(f"Boxplot de {variable_to_plot}")
    st.pyplot(fig2)


st.markdown("### Comparaison selon la variable cible (DEFAULT_NEXT_MONTH)")
col_cat = st.selectbox(
    "Choisissez une variable catégorielle (ex. SEX, EDUCATION, etc.)",
    ["SEX", "EDUCATION", "MARRIAGE", "PAY_0"]
)

fig3, ax3 = plt.subplots(figsize=(4,3))
sns.barplot(x=col_cat, y="LIMIT_BAL", hue="DEFAULT_NEXT_MONTH", data=df, ci=None, ax=ax3)
ax3.set_title(f"LIMIT_BAL moyen par {col_cat} et DEFAULT_NEXT_MONTH")
st.pyplot(fig3)

st.markdown("""
---
### **Informations Clés :**
- Les **histogrammes** et **boxplots** permettent d’observer la distribution des principales variables (montant de crédit, montant de factures, etc.).
- La **comparaison par DEFAULT_NEXT_MONTH** indique comment les caractéristiques diffèrent entre les clients qui font défaut et ceux qui remboursent.
- Les **recommandations** proposées se basent sur la probabilité de défaut et peuvent être adaptées selon le contexte de la banque ou du gestionnaire de risque.
""")