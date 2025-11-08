import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_sample_weight
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2, f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.model_selection import  train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
df = pd.read_csv('Credit Score Classification Dataset.csv')

# Aperçu des 5 premières lignes
df.head()

# Variables catégorielles
cat_vars = ['Gender', 'Education', 'Marital Status', 'Home Ownership']
for col in cat_vars:
    plt.figure(figsize=(6,4))
    sns.countplot(x=col, hue='Credit Score', data=df)
    plt.title(f"Répartition du Credit Score selon {col}")
    plt.xticks(rotation=45)
    plt.show()

#  Variables numériques
num_vars = ['Age', 'Income', 'Number of Children']
for col in num_vars:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='Credit Score', y=col, data=df)
    plt.title(f"{col} vs Credit Score")
    plt.show()
# Encodage de la cible
y = LabelEncoder().fit_transform(df['Credit Score'])

# Séparation des types
categorical_cols = df.select_dtypes(include='object').columns.drop('Credit Score')
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Chi² pour les variables catégorielles
df_chi2 = df[categorical_cols].apply(LabelEncoder().fit_transform)
chi2_selector = SelectKBest(score_func=chi2, k='all')
chi2_selector.fit(df_chi2, y)
chi2_scores = pd.Series(chi2_selector.scores_, index=categorical_cols).sort_values(ascending=False)

# ANOVA pour les variables numériques
anova_selector = SelectKBest(score_func=f_classif, k='all')
anova_selector.fit(df[numeric_cols], y)
anova_scores = pd.Series(anova_selector.scores_, index=numeric_cols).sort_values(ascending=False)

# Résumé
print("Scores Chi² (variables catégorielles) :")
print(chi2_scores)
print("\n Scores ANOVA (variables numériques) :")
print(anova_scores)

selected_features = ['Home Ownership', 'Marital Status', 'Education', 'Income', 'Age']

import pandas as pd

# Variables catégorielles à encoder
categorical_features = ['Home Ownership', 'Marital Status', 'Education']
target_map = {"Low": 0, "Average": 1, "High": 2}
# Encoder avec get_dummies
df_encoded = pd.get_dummies(df[selected_features], columns=categorical_features, drop_first=True)

df_encoded.head()

x =df_encoded
y = df["Credit Score"].map(target_map)

x_train ,x_test , y_train ,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# Normalisation
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
x_train_res, y_train_res = smote.fit_resample(x_train_scaled, y_train)


model = GradientBoostingClassifier(random_state=42,
                                   learning_rate=0.01,
                                   max_leaf_nodes=3,
                                   n_estimators=50)

model.fit(x_train_res, y_train_res)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Recall:", recall)
print("F1-score:", f1)

import joblib

# Sauvegarder le modèle et le scaler
joblib.dump(model, "credit_model.pkl")
joblib.dump(scaler, "scaler.pkl")
columns = df_encoded.columns  # toutes les colonnes après encodage
joblib.dump(columns, "columns.pkl")  # sauvegarde-les

print(" Modèle et scaler sauvegardés avec succès !")
