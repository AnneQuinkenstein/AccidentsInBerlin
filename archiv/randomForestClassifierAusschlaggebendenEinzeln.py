import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# Laden des Datensatzes
data = pd.read_csv('../data/GeneralDatensatz18-21ohneGeo.csv', sep=';')

# Annahme: UKATEGORIE ist die Zielvariable und alle anderen Merkmale sind Prädiktoren
target = 'UKATEGORIE'
features = ['USTUNDE', 'UMONAT', 'UWOCHENTAG', 'BEZ', 'UTYP1', 'UJAHR']


X = data[features]
y = (data[target] == 1).astype(int)

# Codieren der kategorischen Merkmale
X = pd.get_dummies(X, columns=features)

# Trainieren des Modells
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Extrahieren der Feature-Importanzen
importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Aggregieren der Importanzen nach den ursprünglichen Merkmalen
feature_importance_df['OriginalFeature'] = feature_importance_df['Feature'].apply(lambda x: x.split('_')[0])
category_importance = feature_importance_df.groupby('OriginalFeature').sum().reset_index()

# Plotten der Feature-Importanzen
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='OriginalFeature', data=category_importance.sort_values(by='Importance', ascending=False))
plt.title('Wichtigkeit der ursprünglichen Merkmale')
plt.show()

# Plotten der wichtigsten Kategorien innerhalb der Merkmale
for feature in features:
    subset = feature_importance_df[feature_importance_df['OriginalFeature'] == feature]
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=subset.sort_values(by='Importance', ascending=False))
    plt.title(f'Wichtigkeit der Kategorien innerhalb des Merkmals {feature}')
    plt.show()
