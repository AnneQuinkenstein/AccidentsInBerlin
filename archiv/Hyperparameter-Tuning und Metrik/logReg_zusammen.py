import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# Laden des Datensatzes
df = pd.read_csv('../data/GeneralDatensatz18-21ohneGeo-mitLockdown_mitCorona.csv', sep=';')

# Features und Zielvariable definieren
X = df[['UMONAT','USTUNDE','UWOCHENTAG','UART','USTRZUSTAND','BEZ','UTYP1','ULICHTVERH','IstRad','IstPKW','IstFuss','IstKrad','IstGkfz','IstSonstige', 'LOCKDOWN', 'COVID']]
y = df['UKATEGORIE'].isin([1, 2]).astype(int)  # 1 für schwere/tödliche Unfälle, 0 für leichte Unfälle

# KFold-Konfiguration
sf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Definieren des F-beta-Scores mit beta = 2
beta = 2
fbeta_scorer = make_scorer(fbeta_score, beta=beta)

float_list_Cs = [x * 0.001 for x in range(10, 1000)]
# Definieren des Parametergrids für die Grid Search
param_grid = {
    'logistic__C': float_list_Cs,
    'logistic__penalty': ['l2'],
    'logistic__solver': ['lbfgs'],
    'logistic__max_iter': [150, 200, 250, 300, 450],
    'logistic__tol': [1e-4, 1e-3, 1e-2],
    'logistic__class_weight': [{0: 1, 1: 9}]
}


# Pipeline mit Skalierung und Modell
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logistic', LogisticRegression())
])
# Grid Search durchführen
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=sf, scoring=fbeta_scorer, n_jobs=-1, verbose=1)
grid_search.fit(X, y)

# Beste Parameter und Modellleistung anzeigen
print("Beste Parameter:", grid_search.best_params_)
print("Beste Modellleistung (F-beta-Score):", grid_search.best_score_)

# Ergebnisse der Grid Search visualisieren
results = pd.DataFrame(grid_search.cv_results_)
results = results.sort_values(by='rank_test_score')

print(results[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']])


# Fbeta-Scores plotten
#plt.figure(figsize=(10, 6))
#plt.plot(logistic__C, logReg_logistic__C, marker='o', label='Logistic Regression')
#plt.title('Fbeta-Scores für verschiedene C-Werte der Logistischen Regression')
#plt.xlabel('weights')
#plt.ylabel('Fbeta-Score')
#plt.legend()
#plt.grid(True)
#plt.show()
