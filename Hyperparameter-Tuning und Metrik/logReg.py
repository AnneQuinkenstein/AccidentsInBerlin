import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
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
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Definieren des F-beta-Scores mit beta = 2
beta = 3
fbeta_scorer = make_scorer(fbeta_score, beta=beta)

# Definieren des Parametergrids für die Grid Search
param_grid = {
    'logistic__C': [0.01, 0.1, 1, 10, 100],
    'logistic__penalty': ['l2'],
    'logistic__solver': ['lbfgs'],
    'logistic__max_iter': [500, 1000, 2000],
    'logistic__tol': [1e-4, 1e-3, 1e-2],
    'logistic__class_weight': [{0: 1, 1: 9}]
}

logistic__C = [0.001, 0.01, 0.1, 1, 10, 100]
logistic__max_iter = [100, 1000, 10000]

logReg_logistic__C = []
logReg_max_iter = []

for C_value in logistic__C:
    log_reg = LogisticRegression(C=C_value, max_iter=1000, class_weight={0: 1, 1: 9})
    fbeta_reg = cross_val_score(log_reg, X, y, cv=kf, scoring=fbeta_scorer)
    print(f"C_value: {C_value} Fbeta Score Logistische Regression (k-fold): {fbeta_reg.mean()}")
    logReg_logistic__C.append(fbeta_reg.mean())
    

# Fbeta-Scores plotten
plt.figure(figsize=(10, 6))
plt.plot(logistic__C, logReg_logistic__C, marker='o', label='Logistic Regression')
plt.title('Fbeta-Scores für verschiedene C-Werte der Logistischen Regression')
plt.xlabel('Stärke der Regularisierung (C)')
plt.ylabel('Fbeta-Score')
plt.legend()
plt.grid(True)
plt.show()

for max_iter_value in logistic__max_iter:
    log_reg = LogisticRegression(C=1, max_iter=max_iter_value, class_weight={0: 1, 1: 9})
    fbeta_reg = cross_val_score(log_reg, X, y, cv=kf, scoring=fbeta_scorer)
    print(f"max_iter_value: {max_iter_value} Fbeta Score Logistische Regression (k-fold): {fbeta_reg.mean()}")
    logReg_max_iter.append(fbeta_reg.mean())

# Fbeta-Scores plotten
plt.figure(figsize=(10, 6))
plt.plot(logistic__max_iter, logReg_max_iter, marker='o', label='Logistic Regression')
plt.title('Fbeta-Scores für verschiedene max_iter')
plt.xlabel('max_iter')
plt.ylabel('Fbeta-Score')
plt.legend()
plt.grid(True)
plt.show()
