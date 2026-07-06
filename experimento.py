import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix, roc_curve)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(42)
N = 3000

# ---------------------------------------------------------------
# 1. Geração da base sintética de operações de crédito
# (simula uma carteira de uma cooperativa de crédito, com variáveis
#  tipicamente usadas em análise de risco/auditoria de crédito)
# ---------------------------------------------------------------
idade = RNG.normal(42, 12, N).clip(18, 80)
tempo_associado_anos = RNG.exponential(5, N).clip(0.1, 40)
renda_mensal = RNG.lognormal(mean=8.1, sigma=0.55, size=N).clip(1400, 60000)
valor_operacao = RNG.lognormal(mean=9.0, sigma=0.9, size=N).clip(500, 300000)
prazo_meses = RNG.integers(6, 84, N)
comprometimento_renda = (valor_operacao / (prazo_meses + 1)) / renda_mensal
comprometimento_renda = comprometimento_renda.clip(0.01, 1.5)
score_bureau = RNG.normal(650, 120, N).clip(300, 1000)
atrasos_12m = RNG.poisson(0.6, N)
possui_garantia = RNG.binomial(1, 0.4, N)
canal_digital = RNG.binomial(1, 0.55, N)
qtd_operacoes_ativas = RNG.poisson(2.2, N)

# variável latente de risco (combina fatores de forma realista, com ruído)
logit = (
    -3.0
    + 2.6 * comprometimento_renda
    - 0.0035 * (score_bureau - 650)
    + 0.55 * atrasos_12m
    - 0.05 * tempo_associado_anos
    - 0.35 * possui_garantia
    + 0.10 * qtd_operacoes_ativas
    + 0.000006 * valor_operacao
    + RNG.normal(0, 0.9, N)  # ruído / fatores não observados
)
prob_risco = 1 / (1 + np.exp(-logit))
risco_alto = RNG.binomial(1, prob_risco)

df = pd.DataFrame({
    "idade": idade.round(1),
    "tempo_associado_anos": tempo_associado_anos.round(2),
    "renda_mensal": renda_mensal.round(2),
    "valor_operacao": valor_operacao.round(2),
    "prazo_meses": prazo_meses,
    "comprometimento_renda": comprometimento_renda.round(3),
    "score_bureau": score_bureau.round(0),
    "atrasos_12m": atrasos_12m,
    "possui_garantia": possui_garantia,
    "canal_digital": canal_digital,
    "qtd_operacoes_ativas": qtd_operacoes_ativas,
    "risco_alto": risco_alto,
})

df.to_csv("/home/base_operacoes_credito.csv", index=False)
print("Base gerada:", df.shape)
print(df["risco_alto"].value_counts(normalize=True))
print(df.describe().T[["mean", "std", "min", "max"]])

# ---------------------------------------------------------------
# 2. Pré-processamento e divisão treino/teste
# ---------------------------------------------------------------
X = df.drop(columns=["risco_alto"])
y = df["risco_alto"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ---------------------------------------------------------------
# 3. Modelagem: 3 algoritmos comparados
# ---------------------------------------------------------------
models = {
    "Regressao Logistica": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

results = []
roc_data = {}
for name, model in models.items():
    if name == "Regressao Logistica":
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_proba = model.predict_proba(X_test_s)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    results.append({
        "Modelo": name, "Acuracia": acc, "Precisao": prec,
        "Recall": rec, "F1": f1, "AUC-ROC": auc
    })
    roc_data[name] = roc_curve(y_test, y_proba)
    print(f"\n=== {name} ===")
    print(f"Acuracia={acc:.3f} Precisao={prec:.3f} Recall={rec:.3f} F1={f1:.3f} AUC={auc:.3f}")
    print("Matriz de confusao:\n", cm)

results_df = pd.DataFrame(results).round(3)
results_df.to_csv("/home/resultados_modelos.csv", index=False)
print("\n\nTABELA FINAL:\n", results_df)

# Importancia de variaveis (Random Forest)
rf = models["Random Forest"]
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
importances.to_csv("/home/importancia_variaveis.csv")
print("\nImportancia das variaveis (Random Forest):\n", importances)

# ---------------------------------------------------------------
# 4. Graficos
# ---------------------------------------------------------------
plt.figure(figsize=(6, 5))
for name, (fpr, tpr, _) in roc_data.items():
    auc_val = results_df.loc[results_df.Modelo == name, "AUC-ROC"].values[0]
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
plt.xlabel("Taxa de Falsos Positivos")
plt.ylabel("Taxa de Verdadeiros Positivos")
plt.title("Curva ROC - Comparação dos Modelos")
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig("/home/curva_roc.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 5))
importances.sort_values().plot(kind="barh", color="#2E5C8A")
plt.xlabel("Importância relativa")
plt.title("Importância das Variáveis - Random Forest")
plt.tight_layout()
plt.savefig("/home/importancia_variaveis.png", dpi=150)
plt.close()

print("\nOK - graficos salvos")
