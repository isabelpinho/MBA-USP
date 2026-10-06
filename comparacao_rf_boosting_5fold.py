"""Busca de RF e boosting usando exclusivamente o treino da simulação."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split


def metricas(nome, modelo, X, y):
    p = modelo.predict(X)
    ordem = {"detrator": 0, "passivo": 1, "promotor": 2}
    d = np.abs(np.array([ordem[c] for c in y]) - np.array([ordem[c] for c in p]))
    return {"modelo": nome, "acuracia": accuracy_score(y, p),
            "balanced_accuracy": balanced_accuracy_score(y, p),
            "f1_macro": f1_score(y, p, average="macro"),
            "recall_detrator": recall_score(y, p, labels=["detrator"], average="macro"),
            "mae_ordinal": d.mean(), "erros_opostos": int((d == 2).sum())}


def main():
    df = pd.read_csv(Path(__file__).with_name("resultados_csv") / "dataset_simulado_pnps.csv")
    # Somente os 24 drivers secundários e os quatro sinais adicionais.
    cols = list(df.columns[:24]) + ["engajamento", "recencia_dias", "frequencia_eventos_30d", "tendencia"]
    assert len(cols) == 28 and "latent_score" not in cols and "nota_nps_simulada" not in cols
    X = df[cols].to_numpy()
    y = df["classe"].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.3, stratify=y, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    busca_rf = RandomizedSearchCV(
        RandomForestClassifier(n_estimators=100, max_features="sqrt", random_state=42, n_jobs=-1),
        {"max_depth": [8, 12, 16, 24, None], "min_samples_leaf": [5, 10, 20, 40]},
        n_iter=12, scoring="f1_macro", cv=cv, random_state=42, n_jobs=1, refit=True
    )
    busca_rf.fit(X_tr, y_tr)
    print("RF, melhor configuração:", busca_rf.best_params_,
          "F1 interno:", round(busca_rf.best_score_, 4), flush=True)

    busca_boost = GridSearchCV(
        HistGradientBoostingClassifier(max_iter=200, random_state=42,
                                       early_stopping=False, l2_regularization=.1),
        {"learning_rate": [.05, .1], "max_leaf_nodes": [15, 31],
         "min_samples_leaf": [20, 50]},
        scoring="f1_macro", cv=cv, n_jobs=1, refit=True
    )
    busca_boost.fit(X_tr, y_tr)
    print("Boosting, melhor configuração:", busca_boost.best_params_,
          "F1 interno:", round(busca_boost.best_score_, 4), flush=True)

    rf_antigo = RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                        max_depth=12, min_samples_leaf=20,
                                        random_state=42, n_jobs=-1).fit(X_tr, y_tr)
    rf_novo = RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                     random_state=42, n_jobs=-1,
                                     **busca_rf.best_params_).fit(X_tr, y_tr)
    rows = [metricas("RF anterior", rf_antigo, X_te, y_te),
            metricas("RF grade ampliada", rf_novo, X_te, y_te),
            metricas("Gradient Boosting", busca_boost.best_estimator_, X_te, y_te)]
    tabela = pd.DataFrame(rows)
    tabela.to_csv(Path(__file__).with_name("resultados_csv") / "comparacao_rf_boosting_teste.csv", index=False)
    print("\nAvaliação exploratória no teste já utilizado:\n", tabela.to_string(index=False), flush=True)
    detalhes = pd.DataFrame([
        {"modelo": "RF grade ampliada", "f1_cv_interna": busca_rf.best_score_,
         **busca_rf.best_params_},
        {"modelo": "Gradient Boosting", "f1_cv_interna": busca_boost.best_score_,
         **busca_boost.best_params_},
    ])
    detalhes.to_csv(Path(__file__).with_name("resultados_csv") / "busca_rf_boosting_cv.csv", index=False)


if __name__ == "__main__":
    main()
