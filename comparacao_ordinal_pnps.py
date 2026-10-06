"""Comparacao pareada de logit ordinal e multinomial na simulacao original.

Execute na mesma pasta do arquivo simulacao_pnps.py. Nenhum dado real e usado.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from statsmodels.miscmodels.ordinal_model import OrderedModel

ORDEM = np.array(["detrator", "passivo", "promotor"])
CAMINHO_ORIGINAL = Path(__file__).with_name("simulacao_pnps.py")


def preparar_dados():
    # Reutiliza, sem alterar, o gerador, as features e a particao de teste.
    fonte = CAMINHO_ORIGINAL.read_text(encoding="utf-8")
    marcador = "# 6b. Random Forest ORIGINAL"
    assert fonte.count(marcador) == 1
    escopo = {"__name__": "simulacao_original_parcial"}
    exec(compile(fonte.split(marcador)[0], str(CAMINHO_ORIGINAL), "exec"), escopo)
    return escopo


class OrdinalLogit(BaseEstimator, ClassifierMixin):
    def __init__(self, maxiter=200):
        self.maxiter = maxiter

    def fit(self, X, y):
        self.classes_ = ORDEM.copy()
        self.scaler_ = StandardScaler().fit(X)
        X_scaled = self.scaler_.transform(X)
        indices = pd.Categorical(y, categories=list(self.classes_), ordered=True).codes
        if np.any(indices < 0):
            raise ValueError("Classe desconhecida no alvo ordinal")
        self.result_ = OrderedModel(indices, X_scaled, distr="logit").fit(
            method="bfgs", disp=False, maxiter=self.maxiter
        )
        if not self.result_.mle_retvals.get("converged", False):
            raise RuntimeError("O ajuste ordinal nao convergiu")
        return self

    def predict_proba(self, X):
        proba = np.asarray(self.result_.model.predict(
            self.result_.params, exog=self.scaler_.transform(X)))
        return proba

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def metricas(modelo, X, y):
    pred = modelo.predict(X)
    proba = modelo.predict_proba(X)
    ordem_modelo = list(modelo.classes_)
    proba = proba[:, [ordem_modelo.index(c) for c in ORDEM]]
    return {
        "acuracia": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro"),
        "auc_roc_macro": roc_auc_score(label_binarize(y, classes=ORDEM), proba, average="macro"),
        "recall_detrator": np.mean(pred[y == "detrator"] == "detrator"),
    }


def main():
    d = preparar_dados()
    X, y = d["X"], d["y"]
    X_train, X_test, y_train, y_test = [d[k] for k in ("X_train", "X_test", "y_train", "y_test")]
    assert np.array_equal(ORDEM, np.array(["detrator", "passivo", "promotor"]))
    modelos = {
        "Multinomial": LogisticRegression(max_iter=2000, random_state=d["SEED"]),
        "Ordinal": OrdinalLogit(),
    }
    rows = []
    for nome, modelo in modelos.items():
        if nome == "Multinomial":
            # O pipeline original foi ajustado no mesmo treino/teste.
            modelo = d["modelo_logistico"]
        else:
            modelo.fit(X_train, y_train)
        m = metricas(modelo, X_test, y_test)
        rows.append({"modelo": nome, **m})
        print(f"\n{nome}, conjunto de teste:\n{classification_report(y_test, modelo.predict(X_test), digits=3)}")
    teste = pd.DataFrame(rows)
    print("\nComparacao no mesmo conjunto de teste:\n", teste.to_string(index=False))
    teste.to_csv(Path(__file__).with_name("resultados_csv") / "comparacao_ordinal_teste.csv", index=False)

    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=d["SEED"]).split(X, y))
    cv_rows = []
    for i, (tr, te) in enumerate(folds, start=1):
        for nome in modelos:
            if nome == "Multinomial":
                scaler = StandardScaler().fit(X[tr])
                m = LogisticRegression(max_iter=2000, random_state=d["SEED"])
                m.fit(scaler.transform(X[tr]), y[tr])
                met = metricas(m, scaler.transform(X[te]), y[te])
            else:
                m = OrdinalLogit().fit(X[tr], y[tr])
                met = metricas(m, X[te], y[te])
            cv_rows.append({"fold": i, "modelo": nome, **met})
        print(f"Fold {i}/5 concluido", flush=True)
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(Path(__file__).with_name("resultados_csv") / "comparacao_ordinal_cv.csv", index=False)
    print("\nMedias CV:\n", cv.groupby("modelo").mean(numeric_only=True).drop(columns="fold").to_string())

    pivot = cv.pivot(index="fold", columns="modelo", values="f1_macro")
    diff = (pivot["Ordinal"] - pivot["Multinomial"]).to_numpy()
    media, var = diff.mean(), diff.var(ddof=1)
    razao = np.mean([len(te) / len(tr) for tr, te in folds])
    se = np.sqrt((1 / len(diff) + razao) * var)
    gl = len(diff) - 1
    t = media / se if se > 0 else np.nan
    p = 2 * stats.t.sf(abs(t), gl) if np.isfinite(t) else np.nan
    intervalo = media + np.array([-1, 1]) * stats.t.ppf(.975, gl) * se
    print(f"\nNadeau-Bengio, F1-macro ordinal menos multinomial: "
          f"{media:.4f}; IC95% [{intervalo[0]:.4f}, {intervalo[1]:.4f}]; p={p:.4f}")
    pd.DataFrame([{"diferenca_f1_macro": media, "ic95_inferior": intervalo[0],
                   "ic95_superior": intervalo[1], "p_nb": p}]).to_csv(
                       Path(__file__).with_name("resultados_csv") / "comparacao_ordinal_nadeau_bengio.csv",
                       index=False)


if __name__ == "__main__":
    main()
