"""Experimento: nota NPS 0-10 prevista e posteriormente classificada."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def classe_da_nota(notas):
    # Notas observadas sao inteiras: converter a previsao para a nota
    # discreta mais proxima equivale a fronteiras 6.5 e 8.5.
    notas = np.clip(np.asarray(notas), 0, 10)
    notas = np.floor(notas + .5)
    return np.select([notas >= 9, notas >= 7], ["promotor", "passivo"], default="detrator")


def resumo(nome, real_nota, real_classe, previsto_nota=None, previsto_classe=None):
    if previsto_classe is None:
        previsto_classe = classe_da_nota(previsto_nota)
    linha = {"modelo": nome,
             "acuracia": accuracy_score(real_classe, previsto_classe),
             "balanced_accuracy": balanced_accuracy_score(real_classe, previsto_classe),
             "f1_macro": f1_score(real_classe, previsto_classe, average="macro"),
             "recall_detrator": np.mean(np.asarray(previsto_classe)[np.asarray(real_classe) == "detrator"] == "detrator"),
             "promotores_previstos": int(np.sum(np.asarray(previsto_classe) == "promotor"))}
    if previsto_nota is not None:
        linha.update(mae_nota=mean_absolute_error(real_nota, previsto_nota),
                     rmse_nota=np.sqrt(mean_squared_error(real_nota, previsto_nota)))
    return linha


def main():
    caminho = Path(__file__).with_name("simulacao_pnps.py")
    codigo = caminho.read_text(encoding="utf-8")
    marcador = "# 7. Avaliacao no conjunto de teste"
    assert codigo.count(marcador) == 1
    d = {"__name__": "simulacao_original_parcial"}
    exec(compile(codigo.split(marcador)[0], str(caminho), "exec"), d)
    X, y = d["X"], d["y"]
    notas = d["df"]["nota_nps_simulada"].to_numpy()
    assert np.array_equal(classe_da_nota(notas), y)
    treino, teste = train_test_split(np.arange(len(y)), test_size=.3,
                                    stratify=y, random_state=d["SEED"])
    assert np.array_equal(X[treino], d["X_train"])
    assert np.array_equal(y[teste], d["y_test"])
    parametros = d["melhores_parametros_rf"]
    linear = make_pipeline(StandardScaler(), LinearRegression()).fit(X[treino], notas[treino])
    reg = RandomForestRegressor(n_estimators=300, random_state=d["SEED"],
                                n_jobs=-1, max_features="sqrt", **parametros).fit(X[treino], notas[treino])
    resultados = [
        resumo("Logistica multinomial", notas[teste], y[teste], previsto_classe=d["modelo_logistico"].predict(X[teste])),
        resumo("RF Classifier", notas[teste], y[teste], previsto_classe=d["modelo_rf"].predict(X[teste])),
        resumo("Linear > categoria", notas[teste], y[teste], previsto_nota=linear.predict(X[teste])),
        resumo("RF Regressor > categoria", notas[teste], y[teste], previsto_nota=reg.predict(X[teste])),
    ]
    pd.DataFrame(resultados).to_csv(Path(__file__).with_name("resultados_csv") / "comparacao_nota_continua_teste.csv", index=False)
    print("\nTeste:\n", pd.DataFrame(resultados).to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)

    linhas = []
    for i, (tr, te) in enumerate(StratifiedKFold(n_splits=5, shuffle=True, random_state=d["SEED"]).split(X, y), 1):
        clf = RandomForestClassifier(n_estimators=300, random_state=d["SEED"],
                                     n_jobs=-1, max_features="sqrt", **parametros).fit(X[tr], y[tr])
        regr = RandomForestRegressor(n_estimators=300, random_state=d["SEED"],
                                     n_jobs=-1, max_features="sqrt", **parametros).fit(X[tr], notas[tr])
        linhas.extend([{"fold": i, **resumo("RF Classifier", notas[te], y[te], previsto_classe=clf.predict(X[te]))},
                       {"fold": i, **resumo("RF Regressor > categoria", notas[te], y[te], previsto_nota=regr.predict(X[te]))}])
        print(f"Fold {i}/5 concluido", flush=True)
    cv = pd.DataFrame(linhas)
    cv.to_csv(Path(__file__).with_name("resultados_csv") / "comparacao_nota_continua_cv.csv", index=False)
    print("\nCV:\n", cv.groupby("modelo").mean(numeric_only=True).to_string(), flush=True)
    p = cv.pivot(index="fold", columns="modelo", values="f1_macro")
    diff = (p["RF Regressor > categoria"] - p["RF Classifier"]).to_numpy()
    se = np.sqrt((1 / 5 + 1 / 4) * np.var(diff, ddof=1))
    t = diff.mean() / se if se else np.nan
    valor_p = 2 * stats.t.sf(abs(t), 4) if np.isfinite(t) else np.nan
    ic = diff.mean() + np.array([-1, 1]) * stats.t.ppf(.975, 4) * se
    print(f"\nNadeau-Bengio F1 regressor - classifier: {diff.mean():.4f}, IC95% [{ic[0]:.4f}, {ic[1]:.4f}], p={valor_p:.4f}")


if __name__ == "__main__":
    main()
