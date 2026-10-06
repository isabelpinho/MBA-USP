"""Experimentos complementares: arquitetura de duas saidas (classificacao +
estimativa de nota, para ordenar gravidade entre detratores) e um
classificador Random Forest com 11 classes (notas de 0 a 10), cujas
probabilidades sao agregadas nas tres categorias de NPS.

Reaproveita a simulacao e os modelos principais de simulacao_pnps.py (exec
parcial, ate a secao de avaliacao), rederivando apenas os indices de
treino/teste (mesma seed e mesma estratificacao) para obter os vetores de
nota continua associados a cada observacao de teste.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error,
)
from sklearn.model_selection import train_test_split


def main():
    caminho = Path(__file__).with_name("simulacao_pnps.py")
    codigo = caminho.read_text(encoding="utf-8")
    marcador = "# 7. Avaliacao no conjunto de teste"
    assert codigo.count(marcador) == 1
    d = {"__name__": "simulacao_original_parcial"}
    exec(compile(codigo.split(marcador)[0], str(caminho), "exec"), d)
    X, y = d["X"], d["y"]
    notas = d["df"]["nota_nps_simulada"].to_numpy()
    treino, teste = train_test_split(np.arange(len(y)), test_size=.3,
                                      stratify=y, random_state=d["SEED"])
    assert np.array_equal(X[treino], d["X_train"])
    assert np.array_equal(y[teste], d["y_test"])
    X_train, X_test, y_test = d["X_train"], d["X_test"], d["y_test"]
    parametros = d["melhores_parametros_rf"]
    SEED = d["SEED"]

    # ------------------------------------------------------------------
    # Arquitetura de duas saidas: classificacao (modelo_rf) + estimativa
    # de nota (rf_regressor), usada para ordenar a gravidade entre os
    # clientes classificados como detratores.
    # ------------------------------------------------------------------
    rf_regressor = RandomForestRegressor(
        n_estimators=300, random_state=SEED, n_jobs=-1, max_features="sqrt",
        **parametros
    ).fit(X_train, notas[treino])
    nota_prevista_teste = rf_regressor.predict(X_test)
    classe_rf_teste = d["modelo_rf"].predict(X_test)
    severidade_linhas = []
    for grupo, mascara in [
        ("Detratores reais (diagnóstico)", y_test == "detrator"),
        ("Detratores previstos (uso operacional)", classe_rf_teste == "detrator"),
    ]:
        nota_real_grupo = notas[teste][mascara]
        nota_prevista_grupo = nota_prevista_teste[mascara]
        correlacao, valor_p = spearmanr(nota_real_grupo, nota_prevista_grupo)
        quantidade_priorizada = max(1, int(np.ceil(.1 * len(nota_real_grupo))))
        indices_priorizados = np.argsort(nota_prevista_grupo)[:quantidade_priorizada]
        severidade_linhas.append({
            "grupo": grupo,
            "n": len(nota_real_grupo),
            "spearman_notas": correlacao,
            "p_spearman_exploratorio": valor_p,
            "mae_nota": mean_absolute_error(nota_real_grupo, nota_prevista_grupo),
            "nota_media_grupo": np.mean(nota_real_grupo),
            "nota_media_10pct_priorizados": np.mean(nota_real_grupo[indices_priorizados]),
            "percentual_notas_0a2_grupo": np.mean(nota_real_grupo <= 2),
            "percentual_notas_0a2_10pct_priorizados": np.mean(nota_real_grupo[indices_priorizados] <= 2),
        })
    severidade_df = pd.DataFrame(severidade_linhas)
    severidade_df.to_csv(Path(__file__).with_name("resultados_csv") / "ordenacao_gravidade_detratores.csv", index=False)
    print("\nOrdenação exploratória por gravidade (duas saídas):\n",
          severidade_df.to_string(index=False), flush=True)

    # ------------------------------------------------------------------
    # Modelo unificado de 11 notas: probabilidades por nota (0-10)
    # agregadas nas tres categorias de NPS.
    # ------------------------------------------------------------------
    rf_11_notas = RandomForestClassifier(
        n_estimators=300, random_state=SEED, n_jobs=-1, max_features="sqrt",
        **parametros
    ).fit(X_train, notas[treino])
    proba_11 = rf_11_notas.predict_proba(X_test)
    notas_modelo = rf_11_notas.classes_
    proba_tres = np.column_stack([
        proba_11[:, notas_modelo <= 6].sum(axis=1),
        proba_11[:, (notas_modelo >= 7) & (notas_modelo <= 8)].sum(axis=1),
        proba_11[:, notas_modelo >= 9].sum(axis=1),
    ])
    assert np.allclose(proba_tres.sum(axis=1), 1)
    classe_11 = np.array(["detrator", "passivo", "promotor"])[proba_tres.argmax(axis=1)]
    nota_esperada_11 = proba_11 @ notas_modelo
    mascara_prevista_11 = classe_11 == "detrator"
    notas_reais_grupo_11 = notas[teste][mascara_prevista_11]
    notas_previstas_grupo_11 = nota_esperada_11[mascara_prevista_11]
    n_priorizados_11 = int(np.ceil(.1 * len(notas_reais_grupo_11)))
    priorizados_11 = np.argsort(notas_previstas_grupo_11)[:n_priorizados_11]
    resumo_11 = pd.DataFrame([{
        "modelo": "RF com 11 notas e agregação de probabilidades",
        "acuracia_tres_classes": accuracy_score(y_test, classe_11),
        "balanced_accuracy_tres_classes": balanced_accuracy_score(y_test, classe_11),
        "f1_macro_tres_classes": f1_score(y_test, classe_11, average="macro"),
        "recall_detrator": np.mean(classe_11[y_test == "detrator"] == "detrator"),
        "mae_nota_esperada": mean_absolute_error(notas[teste], nota_esperada_11),
        "acuracia_nota_exata": accuracy_score(
            notas[teste], notas_modelo[proba_11.argmax(axis=1)]
        ),
        "acuracia_nota_arredondada": accuracy_score(
            notas[teste], np.floor(np.clip(nota_esperada_11, 0, 10) + .5)
        ),
        "spearman_nota_geral": spearmanr(notas[teste], nota_esperada_11).statistic,
        "n_detratores_previstos": len(notas_reais_grupo_11),
        "spearman_nota_detratores_previstos": spearmanr(
            notas_reais_grupo_11, notas_previstas_grupo_11
        ).statistic,
        "taxa_0a2_detratores_previstos": np.mean(notas_reais_grupo_11 <= 2),
        "taxa_0a2_decil_priorizado": np.mean(notas_reais_grupo_11[priorizados_11] <= 2),
    }])
    resumo_11.to_csv(Path(__file__).with_name("resultados_csv") / "comparacao_modelo_11_notas.csv", index=False)
    pd.DataFrame({
        "nota_real": notas[teste],
        "nota_esperada": nota_esperada_11,
    }).groupby("nota_real").agg(
        clientes=("nota_esperada", "size"),
        nota_prevista_media=("nota_esperada", "mean"),
        nota_prevista_mediana=("nota_esperada", "median"),
    ).to_csv(Path(__file__).with_name("resultados_csv") / "discriminacao_por_nota_real.csv")
    print("\nModelo unificado de 11 notas:\n", resumo_11.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
