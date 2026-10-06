import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, cross_val_predict
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, classification_report,
    roc_auc_score, confusion_matrix
)
from sklearn.inspection import permutation_importance
from sklearn.base import clone

# --------------------------------------------------------------------------
# 1. Parametros da simulacao
# --------------------------------------------------------------------------
N = 25000
SEED = 42
np.random.seed(SEED)

DIMENSOES = [
    "entrega_logistica",
    "atendimento_cliente",
    "pagamento_credito",
    "programa_fidelidade",
    "preco_promocoes",
    "experiencia_app",
    "time_comercial",
    "sortimento",
]

LABELS_PT = {
    "entrega_logistica": "Entrega e logística",
    "atendimento_cliente": "Atendimento ao cliente",
    "pagamento_credito": "Pagamento e crédito",
    "programa_fidelidade": "Programa de fidelidade",
    "preco_promocoes": "Preço e promoções",
    "experiencia_app": "Experiência do aplicativo",
    "time_comercial": "Time comercial",
    "sortimento": "Sortimento",
}

SECUNDARIOS = {
    "entrega_logistica": ["prazo_entrega", "integridade_pedido", "rastreabilidade"],
    "atendimento_cliente": ["tempo_resposta", "resolucao_primeiro_contato", "qualidade_atendimento"],
    "pagamento_credito": ["facilidade_pagamento", "condicoes_credito", "clareza_cobranca"],
    "programa_fidelidade": ["atratividade_beneficios", "facilidade_resgate", "comunicacao_programa"],
    "preco_promocoes": ["percepcao_justica_preco", "relevancia_promocoes", "transparencia_precificacao"],
    "experiencia_app": ["usabilidade_app", "estabilidade_tecnica", "velocidade_app"],
    "time_comercial": ["consultoria_tecnica", "disponibilidade_vendedor", "confiabilidade_compromissos"],
    "sortimento": ["amplitude_catalogo", "disponibilidade_estoque", "relevancia_sortimento"],
}

LABELS_SECUNDARIOS_PT = {
    "prazo_entrega": "Prazo de entrega", "integridade_pedido": "Integridade do pedido",
    "rastreabilidade": "Rastreabilidade",
    "tempo_resposta": "Tempo de resposta", "resolucao_primeiro_contato": "Resolução no 1º contato",
    "qualidade_atendimento": "Qualidade do atendimento",
    "facilidade_pagamento": "Facilidade de pagamento", "condicoes_credito": "Condições de crédito",
    "clareza_cobranca": "Clareza da cobrança",
    "atratividade_beneficios": "Atratividade dos benefícios", "facilidade_resgate": "Facilidade de resgate",
    "comunicacao_programa": "Comunicação do programa",
    "percepcao_justica_preco": "Percepção de justiça do preço", "relevancia_promocoes": "Relevância das promoções",
    "transparencia_precificacao": "Transparência da precificação",
    "usabilidade_app": "Usabilidade do app", "estabilidade_tecnica": "Estabilidade técnica",
    "velocidade_app": "Velocidade do app",
    "consultoria_tecnica": "Consultoria técnica", "disponibilidade_vendedor": "Disponibilidade do vendedor",
    "confiabilidade_compromissos": "Confiabilidade dos compromissos",
    "amplitude_catalogo": "Amplitude do catálogo", "disponibilidade_estoque": "Disponibilidade de estoque",
    "relevancia_sortimento": "Relevância do sortimento",
}

SECUNDARIO_COLS = [s for d in DIMENSOES for s in SECUNDARIOS[d]]
SECUNDARIO_PARA_PRIMARIO = {s: d for d in DIMENSOES for s in SECUNDARIOS[d]}

PESOS_NEG = {
    "entrega_logistica": 0.290,
    "time_comercial": 0.230,
    "preco_promocoes": 0.137,
    "pagamento_credito": 0.114,
    "sortimento": 0.091,
    "atendimento_cliente": 0.069,
    "experiencia_app": 0.046,
    "programa_fidelidade": 0.023,
}

PESOS_POS = {
    "programa_fidelidade": 0.222,
    "experiencia_app": 0.185,
    "time_comercial": 0.155,
    "sortimento": 0.139,
    "atendimento_cliente": 0.110,
    "pagamento_credito": 0.085,
    "preco_promocoes": 0.062,
    "entrega_logistica": 0.042,
}

assert abs(sum(PESOS_NEG.values()) - 1.0) < 1e-9, "PESOS_NEG deve somar 1,000"
assert abs(sum(PESOS_POS.values()) - 1.0) < 1e-9, "PESOS_POS deve somar 1,000"
assert all(len(v) == 3 for v in SECUNDARIOS.values()), "cada dimensao deve ter 3 drivers secundarios"
assert len(SECUNDARIO_COLS) == 24, "esperado 24 drivers secundarios no total"

PESOS_SEC_NEG = {
    "entrega_logistica": {"prazo_entrega": 0.35, "integridade_pedido": 0.45, "rastreabilidade": 0.20},
    "atendimento_cliente": {"tempo_resposta": 0.30, "resolucao_primeiro_contato": 0.45, "qualidade_atendimento": 0.25},
    "pagamento_credito": {"facilidade_pagamento": 0.25, "condicoes_credito": 0.30, "clareza_cobranca": 0.45},
    "programa_fidelidade": {"atratividade_beneficios": 0.28, "facilidade_resgate": 0.40, "comunicacao_programa": 0.32},
    "preco_promocoes": {"percepcao_justica_preco": 0.35, "relevancia_promocoes": 0.23, "transparencia_precificacao": 0.42},
    "experiencia_app": {"usabilidade_app": 0.25, "estabilidade_tecnica": 0.45, "velocidade_app": 0.30},
    "time_comercial": {"consultoria_tecnica": 0.25, "disponibilidade_vendedor": 0.30, "confiabilidade_compromissos": 0.45},
    "sortimento": {"amplitude_catalogo": 0.25, "disponibilidade_estoque": 0.45, "relevancia_sortimento": 0.30},
}

PESOS_SEC_POS = {
    "entrega_logistica": {"prazo_entrega": 0.45, "integridade_pedido": 0.35, "rastreabilidade": 0.20},
    "atendimento_cliente": {"tempo_resposta": 0.25, "resolucao_primeiro_contato": 0.35, "qualidade_atendimento": 0.40},
    "pagamento_credito": {"facilidade_pagamento": 0.35, "condicoes_credito": 0.40, "clareza_cobranca": 0.25},
    "programa_fidelidade": {"atratividade_beneficios": 0.45, "facilidade_resgate": 0.30, "comunicacao_programa": 0.25},
    "preco_promocoes": {"percepcao_justica_preco": 0.40, "relevancia_promocoes": 0.35, "transparencia_precificacao": 0.25},
    "experiencia_app": {"usabilidade_app": 0.42, "estabilidade_tecnica": 0.25, "velocidade_app": 0.33},
    "time_comercial": {"consultoria_tecnica": 0.42, "disponibilidade_vendedor": 0.25, "confiabilidade_compromissos": 0.33},
    "sortimento": {"amplitude_catalogo": 0.35, "disponibilidade_estoque": 0.25, "relevancia_sortimento": 0.40},
}

for _d in DIMENSOES:
    assert abs(sum(PESOS_SEC_NEG[_d].values()) - 1.0) < 1e-9, f"PESOS_SEC_NEG[{_d}] deve somar 1,000"
    assert abs(sum(PESOS_SEC_POS[_d].values()) - 1.0) < 1e-9, f"PESOS_SEC_POS[{_d}] deve somar 1,000"
    assert set(PESOS_SEC_NEG[_d]) == set(SECUNDARIOS[_d]) == set(PESOS_SEC_POS[_d]), f"drivers de {_d} inconsistentes"

PESO_DIMENSOES = 0.35
PESO_ENGAJAMENTO = 0.25
PESO_FREQUENCIA = 0.20
PESO_RECENCIA = 0.13
PESO_TENDENCIA = 0.07
assert abs((PESO_DIMENSOES + PESO_ENGAJAMENTO + PESO_RECENCIA
            + PESO_FREQUENCIA + PESO_TENDENCIA) - 1.0) < 1e-9, \
    "pesos de bloco (PESO_DIMENSOES..PESO_TENDENCIA) devem somar 1,000"

DISTRIBUICAO_ALVO_NOTA = {
    0: 0.003, 1: 0.005, 2: 0.010, 3: 0.020, 4: 0.040, 5: 0.090, 6: 0.162,
    7: 0.210, 8: 0.200,
    9: 0.150, 10: 0.110,
}
assert abs(sum(DISTRIBUICAO_ALVO_NOTA.values()) - 1.0) < 1e-9, "DISTRIBUICAO_ALVO_NOTA deve somar 1,000"

# --------------------------------------------------------------------------
# 2. Geracao das variaveis preditoras (sinais)
# --------------------------------------------------------------------------
sub_scores = {s: np.random.beta(2, 2, size=N) for s in SECUNDARIO_COLS}

dim_scores = {}
for d in DIMENSOES:
    sec_contrib = np.zeros(N)
    for s in SECUNDARIOS[d]:
        sub_score = sub_scores[s]
        sub_surplus = np.clip(sub_score - 0.5, 0, None)
        sub_shortfall = np.clip(0.5 - sub_score, 0, None)
        sec_contrib += PESOS_SEC_POS[d][s] * sub_surplus - PESOS_SEC_NEG[d][s] * sub_shortfall
    dim_scores[d] = np.clip(0.5 + sec_contrib, 0, 1)

engajamento = np.random.beta(2, 2, size=N)

recencia_dias = np.random.exponential(scale=15, size=N)
frequencia_eventos_30d = np.random.poisson(lam=1.5, size=N)
tendencia = np.random.normal(0, 0.05, size=N)

recencia_score = 1 - np.exp(-recencia_dias / 10)
frequencia_norm = frequencia_eventos_30d / max(frequencia_eventos_30d.max(), 1)

receita_anual = np.random.lognormal(mean=10.0, sigma=0.9, size=N)
crescimento_receita = np.random.normal(0.05, 0.15, size=N)
margem_bruta = np.random.beta(5, 5, size=N)

receita_pct = pd.Series(receita_anual).rank(pct=True).values
crescimento_pct = pd.Series(crescimento_receita).rank(pct=True).values
margem_pct = pd.Series(margem_bruta).rank(pct=True).values

valor_cliente_score = 0.6 * receita_pct + 0.2 * crescimento_pct + 0.2 * margem_pct
valor_cliente_mediana = np.median(valor_cliente_score)
valor_cliente_categoria = np.where(valor_cliente_score >= valor_cliente_mediana, "Alto", "Baixo")

# --------------------------------------------------------------------------
# 3. Variavel latente de satisfacao = combinacao ponderada (assimetrica) + ruido
# --------------------------------------------------------------------------
dimensao_contrib = np.zeros(N)
for d in DIMENSOES:
    score = dim_scores[d]
    surplus = np.clip(score - 0.5, 0, None)
    shortfall = np.clip(0.5 - score, 0, None)
    dimensao_contrib += PESOS_POS[d] * surplus - PESOS_NEG[d] * shortfall

dimensao_score_ponderado = np.clip(0.5 + dimensao_contrib, 0, 1)

noise = np.random.normal(0, 0.08, size=N)

latent = (
    PESO_DIMENSOES * dimensao_score_ponderado
    + PESO_ENGAJAMENTO * engajamento
    + PESO_RECENCIA * recencia_score
    - PESO_FREQUENCIA * frequencia_norm
    + PESO_TENDENCIA * tendencia
    + noise
)
latent = np.clip(latent, 0, 1)

percentil_latente = pd.Series(latent).rank(pct=True, method="first").values

_notas_ordenadas = sorted(DISTRIBUICAO_ALVO_NOTA.keys())
_cum = 0.0
_cortes_cumulativos = []
for _n in _notas_ordenadas:
    _cum += DISTRIBUICAO_ALVO_NOTA[_n]
    _cortes_cumulativos.append(_cum)
_cortes_cumulativos[-1] = 1.0 + 1e-9

nota_nps_simulada = np.digitize(percentil_latente, _cortes_cumulativos[:-1], right=True)
nota_nps_simulada = np.array(_notas_ordenadas)[nota_nps_simulada]

std_camada3_secundario = float(np.mean([sub_scores[s].std() for s in SECUNDARIO_COLS]))
std_camada2_primario = float(np.mean([dim_scores[d].std() for d in DIMENSOES]))
std_camada1_total = float(dimensao_score_ponderado.std())

precisao_por_camada = pd.DataFrame({
    "camada": ["Camada 3 - driver secundário (individual)",
               "Camada 2 - dimensão primária (combinação ponderada de 3 secundários)",
               "Camada 1 - escore agregado total (combinação das 8 dimensões)"],
    "n_componentes_agregados": [1, 3, 8],
    "desvio_padrao_medio": [round(std_camada3_secundario, 4),
                             round(std_camada2_primario, 4),
                             round(std_camada1_total, 4)],
})
print("Evidência de volumetria x dispersão (desvio-padrão cai com a agregação; NÃO é medida de acurácia):")
print(precisao_por_camada.to_string(index=False))
print()

classe = np.select(
    [nota_nps_simulada >= 9, nota_nps_simulada >= 7],
    ["promotor", "passivo"],
    default="detrator",
)

df = pd.DataFrame({**sub_scores, **dim_scores,
                    "engajamento": engajamento,
                    "recencia_dias": recencia_dias,
                    "frequencia_eventos_30d": frequencia_eventos_30d,
                    "tendencia": tendencia,
                    "receita_anual": receita_anual,
                    "crescimento_receita": crescimento_receita,
                    "margem_bruta": margem_bruta,
                    "valor_cliente_score": valor_cliente_score,
                    "valor_cliente": valor_cliente_categoria,
                    "latent_score": latent,
                    "nota_nps_simulada": nota_nps_simulada,
                    "classe": classe})

print("Distribuicao das classes simuladas (classe verdadeira):")
print(df["classe"].value_counts(normalize=True).round(3))
print()
print("Distribuicao de valor_cliente:")
print(df["valor_cliente"].value_counts(normalize=True).round(3))
print()
print(f"Correlacao valor_cliente_score x latent_score: {np.corrcoef(valor_cliente_score, latent)[0,1]:.3f} "
      "(esperado: baixa, por construcao)")
print()

share_true = df["classe"].value_counts(normalize=True)
pnps_verdadeiro = 100 * (share_true.get("promotor", 0) - share_true.get("detrator", 0))
print(f"pNPS agregado (classe verdadeira simulada): {pnps_verdadeiro:.1f}")
print()

# --------------------------------------------------------------------------
# 5. Sensibilidade à correlação geral entre os 24 drivers secundários
# --------------------------------------------------------------------------
from pathlib import Path
from scipy.stats import norm, beta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def gerar_sub_scores_correlacionados(rho, scores_base, fator_comum):
    if not 0 <= rho < 1:
        raise ValueError("rho deve estar no intervalo [0, 1).")
    eps = np.finfo(float).eps
    saida = {}
    for nome in SECUNDARIO_COLS:
        uniforme_original = beta.cdf(scores_base[nome], 2, 2)
        normal_propria = norm.ppf(np.clip(uniforme_original, eps, 1 - eps))
        normal_mista = np.sqrt(rho) * fator_comum + np.sqrt(1 - rho) * normal_propria
        uniforme_misto = norm.cdf(normal_mista)
        saida[nome] = beta.ppf(np.clip(uniforme_misto, eps, 1 - eps), 2, 2)
    return saida


def gerar_rotulos_e_x(scores):
    dimensoes = {}
    for d in DIMENSOES:
        contrib = np.zeros(N)
        for s in SECUNDARIOS[d]:
            valor = scores[s]
            contrib += (PESOS_SEC_POS[d][s] * np.clip(valor - .5, 0, None)
                        - PESOS_SEC_NEG[d][s] * np.clip(.5 - valor, 0, None))
        dimensoes[d] = np.clip(.5 + contrib, 0, 1)
    total = np.zeros(N)
    for d in DIMENSOES:
        valor = dimensoes[d]
        total += (PESOS_POS[d] * np.clip(valor - .5, 0, None)
                  - PESOS_NEG[d] * np.clip(.5 - valor, 0, None))
    escore_dimensoes = np.clip(.5 + total, 0, 1)
    latente = np.clip(
        PESO_DIMENSOES * escore_dimensoes
        + PESO_ENGAJAMENTO * engajamento
        + PESO_RECENCIA * recencia_score
        - PESO_FREQUENCIA * frequencia_norm
        + PESO_TENDENCIA * tendencia
        + noise, 0, 1
    )
    percentil = pd.Series(latente).rank(pct=True, method="first").to_numpy()
    notas = np.array(sorted(DISTRIBUICAO_ALVO_NOTA))
    cortes = np.cumsum([DISTRIBUICAO_ALVO_NOTA[n] for n in notas])
    notas_nps = notas[np.digitize(percentil, cortes[:-1], right=True)]
    classes = np.select([notas_nps >= 9, notas_nps >= 7],
                        ["promotor", "passivo"], default="detrator")
    colunas = SECUNDARIO_COLS + ["engajamento", "recencia_dias", "frequencia_eventos_30d", "tendencia"]
    sinais = {**scores, "engajamento": engajamento,
              "recencia_dias": recencia_dias,
              "frequencia_eventos_30d": frequencia_eventos_30d,
              "tendencia": tendencia}
    X = np.column_stack([sinais[c] for c in colunas])
    return X, classes


fator_insatisfacao = np.random.default_rng(SEED + 101).standard_normal(N)
fator_qualidade = -fator_insatisfacao
indices_treino, indices_teste = train_test_split(
    np.arange(N), test_size=.3, stratify=classe, random_state=SEED
)
linhas = []
for rho in [0.0, 0.3, 0.6]:
    drivers = gerar_sub_scores_correlacionados(rho, sub_scores, fator_qualidade)
    X_cenario, y_cenario = gerar_rotulos_e_x(drivers)
    if rho == 0:
        assert np.allclose(np.column_stack([drivers[c] for c in SECUNDARIO_COLS]),
                           np.column_stack([sub_scores[c] for c in SECUNDARIO_COLS]))
        assert np.array_equal(y_cenario, classe)
    modelo = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=20,
        max_features="sqrt", random_state=SEED, n_jobs=-1
    ).fit(X_cenario[indices_treino], y_cenario[indices_treino])
    previsto = modelo.predict(X_cenario[indices_teste])
    matriz = np.column_stack([drivers[c] for c in SECUNDARIO_COLS])
    corr = np.corrcoef(matriz, rowvar=False)
    media_fora_diagonal = corr[np.triu_indices_from(corr, k=1)].mean()
    linhas.append({
        "rho_normal_latente": rho,
        "correlacao_pearson_media_drivers": media_fora_diagonal,
        "media_marginal_drivers": matriz.mean(axis=0).mean(),
        "variancia_marginal_media": matriz.var(axis=0).mean(),
        "n_treino": len(indices_treino), "n_teste": len(indices_teste),
        "f1_macro_teste": f1_score(y_cenario[indices_teste], previsto, average="macro"),
        "balanced_accuracy_teste": balanced_accuracy_score(y_cenario[indices_teste], previsto),
        "recall_detrator_teste": np.mean(
            previsto[y_cenario[indices_teste] == "detrator"] == "detrator"
        ),
        "acuracia_teste": accuracy_score(y_cenario[indices_teste], previsto),
    })
    print(f"Correlação latente {rho:.1f}: "
          f"Pearson média observada {media_fora_diagonal:.3f}, "
          f"F1-macro {linhas[-1]['f1_macro_teste']:.4f}", flush=True)

resultado = pd.DataFrame(linhas)
resultado.to_csv(Path(__file__).with_name("resultados_csv") / "correlacao_drivers.csv", index=False)
print("\nResultados completos:\n", resultado.to_string(index=False), flush=True)
