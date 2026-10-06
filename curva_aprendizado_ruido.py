"""
Analise de sensibilidade ao ruido e ao tamanho do treino: reutiliza o
gerador de sinais de simulacao_pnps.py (dataset 100% simulado) e reajusta o
Random Forest para diferentes niveis de ruido da variavel latente e
diferentes tamanhos de treino, medindo o efeito no F1-macro.
"""

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
N = 25000          # tamanho da amostra simulada (alinhado ao volume mensal real de respostas de NPS)
SEED = 42
np.random.seed(SEED)

# Oito dimensoes de experiencia operacional (categorias genericas, nao proprietarias)
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

# Cada dimensao se desdobra em 3 drivers secundarios (Camada 3).
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

# Peso de cada dimensao quando o escore esta ABAIXO da media (soma = 1,000).
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

# Peso de cada dimensao quando o escore esta ACIMA da media (soma = 1,000;
# assimetrico em relacao aos pesos negativos).
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

# Pesos dos drivers secundarios dentro de cada dimensao. Cada dimensao soma
# 1,000.
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

PESO_DIMENSOES = 0.35    # bloco das 8 dimensoes operacionais (assimetrico, ver secao 3)
PESO_ENGAJAMENTO = 0.25  # sinal comportamental
PESO_FREQUENCIA = 0.20   # quantos eventos negativos nos ultimos 30 dias
PESO_RECENCIA = 0.13     # ha quanto tempo ocorreu o ultimo evento negativo
PESO_TENDENCIA = 0.07    # a satisfacao esta melhorando ou piorando
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
# 1. Geracao das variaveis preditoras (sinais), identica a simulacao_pnps.py
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
# 2. Variavel latente de satisfacao = combinacao ponderada (assimetrica) + ruido
# --------------------------------------------------------------------------
dimensao_contrib = np.zeros(N)
for d in DIMENSOES:
    score = dim_scores[d]
    surplus = np.clip(score - 0.5, 0, None)
    shortfall = np.clip(0.5 - score, 0, None)
    dimensao_contrib += PESOS_POS[d] * surplus - PESOS_NEG[d] * shortfall

dimensao_score_ponderado = np.clip(0.5 + dimensao_contrib, 0, 1)


# 3. Sensibilidade: mesma populacao de sinais, com ruido e tamanho de treino variaveis.
from pathlib import Path

X = np.column_stack([sub_scores[c] for c in SECUNDARIO_COLS] + [
    engajamento, recencia_dias, frequencia_eventos_30d, tendencia
])
z_ruido = np.random.normal(0, 1, size=N)
sinal_sem_ruido = (
    PESO_DIMENSOES * dimensao_score_ponderado
    + PESO_ENGAJAMENTO * engajamento
    + PESO_RECENCIA * recencia_score
    - PESO_FREQUENCIA * frequencia_norm
    + PESO_TENDENCIA * tendencia
)


def classes_para_ruido(desvio):
    latente = np.clip(sinal_sem_ruido + desvio * z_ruido, 0, 1)
    percentis = pd.Series(latente).rank(pct=True, method="first").to_numpy()
    notas = np.array(sorted(DISTRIBUICAO_ALVO_NOTA))
    cortes = np.cumsum([DISTRIBUICAO_ALVO_NOTA[n] for n in notas])
    notas_simuladas = notas[np.digitize(percentis, cortes[:-1], right=True)]
    return np.select(
        [notas_simuladas >= 9, notas_simuladas >= 7],
        ["promotor", "passivo"], default="detrator"
    )


def indices_aninhados(pool, classes, tamanho, seed):
    rng = np.random.default_rng(seed)
    blocos = {}
    for categoria in ["detrator", "passivo", "promotor"]:
        bloco = pool[classes[pool] == categoria].copy()
        rng.shuffle(bloco)
        blocos[categoria] = bloco
    proporcoes = np.array([len(blocos[c]) for c in blocos]) / len(pool)
    contagens = np.floor(proporcoes * tamanho).astype(int)
    sobras = tamanho - contagens.sum()
    for j in np.argsort(-(proporcoes * tamanho - contagens))[:sobras]:
        contagens[j] += 1
    return np.concatenate([blocos[c][:n] for c, n in zip(blocos, contagens)])


rotulos_base = classes_para_ruido(.08)
indices_treino_pool, indices_teste = train_test_split(
    np.arange(N), test_size=.3, stratify=rotulos_base, random_state=SEED
)
assert len(indices_treino_pool) == 17500 and len(indices_teste) == 7500
linhas = []
for desvio in [0.04, 0.08, 0.16]:
    classes = classes_para_ruido(desvio)
    for tamanho in [1000, 3000, 7500, 17500]:
        tr = indices_aninhados(indices_treino_pool, classes, tamanho, SEED)
        modelo = RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=20,
            max_features="sqrt", random_state=SEED, n_jobs=-1
        ).fit(X[tr], classes[tr])
        pred = modelo.predict(X[indices_teste])
        treino_pred = modelo.predict(X[tr])
        linhas.append({
            "desvio_ruido": desvio, "n_treino": tamanho,
            "n_teste": len(indices_teste),
            "f1_macro_teste": f1_score(classes[indices_teste], pred, average="macro"),
            "f1_macro_treino": f1_score(classes[tr], treino_pred, average="macro"),
            "balanced_accuracy_teste": balanced_accuracy_score(classes[indices_teste], pred),
            "recall_detrator_teste": np.mean(pred[classes[indices_teste] == "detrator"] == "detrator"),
            "acuracia_teste": accuracy_score(classes[indices_teste], pred),
        })
        print(f"Ruído={desvio:.2f}, treino={tamanho}, F1={linhas[-1]['f1_macro_teste']:.4f}", flush=True)
resultado = pd.DataFrame(linhas)
destino = Path(__file__).with_name("resultados_csv") / "curva_aprendizado_ruido.csv"
resultado.to_csv(destino, index=False)
print("\nResultados:\n", resultado.to_string(index=False), flush=True)
