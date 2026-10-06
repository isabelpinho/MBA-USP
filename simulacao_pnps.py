"""
Simulacao de dados e modelagem preditiva de NPS em nivel de usuario (pNPS)
TCC - MBA em Data Science e Analytics (USP/Esalq) - Isabel Pinho

Dataset 100% SIMULADO (nenhum dado real de nenhuma empresa). As 8 dimensoes
operacionais e os 24 drivers secundarios sao categorias genericas de CX em
plataformas B2B, nao um recorte de nenhum sistema proprietario.

Os pesos por dimensao/driver (positivos e negativos) seguem a Teoria dos
Dois Fatores de Herzberg, Mausner e Snyderman (1959) e o modelo de Kano et
al. (1984): cada driver pode pesar de forma assimetrica entre desempenho
abaixo e acima da media. Isso gera 4 camadas de saida, da mais agregada a
mais granular:
    Camada 1 - classificacao pNPS agregada
    Camada 2 - 8 dimensoes primarias (combinacao ponderada de 3 secundarios)
    Camada 3 - 24 drivers secundarios (sinal bruto)
    Camada 4 - priorizacao por valor do cliente

IMPORTANTE: a classe (variavel-resposta) e gerada por uma formula definida
neste proprio script - e uma prova de conceito computacional, nao uma
validacao preditiva em dados reais (ver TCC, Metodologia e Conclusao).
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
# assimetrico em relacao aos pesos negativos, ver Herzberg/Kano acima).
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

# Pesos dos drivers secundarios dentro de cada dimensao (mesma logica
# assimetrica, um nivel abaixo). Cada dimensao soma 1,000.
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

# Peso de cada bloco de sinais na variavel latente.
PESO_DIMENSOES = 0.35    # bloco das 8 dimensoes operacionais (assimetrico, ver secao 3)
PESO_ENGAJAMENTO = 0.25  # sinal comportamental
PESO_FREQUENCIA = 0.20   # quantos eventos negativos nos ultimos 30 dias
PESO_RECENCIA = 0.13     # ha quanto tempo ocorreu o ultimo evento negativo
PESO_TENDENCIA = 0.07    # a satisfacao esta melhorando ou piorando
assert abs((PESO_DIMENSOES + PESO_ENGAJAMENTO + PESO_RECENCIA
            + PESO_FREQUENCIA + PESO_TENDENCIA) - 1.0) < 1e-9, \
    "pesos de bloco (PESO_DIMENSOES..PESO_TENDENCIA) devem somar 1,000"

# Distribuicao-alvo da nota (0-10), usada para mapear o ranking percentil do
# latente em nota discreta (secao 3c).
DISTRIBUICAO_ALVO_NOTA = {
    0: 0.003, 1: 0.005, 2: 0.010, 3: 0.020, 4: 0.040, 5: 0.090, 6: 0.162,
    7: 0.210, 8: 0.200,
    9: 0.150, 10: 0.110,
}
assert abs(sum(DISTRIBUICAO_ALVO_NOTA.values()) - 1.0) < 1e-9, "DISTRIBUICAO_ALVO_NOTA deve somar 1,000"

# --------------------------------------------------------------------------
# 2. Geracao das variaveis preditoras (sinais)
# --------------------------------------------------------------------------
# 2a. Escore [0,1] por driver secundario (Camada 3, Beta(2,2)). A dimensao
#     primaria (Camada 2) e a combinacao ponderada assimetrica (Herzberg/
#     Kano) dos seus 3 drivers, nao a media simples.
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

# 2b. Sinal comportamental (engajamento / uso / adocao de funcionalidades)
engajamento = np.random.beta(2, 2, size=N)

# 2c. Sinais temporais (recencia, frequencia, tendencia).
recencia_dias = np.random.exponential(scale=15, size=N)          # dias desde o ultimo evento negativo
frequencia_eventos_30d = np.random.poisson(lam=1.5, size=N)      # nº de eventos negativos nos ultimos 30 dias
tendencia = np.random.normal(0, 0.05, size=N)                    # positivo = melhorando, negativo = piorando

# Normalizacoes usadas so para construir a variavel latente; o modelo recebe
# frequencia_eventos_30d na escala bruta (sem vazamento de informacao).
recencia_score = 1 - np.exp(-recencia_dias / 10)                 # ~0 se evento negativo muito recente, ->1 se ha muito tempo
frequencia_norm = frequencia_eventos_30d / max(frequencia_eventos_30d.max(), 1)

# 2e. Indice de valor do cliente, gerado independente da satisfacao,
#     binarizado em Alto/Baixo por mediana.
receita_anual = np.random.lognormal(mean=10.0, sigma=0.9, size=N)      # distribuicao tipica de receita (assimetrica)
crescimento_receita = np.random.normal(0.05, 0.15, size=N)             # crescimento anual, media 5%, dp 15%
margem_bruta = np.random.beta(5, 5, size=N)                            # margem normalizada, concentrada perto de 0.5

# Normalizacao por percentil (robusta a outliers, especialmente na receita)
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

# --------------------------------------------------------------------------
# 3c. Nota de NPS simulada (0-10): mapeamento por quantil (quantile mapping)
#     do percentil do latente contra a distribuicao-alvo. A classe e
#     determinada pela nota, nas mesmas faixas do NPS tradicional (Figura 1):
#     9-10 promotor, 7-8 passivo, 0-6 detrator.
percentil_latente = pd.Series(latent).rank(pct=True, method="first").values

_notas_ordenadas = sorted(DISTRIBUICAO_ALVO_NOTA.keys())
_cum = 0.0
_cortes_cumulativos = []
for _n in _notas_ordenadas:
    _cum += DISTRIBUICAO_ALVO_NOTA[_n]
    _cortes_cumulativos.append(_cum)
_cortes_cumulativos[-1] = 1.0 + 1e-9  # garante que o maior percentil caia na ultima nota

nota_nps_simulada = np.digitize(percentil_latente, _cortes_cumulativos[:-1], right=True)
nota_nps_simulada = np.array(_notas_ordenadas)[nota_nps_simulada]

# --------------------------------------------------------------------------
# 3b. Dispersao (desvio-padrao) por camada de agregacao - NAO e medida de
#     acuracia preditiva (essa e feita separadamente na secao 7).
# --------------------------------------------------------------------------
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

# --------------------------------------------------------------------------
# 4. Classe VERDADEIRA (variavel-resposta), determinada pela nota de NPS
#    simulada (secao 3c). Nao confundir com a classe PREVISTA (secao 7c).
# --------------------------------------------------------------------------
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

# --------------------------------------------------------------------------
# pNPS agregado (classe verdadeira) = %promotores - %detratores.
# --------------------------------------------------------------------------
share_true = df["classe"].value_counts(normalize=True)
pnps_verdadeiro = 100 * (share_true.get("promotor", 0) - share_true.get("detrator", 0))
print(f"pNPS agregado (classe verdadeira simulada): {pnps_verdadeiro:.1f}")
print()

# --------------------------------------------------------------------------
# 6. Divisao treino/teste estratificada. O modelo usa os 24 drivers
#    secundarios (Camada 3); valor_cliente nao entra como feature (so na
#    priorizacao, Camada 4).
# --------------------------------------------------------------------------
feature_cols = SECUNDARIO_COLS + ["engajamento", "recencia_dias",
                                   "frequencia_eventos_30d", "tendencia"]
X = df[feature_cols].values
y = df["classe"].values
classes_ordenadas = sorted(df["classe"].unique())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=SEED
)

# 6a. Regressao logistica multinomial, com StandardScaler (features em
#     escalas muito diferentes).
modelo_logistico = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=2000, random_state=SEED)),
])
modelo_logistico.fit(X_train, y_train)

# 6b. Random Forest original (sem regularizacao) - usado so no diagnostico
#     de sobreajuste (secao 7a), nao e o modelo reportado no restante do
#     trabalho.
modelo_rf_original = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
modelo_rf_original.fit(X_train, y_train)

# 6c. Random Forest regularizado via grid search pequeno (2-fold CV) para
#     controlar o sobreajuste identificado no modelo original.
print("Buscando hiperparametros do Random Forest regularizado (grid search, 2-fold CV)...")
param_grid = {
    "max_depth": [4, 8, 12],
    "min_samples_leaf": [20, 50, 100],
}
grid_rf = GridSearchCV(
    RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1, max_features="sqrt"),
    param_grid=param_grid,
    cv=2,
    scoring="f1_macro",
    n_jobs=1,
)
grid_rf.fit(X_train, y_train)
melhores_parametros_rf = grid_rf.best_params_
print("Melhores hiperparametros encontrados:", melhores_parametros_rf)
print()

# Mantem os mesmos pesos de classe do RF original; a unica mudanca e a
# complexidade da arvore (max_depth / min_samples_leaf).
modelo_rf = RandomForestClassifier(
    n_estimators=300, random_state=SEED, n_jobs=-1, max_features="sqrt", **melhores_parametros_rf
)
modelo_rf.fit(X_train, y_train)

modelos = {
    "Regressao Logistica (multinomial)": modelo_logistico,
    "Random Forest": modelo_rf,
}

# --------------------------------------------------------------------------
# 7. Avaliacao no conjunto de teste: acuracia, balanced accuracy, F1 macro,
#    AUC-ROC (one-vs-rest), e baseline de classe majoritaria.
# --------------------------------------------------------------------------
y_test_bin = label_binarize(y_test, classes=classes_ordenadas)

# 7-baseline. Baseline trivial: classificador de classe majoritaria (a classe
#    mais frequente no TREINO, prevista para todo o conjunto de teste).
classe_majoritaria = pd.Series(y_train).mode()[0]
y_pred_baseline = np.full_like(y_test, fill_value=classe_majoritaria)
resultados = [{
    "modelo": "Baseline (classe majoritária)",
    "acuracia": round(accuracy_score(y_test, y_pred_baseline), 3),
    "balanced_accuracy": round(balanced_accuracy_score(y_test, y_pred_baseline), 3),
    "f1_macro": round(f1_score(y_test, y_pred_baseline, average="macro", zero_division=0), 3),
    "auc_roc_macro": 0.500,  # por definicao: classificador sem poder discriminatorio
}]

y_pred_por_modelo = {}  # guarda as predicoes no MESMO conjunto de teste, por
                         # modelo - usado no teste de McNemar (secao 7b-bis)
for nome, modelo in modelos.items():
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)
    y_pred_por_modelo[nome] = y_pred

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    auc = roc_auc_score(y_test_bin, y_proba, multi_class="ovr", average="macro")
    auc_por_classe = roc_auc_score(y_test_bin, y_proba, multi_class="ovr", average=None)

    resultados.append({"modelo": nome, "acuracia": round(acc, 3), "balanced_accuracy": round(bal_acc, 3),
                        "f1_macro": round(f1_macro, 3), "auc_roc_macro": round(auc, 3)})

    print(f"=== {nome} ===")
    print(classification_report(y_test, y_pred, digits=3))
    print("AUC-ROC por classe (one-vs-rest):",
          dict(zip(classes_ordenadas, np.round(auc_por_classe, 3))))
    print("Matriz de confusão (linhas = real, colunas = previsto), ordem", classes_ordenadas)
    print(confusion_matrix(y_test, y_pred, labels=classes_ordenadas))
    print()

resultados_df = pd.DataFrame(resultados)
print("Tabela-resumo (cole no TCC como Tabela 1, com baseline):")
print(resultados_df.to_string(index=False))
print()

# --------------------------------------------------------------------------
# 7a. Diagnostico de sobreajuste: acuracia treino vs. teste (RF original,
#     RF regularizado, regressao logistica).
# --------------------------------------------------------------------------
overfitting_rows = []
for nome, modelo in [
    ("Regressão logística", modelo_logistico),
    ("Random Forest (original, sem regularização)", modelo_rf_original),
    ("Random Forest (regularizado)", modelo_rf),
]:
    acc_treino = accuracy_score(y_train, modelo.predict(X_train))
    acc_teste = accuracy_score(y_test, modelo.predict(X_test))
    overfitting_rows.append({
        "modelo": nome,
        "acuracia_treino": round(acc_treino, 3),
        "acuracia_teste": round(acc_teste, 3),
        "diferenca": round(acc_treino - acc_teste, 3),
    })
overfitting_df = pd.DataFrame(overfitting_rows)
print("Diagnóstico de sobreajuste (acurácia treino vs. teste):")
print(overfitting_df.to_string(index=False))
print()

# --------------------------------------------------------------------------
# 7b. Validacao cruzada (5-fold estratificado) - estabilidade entre
#     particoes (nao substitui o diagnostico de sobreajuste da secao 7a).
# --------------------------------------------------------------------------
K_FOLDS = 5
skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

cv_resultados = []
cv_modelos_base = [
    ("Regressao Logistica (multinomial)", Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, random_state=SEED)),
    ])),
    ("Random Forest", RandomForestClassifier(
        n_estimators=300, random_state=SEED, n_jobs=-1, max_features="sqrt", **melhores_parametros_rf
    )),
]
f1_macro_per_fold = {}
_fold_sizes = []  # (n_train, n_test) por fold - usado na correcao de Nadeau-Bengio
for nome, modelo_base in cv_modelos_base:
    accs, f1s, aucs, bal_accs = [], [], [], []
    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        m = clone(modelo_base)
        m.fit(X_tr, y_tr)
        y_pred_fold = m.predict(X_te)
        y_proba_fold = m.predict_proba(X_te)
        y_te_bin = label_binarize(y_te, classes=classes_ordenadas)
        accs.append(accuracy_score(y_te, y_pred_fold))
        bal_accs.append(balanced_accuracy_score(y_te, y_pred_fold))
        f1s.append(f1_score(y_te, y_pred_fold, average="macro"))
        aucs.append(roc_auc_score(y_te_bin, y_proba_fold, multi_class="ovr", average="macro"))
        if len(_fold_sizes) < K_FOLDS:
            _fold_sizes.append((len(train_idx), len(test_idx)))
    f1_macro_per_fold[nome] = f1s
    cv_resultados.append({
        "modelo": nome,
        "acuracia_media": round(np.mean(accs), 3), "acuracia_dp": round(np.std(accs), 3),
        "balanced_acc_media": round(np.mean(bal_accs), 3), "balanced_acc_dp": round(np.std(bal_accs), 3),
        "f1_macro_media": round(np.mean(f1s), 3), "f1_macro_dp": round(np.std(f1s), 3),
        "auc_roc_media": round(np.mean(aucs), 3), "auc_roc_dp": round(np.std(aucs), 3),
    })

cv_df = pd.DataFrame(cv_resultados)
print(f"\nValidação cruzada ({K_FOLDS}-fold estratificado) - estabilidade entre partições (média +/- desvio-padrão):")
print(cv_df.to_string(index=False))
cv_df.to_csv("resultados_cv.csv", index=False)

# Teste t pareado e Wilcoxon (5 folds) sobre F1-macro, LR vs RF - mantidos
# so como referencia: em k-fold CV os folds tem treino sobreposto, o que
# subestima a variancia (Nadeau e Bengio, 2003). A inferencia usada na
# Discussao e o teste t corrigido (a) e o teste de McNemar (b) abaixo.
from scipy import stats as _stats
_nomes = list(f1_macro_per_fold.keys())
_f1_lr = f1_macro_per_fold[_nomes[0]]
_f1_rf = f1_macro_per_fold[_nomes[1]]
print("\nF1-macro por fold:")
print(f"  {_nomes[0]}: {[round(v,4) for v in _f1_lr]}")
print(f"  {_nomes[1]}: {[round(v,4) for v in _f1_rf]}")
_t_stat, _t_p = _stats.ttest_rel(_f1_lr, _f1_rf)
try:
    _w_stat, _w_p = _stats.wilcoxon(_f1_lr, _f1_rf)
except ValueError as e:
    _w_stat, _w_p = float("nan"), float("nan")
    print(f"  Wilcoxon nao computavel: {e}")
print(f"Teste t pareado NAO CORRIGIDO, so referencia (F1-macro, {_nomes[0]} vs {_nomes[1]}): t={_t_stat:.4f}, p={_t_p:.4f}")
print(f"Teste de Wilcoxon, so referencia (F1-macro, {_nomes[0]} vs {_nomes[1]}): W={_w_stat:.4f}, p={_w_p:.4f}")
pd.DataFrame({"fold": range(1, K_FOLDS+1), _nomes[0]: _f1_lr, _nomes[1]: _f1_rf}).to_csv("f1_macro_por_fold.csv", index=False)

# --- (a) Teste t corrigido para reamostragem (Nadeau & Bengio, 2003) -------
# var_corrigida = (1/k + n_teste/n_treino) * var(d), d_i = diferenca por
# fold entre os modelos. Graus de liberdade: k-1.
_n_train_medio = np.mean([s[0] for s in _fold_sizes])
_n_test_medio = np.mean([s[1] for s in _fold_sizes])
_d = np.array(_f1_lr) - np.array(_f1_rf)
_k = len(_d)
_mean_d = _d.mean()
_var_d = _d.var(ddof=1)
_var_corrigida = (1.0 / _k + _n_test_medio / _n_train_medio) * _var_d
_se_corrigido = np.sqrt(_var_corrigida)
_t_nb = _mean_d / _se_corrigido if _se_corrigido > 0 else 0.0
_df_nb = _k - 1
_p_nb = 2 * (1 - _stats.t.cdf(abs(_t_nb), df=_df_nb))
_tcrit_nb = _stats.t.ppf(0.975, df=_df_nb)
_ci_low = _mean_d - _tcrit_nb * _se_corrigido
_ci_high = _mean_d + _tcrit_nb * _se_corrigido
print(f"\nTeste t corrigido para reamostragem (Nadeau-Bengio, 2003), F1-macro, "
      f"{_nomes[0]} - {_nomes[1]}:")
print(f"  diferenca media = {_mean_d:.4f}, IC 95% = [{_ci_low:.4f}, {_ci_high:.4f}]")
print(f"  t_corrigido = {_t_nb:.4f}, df = {_df_nb}, p = {_p_nb:.4f}")

# --- (b) Teste de McNemar no conjunto de teste unico (Dietterich, 1998) ----
# Compara os dois modelos nas mesmas observacoes de teste (sem folds
# sobrepostos). b = LR acertou e RF errou; c = LR errou e RF acertou.
_y_pred_lr_teste = y_pred_por_modelo[_nomes[0]]
_y_pred_rf_teste = y_pred_por_modelo[_nomes[1]]
_lr_certo = (_y_pred_lr_teste == y_test)
_rf_certo = (_y_pred_rf_teste == y_test)
_b = int(np.sum(_lr_certo & ~_rf_certo))   # LR certo, RF errado
_c = int(np.sum(~_lr_certo & _rf_certo))   # LR errado, RF certo
_mcnemar_n = _b + _c
if _mcnemar_n > 0:
    _mcnemar_res = _stats.binomtest(min(_b, _c), _mcnemar_n, 0.5, alternative="two-sided")
    _p_mcnemar = _mcnemar_res.pvalue
else:
    _p_mcnemar = float("nan")
_chi2_mcnemar = ((abs(_b - _c) - 1) ** 2) / _mcnemar_n if _mcnemar_n > 0 else float("nan")
print(f"\nTeste de McNemar (conjunto de teste unico, n={len(y_test)}), "
      f"{_nomes[0]} vs {_nomes[1]}:")
print(f"  discordantes: LR certo/RF errado (b) = {_b}, LR errado/RF certo (c) = {_c}")
print(f"  qui-quadrado com correcao de continuidade = {_chi2_mcnemar:.4f}")
print(f"  p (teste binomial exato) = {_p_mcnemar:.4f}")

pd.DataFrame([{
    "diferenca_media_f1_macro": round(_mean_d, 4), "ic95_low": round(_ci_low, 4),
    "ic95_high": round(_ci_high, 4), "t_corrigido_nb": round(_t_nb, 4), "df": _df_nb,
    "p_corrigido_nb": round(_p_nb, 4), "mcnemar_b": _b, "mcnemar_c": _c,
    "mcnemar_p": round(_p_mcnemar, 4),
    "t_nao_corrigido": round(_t_stat, 4), "p_nao_corrigido": round(_t_p, 4),
    "wilcoxon_p_nao_corrigido": round(_w_p, 4) if not np.isnan(_w_p) else np.nan,
}]).to_csv("resultados_significancia_corrigida.csv", index=False)
print()

# --------------------------------------------------------------------------
# 7c. Predicoes out-of-fold (cross_val_predict) da regressao logistica,
#     usadas nas Camadas 2, 3 e 4 abaixo (classe PREVISTA, nao a verdadeira).
# --------------------------------------------------------------------------
modelo_oof = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=2000, random_state=SEED)),
])
# Uma unica chamada de cross_val_predict evita rodar a CV duas vezes.
proba_oof = cross_val_predict(modelo_oof, X, y, cv=skf, method="predict_proba")
classes_oof_ordem = classes_ordenadas  # LogisticRegression ordena .classes_ alfabeticamente, igual classes_ordenadas
proba_oof_df = pd.DataFrame(proba_oof, columns=classes_oof_ordem, index=df.index)
df["classe_prevista_oof"] = proba_oof_df.idxmax(axis=1).values

acc_oof = accuracy_score(df["classe"], df["classe_prevista_oof"])
print(f"Acurácia das predições out-of-fold (whole-dataset, regressão logística): {acc_oof:.3f}")
print()

# pNPS agregado com classe PREVISTA (OOF) e com PROBABILIDADES previstas
share_oof = df["classe_prevista_oof"].value_counts(normalize=True)
pnps_previsto_oof = 100 * (share_oof.get("promotor", 0) - share_oof.get("detrator", 0))
pnps_esperado_prob = 100 * (proba_oof_df["promotor"].mean() - proba_oof_df["detrator"].mean())
print(f"pNPS agregado (classe prevista OOF): {pnps_previsto_oof:.1f}")
print(f"pNPS agregado (esperado, a partir das probabilidades OOF): {pnps_esperado_prob:.1f}")
print(f"pNPS agregado (classe verdadeira simulada, referência): {pnps_verdadeiro:.1f}")
pnps_agregado_df = pd.DataFrame({
    "metodo": ["Classe verdadeira (simulação)", "Classe prevista (out-of-fold)", "Esperado (probabilidades OOF)"],
    "pnps": [round(pnps_verdadeiro, 1), round(pnps_previsto_oof, 1), round(pnps_esperado_prob, 1)],
})
pnps_agregado_df.to_csv("pnps_agregado.csv", index=False)
print()

# --------------------------------------------------------------------------
# 5. Camada 2 (diagnostico por dimensao primaria), segmentado pela classe
#    PREVISTA out-of-fold (secao 7c).
# --------------------------------------------------------------------------
diagnostico_dimensoes_oof = df.groupby("classe_prevista_oof")[DIMENSOES].mean().round(3)
print("Média de cada dimensão operacional primária, por classe PREVISTA out-of-fold (Camada 2):")
print(diagnostico_dimensoes_oof)
print()
# Versao com a classe verdadeira, mantida so para comparacao.
diagnostico_dimensoes_true = df.groupby("classe")[DIMENSOES].mean().round(3)

# --------------------------------------------------------------------------
# 5a. Camada 3 (diagnostico por driver secundario), idem.
# --------------------------------------------------------------------------
diagnostico_secundarios_oof = df.groupby("classe_prevista_oof")[SECUNDARIO_COLS].mean().round(3)
print("Média de cada driver secundário, por classe PREVISTA out-of-fold (Camada 3, diagnóstico granular):")
print(diagnostico_secundarios_oof)
print()

# --------------------------------------------------------------------------
# 5b. Camada 4 (priorizacao por valor do cliente), classe prevista e
#     verdadeira:
#     Tier 1 = valor alto + detrator (prioridade maxima)
#     Tier 2 = valor alto + passivo/promotor
#     Tier 3 = valor baixo + detrator
#     Tier 4 = valor baixo + passivo/promotor
# --------------------------------------------------------------------------
def tier_priorizacao(valor, classe_prevista):
    if valor == "Alto" and classe_prevista == "detrator":
        return "Tier 1"
    if valor == "Alto":
        return "Tier 2"
    if valor == "Baixo" and classe_prevista == "detrator":
        return "Tier 3"
    return "Tier 4"

df["tier_previsto_oof"] = [tier_priorizacao(v, c) for v, c in zip(df["valor_cliente"], df["classe_prevista_oof"])]
df["tier_classe_verdadeira"] = [tier_priorizacao(v, c) for v, c in zip(df["valor_cliente"], df["classe"])]

tier_dist_oof = df["tier_previsto_oof"].value_counts(normalize=True).round(3).sort_index()
tier_dist_true = df["tier_classe_verdadeira"].value_counts(normalize=True).round(3).sort_index()
tier_comparacao = pd.DataFrame({
    "tier": tier_dist_true.index,
    "proporcao_classe_verdadeira": tier_dist_true.values,
    "proporcao_classe_prevista_oof": [tier_dist_oof.get(t, 0.0) for t in tier_dist_true.index],
})
tier_comparacao["diferenca"] = (tier_comparacao["proporcao_classe_prevista_oof"]
                                 - tier_comparacao["proporcao_classe_verdadeira"]).round(3)
print("Distribuição da matriz de priorização (Camada 4) - classe verdadeira vs. classe prevista OOF:")
print(tier_comparacao.to_string(index=False))
print()

tier_crosstab_oof = pd.crosstab(df["valor_cliente"], df["classe_prevista_oof"], normalize="all").round(3)
print("Cruzamento valor_cliente x classe prevista (OOF):")
print(tier_crosstab_oof)
print()

# --------------------------------------------------------------------------
# 8. Importancia das variaveis (RF regularizado): nativa (reducao de
#    impureza) e permutation importance (conjunto de teste). Camada 3:
#    por driver; Camada 2: soma por dimensao primaria.
# --------------------------------------------------------------------------
importancias_secundario = pd.Series(modelo_rf.feature_importances_, index=feature_cols).loc[SECUNDARIO_COLS] \
    .sort_values(ascending=False)
print("Importância relativa (nativa, redução de impureza) dos 24 drivers secundários (Random Forest, Camada 3):")
print(importancias_secundario.round(3))

importancias_primario = importancias_secundario.groupby(SECUNDARIO_PARA_PRIMARIO).sum().sort_values(ascending=False)
print("\nImportância relativa agregada por dimensão primária (Camada 2, soma dos 3 secundários):")
print(importancias_primario.round(3))
print()

print("Calculando permutation importance no conjunto de teste (n_repeats=10)...")
perm = permutation_importance(
    modelo_rf, X_test, y_test, scoring="f1_macro", n_repeats=10, random_state=SEED, n_jobs=-1
)
importancias_perm_secundario = pd.Series(perm.importances_mean, index=feature_cols).loc[SECUNDARIO_COLS] \
    .sort_values(ascending=False)
importancias_perm_primario = importancias_perm_secundario.groupby(SECUNDARIO_PARA_PRIMARIO).sum() \
    .sort_values(ascending=False)

# Desvio-padrao da permutation importance entre as 10 repeticoes. Para as
# dimensoes primarias, soma as importancias brutas por repeticao antes de
# calcular o desvio (a soma dos desvios individuais nao seria correta).
importancias_perm_secundario_std = pd.Series(perm.importances_std, index=feature_cols).loc[SECUNDARIO_COLS]
importancias_perm_raw = pd.DataFrame(perm.importances, index=feature_cols).loc[SECUNDARIO_COLS]
importancias_perm_raw_primario = importancias_perm_raw.groupby(SECUNDARIO_PARA_PRIMARIO).sum()
importancias_perm_primario_std = importancias_perm_raw_primario.std(axis=1, ddof=1).loc[importancias_perm_primario.index]

print("Permutation importance (queda de F1-macro) agregada por dimensão primária (Camada 2), média ± desvio-padrão (10 repetições):")
for _dim in importancias_perm_primario.index:
    print(f"  {_dim}: {importancias_perm_primario[_dim]:.4f} ± {importancias_perm_primario_std[_dim]:.4f}")
print()
print("Top-5 por importância nativa vs. top-5 por permutation importance (dimensão primária):")
print("Nativa:", list(importancias_primario.index[:5]))
print("Permutation:", list(importancias_perm_primario.index[:5]))
print()

# Importancia dos demais sinais (comportamental/temporais).
OUTROS_SINAIS = ["engajamento", "recencia_dias", "frequencia_eventos_30d", "tendencia"]
importancias_outros_sinais = pd.Series(modelo_rf.feature_importances_, index=feature_cols).loc[OUTROS_SINAIS]
importancias_perm_outros_sinais = pd.Series(perm.importances_mean, index=feature_cols).loc[OUTROS_SINAIS]
print("Importância relativa (nativa) dos demais sinais (comportamental/temporais):")
print(importancias_outros_sinais.round(4))
importancias_outros_sinais.to_csv("importancias_rf_outros_sinais.csv")
importancias_perm_outros_sinais.to_csv("importancias_perm_outros_sinais.csv")
print()

df.to_csv("dataset_simulado_pnps.csv", index=False)
resultados_df.to_csv("resultados_modelos.csv", index=False)
overfitting_df.to_csv("resultados_treino_teste.csv", index=False)
diagnostico_dimensoes_oof.to_csv("diagnostico_dimensoes_oof.csv")
diagnostico_dimensoes_true.to_csv("diagnostico_dimensoes_true.csv")
diagnostico_secundarios_oof.to_csv("diagnostico_secundarios_oof.csv")
precisao_por_camada.to_csv("precisao_por_camada.csv", index=False)
tier_comparacao.to_csv("tier_comparacao_true_vs_oof.csv", index=False)
tier_crosstab_oof.to_csv("tier_crosstab_oof.csv")
importancias_secundario.to_csv("importancias_rf_secundario.csv")
importancias_primario.to_csv("importancias_rf_primario.csv")
importancias_perm_secundario.to_csv("importancias_perm_secundario.csv")
importancias_perm_primario.to_csv("importancias_perm_primario.csv")
pd.DataFrame({
    "media": importancias_perm_secundario,
    "desvio_padrao": importancias_perm_secundario_std.loc[importancias_perm_secundario.index],
}).to_csv("importancias_perm_secundario_com_dp.csv")
pd.DataFrame({
    "media": importancias_perm_primario,
    "desvio_padrao": importancias_perm_primario_std.loc[importancias_perm_primario.index],
}).to_csv("importancias_perm_primario_com_dp.csv")
print("\nArquivos salvos: dataset_simulado_pnps.csv, resultados_modelos.csv, resultados_treino_teste.csv, "
      "diagnostico_dimensoes_oof.csv, diagnostico_dimensoes_true.csv, diagnostico_secundarios_oof.csv, "
      "precisao_por_camada.csv, resultados_cv.csv, tier_comparacao_true_vs_oof.csv, tier_crosstab_oof.csv, "
      "importancias_rf_secundario.csv, importancias_rf_primario.csv, importancias_perm_secundario.csv, "
      "importancias_perm_primario.csv, importancias_perm_secundario_com_dp.csv, "
      "importancias_perm_primario_com_dp.csv, pnps_agregado.csv")
