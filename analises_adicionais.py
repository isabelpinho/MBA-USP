# -*- coding: utf-8 -*-
"""
Analises complementares, calculadas sem alterar simulacao_pnps.py --
reaproveita o dataset ja gerado (dataset_simulado_pnps.csv) e, quando
necessario, reexecuta apenas o trecho de geracao da variavel latente +
ajuste do Random Forest, com os mesmos parametros do script original
(SEED=42; RF com max_depth=12, min_samples_leaf=20).

- Benchmark teorico (teto de acuracia) via Monte Carlo, dado o ruido
  independente ~ N(0, 0.08^2) da variavel latente.
- Correlacao de postos (Spearman) entre peso "verdadeiro" de cada dimensao
  e a permutation importance, em 3 niveis de ruido.
- Matriz de confusao tier real x tier previsto (out-of-fold).
- Precisao, revocacao e F1-score por classe do Random Forest.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, roc_auc_score
)
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr

SEED = 42
rng_global = np.random.default_rng(12345)  # gerador INDEPENDENTE do stream original, so para as reamostragens de ruido deste script

# Caminhos relativos a localizacao do proprio script: a pasta "resultados_csv"
# ao lado deste arquivo contem tanto o dataset de entrada (gerado por
# simulacao_pnps.py) quanto os CSVs de saida desta analise.
RESULTADOS_DIR = Path(__file__).resolve().parent / "resultados_csv"
DATASET_PATH = RESULTADOS_DIR / "dataset_simulado_pnps.csv"
OUT_DIR = str(RESULTADOS_DIR) + "/"

# ---------------------------------------------------------------------------
# 0. Constantes do modelo gerador (copiadas de simulacao_pnps.py)
# ---------------------------------------------------------------------------
DIMENSOES = [
    "entrega_logistica", "atendimento_cliente", "pagamento_credito",
    "programa_fidelidade", "preco_promocoes", "experiencia_app",
    "time_comercial", "sortimento",
]
LABELS_PT = {
    "entrega_logistica": "Entrega e logística", "atendimento_cliente": "Atendimento ao cliente",
    "pagamento_credito": "Pagamento e crédito", "programa_fidelidade": "Programa de fidelidade",
    "preco_promocoes": "Preço e promoções", "experiencia_app": "Experiência do aplicativo",
    "time_comercial": "Time comercial", "sortimento": "Sortimento",
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
SECUNDARIO_COLS = [s for d in DIMENSOES for s in SECUNDARIOS[d]]
SECUNDARIO_PARA_PRIMARIO = {s: d for d in DIMENSOES for s in SECUNDARIOS[d]}

PESOS_NEG = {
    "entrega_logistica": 0.290, "time_comercial": 0.230, "preco_promocoes": 0.137,
    "pagamento_credito": 0.114, "sortimento": 0.091, "atendimento_cliente": 0.069,
    "experiencia_app": 0.046, "programa_fidelidade": 0.023,
}
PESOS_POS = {
    "programa_fidelidade": 0.222, "experiencia_app": 0.185, "time_comercial": 0.155,
    "sortimento": 0.139, "atendimento_cliente": 0.110, "pagamento_credito": 0.085,
    "preco_promocoes": 0.062, "entrega_logistica": 0.042,
}
PESO_DIMENSOES = 0.35
PESO_ENGAJAMENTO = 0.25
PESO_FREQUENCIA = 0.20
PESO_RECENCIA = 0.13
PESO_TENDENCIA = 0.07
# (a variável "resultado" -- retenção/gasto -- foi removida do modelo: é
# consequência da satisfação, não insumo dela. Não entra nesta fórmula.)

DISTRIBUICAO_ALVO_NOTA = {
    0: 0.003, 1: 0.005, 2: 0.010, 3: 0.020, 4: 0.040, 5: 0.090, 6: 0.162,
    7: 0.210, 8: 0.200, 9: 0.150, 10: 0.110,
}
_notas_ordenadas = sorted(DISTRIBUICAO_ALVO_NOTA.keys())
_cum = 0.0
_cortes_cumulativos = []
for _n in _notas_ordenadas:
    _cum += DISTRIBUICAO_ALVO_NOTA[_n]
    _cortes_cumulativos.append(_cum)
_cortes_cumulativos[-1] = 1.0 + 1e-9

NOISE_SD_ORIGINAL = 0.08
RF_PARAMS = dict(n_estimators=300, random_state=SEED, n_jobs=-1, max_features="sqrt",
                  max_depth=12, min_samples_leaf=20)
feature_cols = SECUNDARIO_COLS + ["engajamento", "recencia_dias", "frequencia_eventos_30d", "tendencia"]


def notas_e_classe(latent):
    """Reproduz EXATAMENTE o mapeamento por quantil da secao 3c do script original."""
    percentil = pd.Series(latent).rank(pct=True, method="first").values
    idx = np.digitize(percentil, _cortes_cumulativos[:-1], right=True)
    nota = np.array(_notas_ordenadas)[idx]
    classe = np.select([nota >= 9, nota >= 7], ["promotor", "passivo"], default="detrator")
    return nota, classe


print("Carregando dataset_simulado_pnps.csv ...")
df = pd.read_csv(DATASET_PATH)
N = len(df)
print(f"N = {N}")

# ---------------------------------------------------------------------------
# 1. Reconstrucao do sinal deterministico (latent SEM ruido), a partir das
#    colunas ja existentes no dataset (os 8 escores de dimensao primaria,
#    engajamento, recencia, frequencia, tendencia sao colunas do CSV).
# ---------------------------------------------------------------------------
dimensao_contrib = np.zeros(N)
for d in DIMENSOES:
    score = df[d].values
    surplus = np.clip(score - 0.5, 0, None)
    shortfall = np.clip(0.5 - score, 0, None)
    dimensao_contrib += PESOS_POS[d] * surplus - PESOS_NEG[d] * shortfall
dimensao_score_ponderado = np.clip(0.5 + dimensao_contrib, 0, 1)

recencia_score = 1 - np.exp(-df["recencia_dias"].values / 10)
frequencia_norm = df["frequencia_eventos_30d"].values / max(df["frequencia_eventos_30d"].values.max(), 1)

latent_clean = (
    PESO_DIMENSOES * dimensao_score_ponderado
    + PESO_ENGAJAMENTO * df["engajamento"].values
    + PESO_RECENCIA * recencia_score
    - PESO_FREQUENCIA * frequencia_norm
    + PESO_TENDENCIA * df["tendencia"].values
)
# IMPORTANTE: NAO clipar aqui -- no script original o clip(0,1) e aplicado
# uma UNICA vez, DEPOIS de somar o ruido (latent = clip(soma + noise, 0, 1)).
# latent_clean fica sem clip, exatamente como a soma antes do ruido no
# script original.

# validacao: latent_clean + ruido dp=0.08 (mesma escala do original), apos UM
# clip, deve reproduzir latent_score de perto (mesma formula, mesmas colunas)
_latent_check = np.clip(latent_clean, 0, 1)
diff = df["latent_score"].values - _latent_check
print(f"[Checagem] corr(clip(latent_clean), latent_score) = {np.corrcoef(_latent_check, df['latent_score'].values)[0,1]:.4f} "
      f"(esperado: muito alta, ruido dp=0.08 é pequeno frente à escala do latente)")
print(f"[Checagem] media(diff) = {diff.mean():.4f} (esperado ~0), dp(diff) = {diff.std():.4f} (esperado ~0.08, um pouco menor por causa do clipping)")

# ---------------------------------------------------------------------------
# 2. ITEM 15 -- benchmark teorico (teto de acuracia dado apenas X)
# ---------------------------------------------------------------------------
print("\n=== ITEM 15: benchmark teorico (oraculo) ===")
M_REPS = 150
classe_true = df["classe"].values
oracle_draws = np.empty((M_REPS, N), dtype=object)
for m in range(M_REPS):
    noise_m = rng_global.normal(0, NOISE_SD_ORIGINAL, size=N)
    latent_m = np.clip(latent_clean + noise_m, 0, 1)
    _, classe_m = notas_e_classe(latent_m)
    oracle_draws[m] = classe_m

# 2a. acuracia "par-a-par": concordancia esperada entre duas realizacoes
#     independentes de ruido para o MESMO cliente (= acuracia do classificador
#     Bayes-otimo, avaliado contra uma nova realizacao de ruido)
pairwise_acc = []
for m in range(M_REPS):
    for k in range(m + 1, M_REPS):
        pairwise_acc.append(np.mean(oracle_draws[m] == oracle_draws[k]))
pairwise_acc = np.array(pairwise_acc)
oracle_acc_mean = pairwise_acc.mean()
oracle_acc_sd = pairwise_acc.std(ddof=1)

# 2b. classificador oraculo (decisao Bayes-otima = classe mais frequente
#     entre os M sorteios) avaliado contra a classe verdadeira observada
classes_ordenadas = sorted(df["classe"].unique())
proba_oracle = pd.DataFrame(
    {c: (oracle_draws == c).mean(axis=0) for c in classes_ordenadas}
)
classe_oracle_pred = proba_oracle.idxmax(axis=1).values
oracle_acc_vs_true = accuracy_score(classe_true, classe_oracle_pred)
y_true_bin = label_binarize(classe_true, classes=classes_ordenadas)
oracle_auc = roc_auc_score(y_true_bin, proba_oracle[classes_ordenadas].values, multi_class="ovr", average="macro")

print(f"Acurácia teórica (concordância esperada entre duas realizações independentes de ruído): "
      f"{oracle_acc_mean:.3f} ± {oracle_acc_sd:.3f} (M={M_REPS} sorteios, {len(pairwise_acc)} pares)")
print(f"Acurácia do classificador oráculo (decisão Bayes-ótima) vs. classe verdadeira observada: {oracle_acc_vs_true:.3f}")
print(f"AUC-ROC macro (oráculo) vs. classe verdadeira observada: {oracle_auc:.3f}")
print(f"Referência -- modelos ajustados: Regressão Logística acc=0.585/AUC=0.762; Random Forest acc=0.588/AUC=0.764")

pd.DataFrame([{
    "acuracia_teorica_media": round(oracle_acc_mean, 3),
    "acuracia_teorica_dp": round(oracle_acc_sd, 3),
    "acuracia_oraculo_vs_verdadeira": round(oracle_acc_vs_true, 3),
    "auc_oraculo_vs_verdadeira": round(oracle_auc, 3),
    "acuracia_regressao_logistica": 0.585,
    "auc_regressao_logistica": 0.762,
    "acuracia_random_forest": 0.588,
    "auc_random_forest": 0.764,
    "m_repeticoes_monte_carlo": M_REPS,
}]).to_csv(OUT_DIR + "resultados_benchmark_teorico.csv", index=False)

# ---------------------------------------------------------------------------
# 3. ITENS 8+11 -- correlacao de postos (pesos verdadeiros x importancia
#    estimada), no nivel de ruido original e em mais dois niveis
# ---------------------------------------------------------------------------
print("\n=== ITENS 8+11: correlação de postos (pesos verdadeiros x importância estimada) ===")
peso_verdadeiro = pd.Series({d: (PESOS_POS[d] + PESOS_NEG[d]) / 2 for d in DIMENSOES})

# nivel de ruido original (0.08): usa a permutation importance JA calculada
# pelo script original (importancias_perm_primario.csv), sem reajustar nada
imp_perm_original = pd.read_csv(OUT_DIR + "importancias_perm_primario.csv", index_col=0).iloc[:, 0]
imp_perm_original = imp_perm_original.reindex(DIMENSOES)

resultados_sensibilidade = []


def ajustar_e_avaliar(noise_sd, label):
    noise = rng_global.normal(0, noise_sd, size=N)
    latent = np.clip(latent_clean + noise, 0, 1)
    _, classe = notas_e_classe(latent)
    X = df[feature_cols].values
    y = classe
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEED
    )
    modelo_rf = RandomForestClassifier(**RF_PARAMS)
    modelo_rf.fit(X_train, y_train)
    perm = permutation_importance(modelo_rf, X_test, y_test, scoring="f1_macro",
                                   n_repeats=10, random_state=SEED, n_jobs=-1)
    imp_sec = pd.Series(perm.importances_mean, index=feature_cols).loc[SECUNDARIO_COLS]
    imp_prim = imp_sec.groupby(SECUNDARIO_PARA_PRIMARIO).sum().reindex(DIMENSOES)
    rho, pval = spearmanr(peso_verdadeiro.values, imp_prim.values)
    acc = accuracy_score(y_test, modelo_rf.predict(X_test))
    print(f"[{label}] ruído dp={noise_sd}: acurácia={acc:.3f}, "
          f"Spearman(peso verdadeiro, importância estimada)={rho:.3f} (p={pval:.4f})")
    resultados_sensibilidade.append({
        "condicao": label, "ruido_dp": noise_sd, "acuracia_rf": round(acc, 3),
        "spearman_rho": round(rho, 3), "spearman_p": round(pval, 4),
    })
    return imp_prim


rho0, p0 = spearmanr(peso_verdadeiro.values, imp_perm_original.values)
print(f"[Ruído original, já calculado] ruído dp=0.08: acurácia RF=0.588, "
      f"Spearman(peso verdadeiro, importância estimada)={rho0:.3f} (p={p0:.4f})")
resultados_sensibilidade.append({
    "condicao": "Ruído original (0.08)", "ruido_dp": 0.08, "acuracia_rf": 0.588,
    "spearman_rho": round(rho0, 3), "spearman_p": round(p0, 4),
})

ajustar_e_avaliar(0.04, "Ruído reduzido (metade)")
ajustar_e_avaliar(0.16, "Ruído aumentado (dobro)")

sens_df = pd.DataFrame(resultados_sensibilidade)
sens_df = sens_df[["condicao", "ruido_dp", "acuracia_rf", "spearman_rho", "spearman_p"]]
sens_df.to_csv(OUT_DIR + "resultados_correlacao_postos_sensibilidade_ruido.csv", index=False)
print("\nTabela de sensibilidade ao ruído:")
print(sens_df.to_string(index=False))

comparacao_pesos = pd.DataFrame({
    "dimensao": [LABELS_PT[d] for d in DIMENSOES],
    "peso_verdadeiro_medio": peso_verdadeiro.reindex(DIMENSOES).round(3).values,
    "importancia_estimada_perm": imp_perm_original.round(4).values,
}).sort_values("peso_verdadeiro_medio", ascending=False)
comparacao_pesos.to_csv(OUT_DIR + "comparacao_pesos_verdadeiros_vs_importancia.csv", index=False)
print("\nComparação pesos verdadeiros x importância estimada (ruído original):")
print(comparacao_pesos.to_string(index=False))

# ---------------------------------------------------------------------------
# 4. ITEM 20 -- matriz de confusão tier verdadeiro x tier previsto (OOF)
# ---------------------------------------------------------------------------
print("\n=== ITEM 20: matriz de confusão de tiers (verdadeiro x previsto OOF) ===")
confusion_tiers = pd.crosstab(df["tier_classe_verdadeira"], df["tier_previsto_oof"])
tier_ordem = ["Tier 1", "Tier 2", "Tier 3", "Tier 4"]
confusion_tiers = confusion_tiers.reindex(index=tier_ordem, columns=tier_ordem, fill_value=0)
print(confusion_tiers)
acerto_tier = np.trace(confusion_tiers.values) / confusion_tiers.values.sum()
print(f"Proporção de clientes no MESMO tier (verdadeiro = previsto): {acerto_tier:.3f}")
confusion_tiers.to_csv(OUT_DIR + "matriz_confusao_tiers.csv")
pd.DataFrame([{"proporcao_tier_correto": round(acerto_tier, 3)}]).to_csv(
    OUT_DIR + "matriz_confusao_tiers_resumo.csv", index=False)

# ---------------------------------------------------------------------------
# 5. ITEM 25 -- F1 por classe do Random Forest (baseline, ruído original)
# ---------------------------------------------------------------------------
print("\n=== ITEM 25: F1 por classe (Random Forest, ruído original) ===")
X = df[feature_cols].values
y = df["classe"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
modelo_rf_base = RandomForestClassifier(**RF_PARAMS)
modelo_rf_base.fit(X_train, y_train)
y_pred_base = modelo_rf_base.predict(X_test)
report = classification_report(y_test, y_pred_base, digits=3, output_dict=True)
f1_por_classe = pd.DataFrame([
    {"classe": c, "precisao": round(report[c]["precision"], 2),
     "revocacao": round(report[c]["recall"], 2), "f1": round(report[c]["f1-score"], 2),
     "n_teste": int(report[c]["support"])}
    for c in classes_ordenadas
])
print(f1_por_classe.to_string(index=False))
f1_por_classe.to_csv(OUT_DIR + "f1_por_classe_rf.csv", index=False)

print("\nConcluído. Arquivos gravados em", OUT_DIR)
