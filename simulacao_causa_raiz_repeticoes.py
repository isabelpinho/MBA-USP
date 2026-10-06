# -*- coding: utf-8 -*-
"""
Simulacao do cenario "causa raiz clara" (Beta(1.5,5.0), media~0.23), com
(a) diagnostico por multiplas observacoes por driver (1, 3, 5 e 10) e
(b) repeticoes com seeds diferentes, para reportar media e intervalo de
confianca de 95% entre repeticoes.

Metade dos clientes (P_CAUSA_RAIZ=0.5) recebe uma causa raiz dominante
amostrada de Beta(1.5,5.0); os demais drivers seguem Beta(2,2). O
diagnostico ranqueia os 24 escores do cliente (media de k observacoes
independentes por driver) e aponta o mais baixo; a acuracia e calculada so
entre os clientes com causa raiz.
"""
import numpy as np
import pandas as pd

N = 25000
N_REPETICOES = 30
SEEDS = range(1000, 1000 + N_REPETICOES)

DIMENSOES = ['entrega_logistica', 'atendimento_cliente', 'pagamento_credito',
             'programa_fidelidade', 'preco_promocoes', 'experiencia_app',
             'time_comercial', 'sortimento']
SECUNDARIOS = {
    'entrega_logistica': ['prazo_entrega', 'integridade_pedido', 'rastreabilidade'],
    'atendimento_cliente': ['tempo_resposta', 'resolucao_primeiro_contato', 'qualidade_atendimento'],
    'pagamento_credito': ['facilidade_pagamento', 'condicoes_credito', 'clareza_cobranca'],
    'programa_fidelidade': ['atratividade_beneficios', 'facilidade_resgate', 'comunicacao_programa'],
    'preco_promocoes': ['percepcao_justica_preco', 'relevancia_promocoes', 'transparencia_precificacao'],
    'experiencia_app': ['usabilidade_app', 'estabilidade_tecnica', 'velocidade_app'],
    'time_comercial': ['consultoria_tecnica', 'disponibilidade_vendedor', 'confiabilidade_compromissos'],
    'sortimento': ['amplitude_catalogo', 'disponibilidade_estoque', 'relevancia_sortimento'],
}
SECUNDARIO_COLS = [s for d in DIMENSOES for s in SECUNDARIOS[d]]
N_DRIVERS = len(SECUNDARIO_COLS)  # 24

P_CAUSA_RAIZ = 0.5
ALPHA_CAUSA, BETA_CAUSA = 1.5, 5.0  # "causa raiz clara": media ~0.2308
K_OBSERVACOES = [1, 3, 5, 10]


def roda_uma_repeticao(seed):
    rng = np.random.default_rng(seed)
    tem_causa_raiz = rng.random(N) < P_CAUSA_RAIZ
    causa_raiz_driver = rng.choice(SECUNDARIO_COLS, size=N)
    idx_causa_raiz = np.where(tem_causa_raiz)[0]
    driver_real = causa_raiz_driver[idx_causa_raiz]
    n_causa_raiz = len(idx_causa_raiz)

    resultado_rep = {}
    for k in K_OBSERVACOES:
        # para cada driver, sorteia k observacoes independentes e usa a media
        # (equivalente ao "corte estatico" quando k=1)
        medias = {}
        for s in SECUNDARIO_COLS:
            obs = rng.beta(2, 2, size=(N, k))
            mask = tem_causa_raiz & (causa_raiz_driver == s)
            if mask.sum() > 0:
                obs[mask] = rng.beta(ALPHA_CAUSA, BETA_CAUSA, size=(mask.sum(), k))
            medias[s] = obs.mean(axis=1)
        df_k = pd.DataFrame(medias)
        sub = df_k.loc[idx_causa_raiz, SECUNDARIO_COLS].values
        ranking = np.argsort(sub, axis=1)
        driver_previsto_top1 = np.array(SECUNDARIO_COLS)[ranking[:, 0]]
        acerto_top1 = (driver_previsto_top1 == driver_real)
        resultado_rep[k] = acerto_top1.mean() * 100
    resultado_rep['n_causa_raiz'] = n_causa_raiz
    return resultado_rep


print(f"Rodando {N_REPETICOES} repeticoes (seeds {SEEDS.start}-{SEEDS.stop - 1}), "
      f"k em {K_OBSERVACOES}...")
linhas = [roda_uma_repeticao(s) for s in SEEDS]
df_rep = pd.DataFrame(linhas)

resumo = []
for k in K_OBSERVACOES:
    vals = df_rep[k].values
    media = vals.mean()
    dp = vals.std(ddof=1)
    se = dp / np.sqrt(N_REPETICOES)
    ic95 = 1.96 * se
    resumo.append({
        'k_observacoes': k,
        'acuracia_media_pct': round(media, 1),
        'desvio_padrao_pct': round(dp, 2),
        'ic95_mais_menos_pct': round(ic95, 2),
        'ic95_low': round(media - ic95, 1),
        'ic95_high': round(media + ic95, 1),
    })
    print(f"k={k:2d} observações/driver: acurácia média = {media:.1f}% "
          f"(dp entre repetições = {dp:.2f}, IC95% = [{media-ic95:.1f}%, {media+ic95:.1f}%])")

df_resumo = pd.DataFrame(resumo)
df_resumo.to_csv('resultados_causa_raiz_repeticoes.csv', index=False)
print(f"\nn médio com causa raiz dominante: {df_rep['n_causa_raiz'].mean():.0f} de {N} "
      f"({df_rep['n_causa_raiz'].mean()/N*100:.1f}%)")
print("Salvo: resultados_causa_raiz_repeticoes.csv")
