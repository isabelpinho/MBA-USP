# Código-fonte — NPS Preditivo em Plataforma Digital B2B (TCC)

Scripts Python usados na simulação de dados e nas análises do TCC de Isabel Pinho
(MBA em Data Science & Analytics, USP/Esalq). Todos dados são sintéticos, gerados pelos próprios scripts.

## Ordem de execução

1. **`simulacao_pnps.py`** — script principal. Gera o dataset simulado e roda os
   modelos de referência (regressão logística e Random Forest). Precisa ser
   executado primeiro: cria a pasta `resultados_csv/` com o dataset e os
   resultados que os demais scripts reaproveitam.

   ```
   python simulacao_pnps.py
   ```

2. Depois de rodar o script principal, os scripts abaixo podem ser executados
   em qualquer ordem (cada um lê `resultados_csv/dataset_simulado_pnps.csv` ou
   reexecuta internamente um trecho de `simulacao_pnps.py`):

   - `analises_adicionais.py` — análises complementares (Tabelas 2, 9, 11 e 12
     do TCC).
   - `comparacao_rf_boosting_5fold.py` — comparação Random Forest vs.
     HistGradientBoostingClassifier com validação cruzada de 5 partições.
   - `comparacao_ordinal_pnps.py` — comparação entre regressão logística
     multinomial e ordinal (modelo de chances proporcionais).
   - `comparacao_duas_saidas_e_11_notas.py` e `comparacao_nota_continua_pnps.py`
     — análises complementares de desenho alternativo (duas saídas / 11 notas;
     nota contínua).
   - `curva_aprendizado_ruido.py` — sensibilidade ao ruído e ao tamanho do
     treino.
   - `sensibilidade_correlacao_drivers.py` — sensibilidade à correlação entre
     drivers.
   - `simulacao_causa_raiz_repeticoes.py` — simulação separada (30 repetições)
     usada no diagnóstico de causa raiz (Tabela 17 / Apêndice).

## Como instalar e rodar

1. Instale o Python 3 (python.org/downloads), caso ainda não tenha.
2. No terminal, dentro desta pasta, instale as dependências:

   ```
   pip install -r requirements.txt
   ```

3. Rode o script principal primeiro:

   ```
   python simulacao_pnps.py
   ```

Cada script imprime os resultados no terminal e salva os CSVs correspondentes
em `resultados_csv/`.
