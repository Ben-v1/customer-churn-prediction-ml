# Previsão de Churn de Clientes

Projeto de Ciência de Dados para prever clientes com maior probabilidade de churn, usando EDA, limpeza de dados, engenharia de atributos, comparação de modelos, explicabilidade com SHAP e recomendações de negócio.

## Problema de negócio

Churn representa perda de receita recorrente e aumento de custo de aquisição. A proposta é criar um modelo que ajude a priorizar clientes com maior risco de cancelamento para ações de retenção, como contato proativo, ofertas direcionadas e revisão de experiência.

## Dataset

Fonte: Kaggle — Customer Churn Dataset  
Link: https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset

Arquivos esperados em `data/raw/`:

```text
customer_churn_dataset-training-master.csv
customer_churn_dataset-testing-master.csv
```

> Observação: os arquivos de dados não devem ser versionados no GitHub. Use a pasta `data/raw/` localmente.

## O que este projeto demonstra

- Análise exploratória com gráficos e tabelas.
- Auditoria de qualidade dos dados.
- Engenharia de atributos orientada a negócio.
- Pipelines com `scikit-learn` para evitar vazamento de dados.
- Comparação de modelos: baseline, regressão logística, árvore de decisão e XGBoost.
- Avaliação com ROC-AUC, PR-AUC, precisão, recall e F1.
- Ajuste de threshold para campanha de retenção.
- Explicabilidade com SHAP.
- Diagnóstico de drift entre treino e teste com PSI.
- Exportação do pipeline para deploy.
- App Streamlit para simulação de score de churn.

## Estrutura do projeto

```text
.
├── app/
│   └── streamlit_app.py
├── data/
│   └── raw/
│       └── .gitkeep
├── models/
│   └── .gitkeep
├── notebooks/
│   └── customer_churn_prediction.ipynb
├── reports/
│   └── figures/
│       └── .gitkeep
├── .gitignore
├── README.md
└── requirements.txt
```

## Resultados principais

Na validação interna, o XGBoost foi o modelo campeão pelo critério de ROC-AUC.

| Split | Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validação | XGBoost | 0.9979 | 1.0000 | 0.9963 | 0.9981 | 1.0000 | 1.0000 |
| Teste externo | XGBoost | 0.5053 | 0.4891 | 0.9983 | 0.6566 | 0.7327 | 0.6569 |

A queda no teste externo é um achado importante: há sinais de mudança de distribuição entre treino e teste. Por isso, o projeto inclui PSI e uma discussão de monitoramento de drift. Em um ambiente real, eu não colocaria o modelo em produção sem calibração, validação temporal e acompanhamento por safra.

## Principais fatores de churn encontrados

Pelos sinais de importância do modelo e SHAP, os fatores mais relevantes incluem:

- número de chamados de suporte;
- gasto total e gasto médio;
- idade;
- atraso de pagamento;
- tipo de contrato, principalmente contrato mensal;
- tempo desde a última interação;
- frequência de uso.

## Recomendações de negócio

1. Criar fila semanal de clientes com maior risco de churn.
2. Priorizar clientes com muitos chamados de suporte para atendimento proativo.
3. Criar régua de comunicação para clientes com atraso de pagamento.
4. Oferecer incentivo para migração de contrato mensal para contratos mais longos.
5. Monitorar drift, calibração e performance por safra antes de escalar o modelo.

## Como executar

Crie e ative um ambiente virtual:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute o notebook:

```bash
jupyter notebook notebooks/customer_churn_prediction.ipynb
```

Depois de rodar o notebook, o modelo será salvo em:

```text
models/churn_xgboost_pipeline.joblib
models/churn_model_metadata.json
```

Para abrir o app Streamlit:

```bash
python -m streamlit run app/streamlit_app.py
``` 

## Próximos passos

- Validação temporal com safras reais.
- Calibração de probabilidades.
- Otimização de threshold por lucro esperado.
- Monitoramento de PSI e performance em produção.
- Experimento A/B para medir o impacto incremental das ações de retenção.

---

© Brenno Gomes
