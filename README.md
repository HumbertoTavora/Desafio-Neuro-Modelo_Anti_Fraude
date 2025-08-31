# Sistema Avançado de Detecção de Propensão a Fraudes em Financiamento Veicular

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Machine Learning](https://img.shields.io/badge/ML-Ensemble%20Learning-green)](https://scikit-learn.org)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success)](https://github.com)

## 📋 Visão Geral

Este projeto implementa uma solução consistente para detecção de propensão a fraudes em financiamento veicular. Duas abordagens foram propostas, uma utilizando **XGBoost**, e outra com utilizando **ensemble learning** com **stacking** de múltiplos algoritmos de machine learning. O modelo foi desenvolvido com foco em maximizar o **impacto financeiro** enquanto mantém **boa precisão** e o **minima impacto na operação** do cliente .

### 🎯 Problema de Negócio

- **Contexto**: Detecção de propensão a fraudes em financiamento veicular
- **Desafio**: Classes altamente desbalanceadas (~4% fraudes)
- **Objetivo**: Maximizar detecção de fraudes minimizando falsos positivos e sem impactar a operação do cliente.

### 🏆 Resultados Alcançados

| Métrica | Baseline (XGBoost) | Modelo Final | Melhoria |
|---------|-------------------|--------------|----------|
| **Precision** | **71.23%** | 59.62x% | -17.7% |
| **Recall** | 13.10% | **15.62%** | +19.0% |
| **F1-Score** | 22.13% | **24.75%** | +11.8x% |
| **Impacto Financeiro** | R$ 1.268.153,70 | **R$ 1.477.419,00** | +16.5% |
| **Alert Rate** | 1.00% | **1.09%** | +0.9% |

## 🔧 Arquitetura da Solução

### Stack Tecnológico
- **Python 3.8+** com bibliotecas científicas
- **Scikit-learn** para pipeline e ensemble
- **XGBoost, LightGBM, CatBoost, RL e RF** como base learners
- **SMOTE e SMOTEENN** para balanceamento de classes
- **Stratified K-Fold** para validação robusta

### Arquitetura do Modelo XGBoost

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Raw Features  │ -> │  Preprocessing   │ -> │ Feature Matrix  │
│                 │    │                  │    │                 │
│ • Numéricas     │    │ • Winsorização   │    │ • Scaled        │
│ • Categóricas   │    │ • Imputação      │    │ • Encoded       │
│ • Texto         │    │ • Scaling        │    │ • Balanced      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                 │
                                 v
                      ┌─────────────────────┐
                      │     (XGBoost)       │
                      └─────────────────────┘   
                                 │
                                 v
                    ┌─────────────────────┐
                    │ Fraud Probability   │
                    └─────────────────────┘

```
### Arquitetura do Modelo Ensemble

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Raw Features  │ -> │  Preprocessing   │ -> │ Feature Matrix  │
│                 │    │                  │    │                 │
│ • Numéricas     │    │ • Winsorização   │    │ • Scaled        │
│ • Categóricas   │    │ • Imputação      │    │ • Encoded       │
│ • Texto         │    │ • Scaling        │    │ • Balanced      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                 │
                                 v
┌─────────────────────────────────────────────────────────────────┐
│                    STACKING ENSEMBLE                            │
│                                                                 │
│  Base Learners:                                                 │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐  │
│  │ Logistic    │ │ Random      │ │ LightGBM    │ │ CatBoost  │  │
│  │ Regression  │ │ Forest      │ │             │ │           │  │
│  └─────────────┘ └─────────────┘ └─────────────┘ └───────────┘  │
│           │             │             │             │           │
│           └─────────────┼─────────────┼─────────────┘           │
│                         v             v                         │
│                   ┌─────────────────────┐                       │
│                   │   Meta Learner      │                       │
│                   │   (XGBoost)         │                       │
│                   └─────────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 v
                    ┌─────────────────────┐
                    │ Fraud Probability   │
                    │    + Threshold      │
                    │   Optimization      │
                    └─────────────────────┘
```

## 📊 Metodologia

### 1. Engenharia de Features
- **Preprocessamento Numérico**: Winsorização Simples (1%) e menos agressiva (0.5%) + RobustScaler e StandardScaler
- **Preprocessamento Categórico**: Imputação + OneHotEncoder (max 50 categorias)
- **Tratamento de Outliers**: Estratégia robusta com quantis

### 2. Balanceamento de Classes
- **SMOTE e SMOTEENN**: SMOTE (oversampling) Tradicional e Combinação de SMOTE (oversampling) + undersampling (ENN)

### 3. Validação e Treinamento
- **Stratified 7-Fold CV**: Validação robusta com estratificação
- **Out-of-Fold Predictions**: Prevenção de overfitting
- **Cross-Validation Interna**: Meta-learner com CV=3

### 4. Otimização de Threshold
- **Análise Multi-Threshold**: Teste de diferentes pontos de corte
- **Otimização Financeira**: Maximização do impacto de negócio
- **Restrições Operacionais**: Consideração da capacidade de revisão

## 📁 Estrutura do Projeto

```
Desafio Neuro - Modelo Anti Fraude/
├── datasets/
│   ├── base_antifraude.gz                 # Dados originais
├── notebooks/
│   ├── eda.ipynb               # Análise exploratória
│   ├── preprocess_modelo.ipynb # Pre-processamento e Desenvolvimento + Avaliação do modelo
├── src/
│   └── utils.py            # Funções auxiliares
├── papers/
    └── varios.pdf          # Estudos interessantes
├── requirements.txt        # Dependências
└── README.md               # Este arquivo
```

## 🔍 Principais Insights

### 1. **Ensemble Pode Superar Modelos Individuais**
- Para alguns cenários o Stacking com 4 algoritmos diversos > XGBoost isolado

### 2. **Balanceamento de Classes é Crítico**
- SMOTEENN > SMOTE
- Melhoria no recall

### 3. ** Otimização do Threshold é Necessário**
- Trade-off precision vs recall bem definido
- Necessidade de alinhamento com capacidade operacional

### 4. **Feature Engineering Impacta Significativamente**
- Winsorização menos agressiva (0.5% vs 1.0%)
- StandardScaler > RobustScaler para ensemble
- Limitação de categorias previne overfitting

## 🆙 Próximos Passos (O que faltou)

### Melhorias Técnicas
- [ ] **EDA mais robusto**: Com mais tempo de projeto, seria muito interessante realizar uma análise mais aprofundada na base de dados, com analise de distribuições, correlações e outliers de mais atributos.
- [ ] **Pipeline de pré-processamento**: Com um EDA mais robusto, poderiamos implementar um Pipeline mais eficiente e utilizar técnicas mais otimizadas para os problemas encontrados.
- [ ] **Hyperparameter Tuning**: Tanto para o XGBoost isolado como para o Ensemble, técnicas de otimização de hiperparametros seriam interessanes para extrair ainda mais performance dos modelos. Ferramentas como Optuna ou Algoritmos Genéticos seriam promissoas. (Um dos artigos que estou a frente no mestrado é demonstrando a capacidade dos algoritmos genéticos otimizarem os hiperparametros de modelos de ML com mais eficiencia :D. Artigo no diretório **Papers** ).
