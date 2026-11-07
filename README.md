# 🎗️ Diagnóstico de Câncer de Mama com Redes Neurais

Rede neural **MLP** que classifica tumores de mama como **benignos** ou **malignos** a partir de características de células extraídas de exames, como apoio à decisão médica em diagnósticos incertos.

> Projeto desenvolvido na disciplina de Redes Neurais do curso de Engenharia de Computação (UERJ).

---

## 🎯 Objetivo

Construir e ajustar uma rede neural capaz de distinguir tumores benignos de malignos com alta acurácia, e verificar a estabilidade do modelo com validação cruzada.

## 📊 Dados

Base pública **Breast Cancer Wisconsin (Diagnostic)** (UCI / Kaggle):

| Item | Detalhe |
|---|---|
| Amostras | 569 (357 benignas, 212 malignas) |
| Variáveis de entrada | 30 características numéricas dos núcleos celulares (raio, textura, perímetro, área, suavidade, concavidade, simetria, dimensão fractal etc.), cada uma em média, erro padrão e pior valor |
| Alvo | `diagnosis`: B (benigno) ou M (maligno) |
| Divisão | 80% treino (455) / 20% teste (114) |

## 🧠 Metodologia

1. **Análise exploratória:** histogramas e boxplots das 30 variáveis, separados por diagnóstico, e matriz de correlação.
2. **Busca em grade manual** com 768 combinações de hiperparâmetros:
   - neurônios da 1ª camada: 16 ou 32; da 2ª camada: 4 ou 8;
   - funções de ativação: tanh, sigmoid, softplus, linear e softmax;
   - taxa de aprendizado: 0,0001, 0,001, 0,003 e 0,01.
3. **Modelo final:** rede totalmente conectada **30 → 32 → 8 → 2** (softplus, softplus, softmax), otimizador Adam (lr = 0,01), 200 épocas, **1.274 parâmetros** treináveis.
4. **Validação cruzada K-Fold** (K de 2 a 10) com `GridSearchCV`, para avaliar a estabilidade do modelo.
5. **Comparação** com um modelo de referência (Regressão Logística) por meio da matriz de confusão.

## ✅ Resultados

### Modelo final (conjunto de teste, 114 amostras)

| Métrica | Valor |
|---|---|
| **Acurácia** | **98,2%** |
| Loss | 0,079 |
| Precisão / Recall: benigno | 0,99 / 0,99 |
| Precisão / Recall: maligno | 0,98 / 0,98 |
| F1-score (macro) | 0,98 |

O modelo errou apenas **2 de 114** casos de teste.

### Validação cruzada K-Fold

| K | Acurácia média na validação cruzada | Acurácia no teste |
|---|---|---|
| 4 | 94,5% | 93% |
| 8 | **95,2%** | 96% |
| 10 | 82,0% | **98%** |

Em algumas execuções (por exemplo, K = 4, 8 e 10 na varredura completa), a rede convergiu para prever apenas a classe majoritária (acurácia de 62%). Isso indica **sensibilidade à inicialização dos pesos** e à falta de normalização das entradas, que têm escalas muito diferentes (como `area_mean` na casa das centenas e `smoothness_mean` perto de 0,1).

**Principais conclusões:**
- Uma rede pequena, com apenas duas camadas ocultas, atinge **acurácia acima de 98%** neste problema.
- O recall de **0,98 na classe maligna** importa muito, porque um falso negativo (tumor maligno classificado como benigno) é o erro mais grave.
- A validação cruzada mostrou que o desempenho varia entre execuções. Padronizar os dados (com `StandardScaler`, por exemplo) e fixar sementes aleatórias são os próximos passos para estabilizar o treinamento.

## 🛠️ Tecnologias

Python · TensorFlow/Keras · Scikit-learn · SciKeras · Pandas · NumPy · Matplotlib · Seaborn

## ▶️ Como executar

1. Baixe o dataset [Breast Cancer Wisconsin (Diagnostic)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) (`data.csv`).
2. Instale as dependências:
   ```bash
   pip install pandas numpy matplotlib seaborn tensorflow scikit-learn==1.3.2 scikeras==0.12.0
   ```
3. Abra `Predição de câncer de mama.ipynb`, ajuste o caminho do `data.csv` na célula de leitura e execute as células em ordem.

## 📁 Estrutura

```
├── Predição de câncer de mama.ipynb   # Análise exploratória, grid search, modelo final e K-Fold
└── README.md
```

---

👤 **Arthur Tavares Lindolpho** · [GitHub](https://github.com/ArthurLindolpho) · [LinkedIn](https://linkedin.com/in/arthur-lindolpho)
