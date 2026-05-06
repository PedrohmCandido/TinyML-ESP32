# TCC TinyML — Reconhecimento de Gestos com ESP32-CAM

**Autores:** Kaiki Andrade Silva e Pedro Henrique Mendes Cândido  
**Curso:** Engenharia de Software — Uni-FACEF  
**Revista:** RECA — Revista Eletrônica de Computação Aplicada

---

## Descrição

Implementação de reconhecimento de gestos (Pedra, Papel e Tesoura) em tempo real utilizando TinyML no microcontrolador ESP32-CAM, com Redes Neurais Convolucionais (CNNs) otimizadas para dispositivos de borda.

---

## Estrutura do Repositório

```
tcc-tinyml/
├── data/
│   ├── raw/              ← datasets originais (não versionados)
│   └── processed/        ← imagens após pré-processamento
├── preprocessing/
│   ├── resize.py         ← redimensionamento para 96x96 px (Kaiki)
│   ├── grayscale.py      ← conversão para tons de cinza (Pedro)
│   └── pipeline.py       ← orquestração do pipeline completo
├── models/
│   └── final/            ← modelo .tflite final (versionado)
├── requirements.txt
└── README.md
```

---

## Pré-requisitos

```bash
pip install -r requirements.txt
```

---

## Como executar o pipeline

```bash
python preprocessing/pipeline.py \
  --input data/raw/ \
  --output data/processed/
```

---

## Branches

| Branch | Responsável | Tarefa |
|---|---|---|
| `main` | Ambos | Estrutura base e integração |
| `branch/kaiki-resize` | Kaiki | Script de redimensionamento (resize.py) |
| `branch/pedro-grayscale` | Pedro | Script de conversão grayscale (grayscale.py) |

---

## Datasets Utilizados

Os datasets devem ser baixados manualmente e colocados em `data/raw/`.  
Não são versionados no repositório por conta do tamanho.

---

## Cronograma

| Período | Etapa |
|---|---|
| Abril/Maio | Pré-processamento do dataset |
| Maio/Junho | Treinamento e conversão TFLite |
| Junho/Julho | Deploy na ESP32-CAM e testes |
| Julho/Agosto | Escrita dos Resultados e Conclusão |
| Agosto/Setembro | Revisão final do artigo |
