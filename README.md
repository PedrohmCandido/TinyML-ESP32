# TCC TinyML — Reconhecimento de Gestos com ESP32-CAM

**Autores:** Kaiki Andrade Silva e Pedro Henrique Mendes Cândido
**Curso:** Engenharia de Software — Uni-FACEF
**Publicação:** RECA — Revista Eletrônica de Computação Aplicada (2026)

---

## Descrição

Investigação da viabilidade de executar redes neurais convolucionais em dispositivos de
borda com recursos severamente restritos, adotando o reconhecimento de gestos
(pedra, papel e tesoura) como tarefa de validação.

O pipeline cobre desde a preparação do conjunto de dados até a inferência embarcada no
ESP32-CAM, sem qualquer dependência de conexão externa em tempo de operação.

---

## Resultados

| Métrica | Valor |
|---|---|
| Redução do volume de dados (pré-processamento) | 96,34% |
| Redução do modelo (quantização int8) | 91,65% |
| Custo da quantização em acurácia | 0,49 ponto percentual |
| Acurácia no conjunto de teste (`.h5`) | 79,76% |
| Acurácia no conjunto de teste (`.tflite`) | 79,27% |
| Ocupação de Flash | 1.460.509 / 3.145.728 bytes (46%) |
| Arena de tensores (PSRAM) | 178.536 / 204.800 bytes (87,2%) |
| Latência de inferência | 1.115,52 ms |
| Taxa do sistema | 0,9 quadro/s |

Ensaios com participantes: 225 quadros com gesto, 150 classificações corretas,
1 incorreta e 74 rejeitadas pelo limiar de confiança de 70%.
Em 145 quadros sem mão no campo de visão, nenhuma classificação ultrapassou o limiar.

---

## Estrutura do repositório

```
tcc-tinyml/
├── data/
│   ├── raw/
│   │   ├── rps-real-original/   ← dataset como baixado (não versionado)
│   │   └── rps-real/            ← dividido em train/val/test
│   └── processed/
│       └── rps-real/            ← 96×96 em escala de cinza
├── preprocessing/
│   ├── resize.py                ← recorte central + 96×96 (Kaiki)
│   ├── grayscale.py             ← conversão para tons de cinza (Pedro)
│   └── pipeline.py              ← orquestração do pré-processamento
├── models/final/
│   ├── modelo_gestos.h5         ← modelo treinado (12.003.200 bytes)
│   ├── modelo_gestos.tflite     ← modelo quantizado (1.002.144 bytes)
│   └── modelo_gestos.h          ← vetor C para o firmware
├── resultados/
│   ├── diagnostico_baseline.txt ← relatório de avaliação
│   ├── confusao_h5.png
│   └── confusao_tflite.png
├── dividir_dataset.py           ← divisão estratificada 70/15/15
├── train.py                     ← treinamento da CNN
├── convert.py                   ← conversão e quantização int8
├── to_c_array.py                ← geração do vetor em linguagem C
├── avaliar_modelo.py            ← acurácia, matrizes de confusão, limiar
├── visualizar_camera.py         ← inspeção do quadro entregue ao modelo
├── requirements.txt
└── README.md
```

O firmware fica em pasta separada, na estrutura esperada pela Arduino IDE:

```
inferencia_gestos/
├── inferencia_gestos.ino
└── modelo_gestos.h
```

---

## Ambiente

**Treinamento**

| Componente | Versão |
|---|---|
| Python | 3.10.11 |
| TensorFlow | 2.10 |
| cuDNN | 8.1.0 |
| OpenCV | 4.9.0.80 |
| NumPy | anterior à série 2.x |

O TensorFlow 2.10 é a última versão com suporte nativo a GPU no Windows. Versões
superiores descontinuaram esse suporte, e o NumPy 2.x quebra a API esperada por ela.

**Embarcado**

| Componente | Versão |
|---|---|
| Arduino IDE | 2.3.10 (CLI 1.5.1) |
| Core ESP32 (Espressif) | 3.3.10 |
| Biblioteca TFLite Micro | ESP_TF 2.0.1 |

A biblioteca `TensorFlowLite_ESP32` é incompatível com o core atual. Use a `ESP_TF`.

**Hardware:** ESP32-CAM AI-Thinker com sensor OV2640 e PSRAM de 4 MB.
A PSRAM é obrigatória — a arena de tensores não cabe na memória interna.

---

## Conjunto de dados

**Rock-Paper-Scissors-Dataset**, de Alexandre Donciu-Julin, obtido no Kaggle.
2.717 imagens de 300×300 pixels sobre fundo cinza uniforme.

Licenciado sob **CC BY-NC 4.0** — exige atribuição ao autor e veda uso comercial.

O dataset não é versionado aqui por conta do tamanho. Baixe e extraia em
`data/raw/rps-real-original/`, mantendo as três pastas de classe (`paper`, `rock`,
`scissors`).

---

## Reprodução

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute as etapas na ordem. **A divisão vem antes do pré-processamento** — o
`pipeline.py` espera receber a pasta já dividida e apenas espelha a estrutura.

```bash
# 1. Divisão estratificada 70/15/15 (semente 42)
python dividir_dataset.py \
  --input data/raw/rps-real-original \
  --output data/raw/rps-real

# 2. Recorte central, 96×96 e escala de cinza
python preprocessing/pipeline.py \
  --input data/raw/rps-real \
  --output data/processed/rps-real

# 3. Treinamento (até 50 épocas, com interrupção antecipada)
python train.py

# 4. Conversão e quantização int8
python convert.py

# 5. Geração do vetor em linguagem C
python to_c_array.py

# 6. Avaliação (acurácia, matrizes de confusão, análise do limiar)
python avaliar_modelo.py
```

Ao final, copie `models/final/modelo_gestos.h` para a pasta do firmware, ao lado do
`.ino`, e grave no dispositivo pela Arduino IDE.

---

## Três parâmetros replicados manualmente

Estes valores são definidos no treinamento e precisam ser reproduzidos no firmware.
**Nenhum deles é verificado automaticamente** — se algum divergir, o sistema continua
compilando e reportando gestos com confiança elevada, porém incorretos.

| Parâmetro | Definido em | Deve corresponder a |
|---|---|---|
| Formato 96×96 em escala de cinza | `resize.py` e `grayscale.py` | configuração do sensor |
| Divisão por 255 | `.map()` do `train.py` | `/ 255.0f` no firmware |
| Ordem alfabética das classes | pastas lidas pelo Keras | vetor `CLASSES[]` em C |

---

## Limitações conhecidas

- **99,50% dos parâmetros** estão na primeira camada densa, o que responde pela
  latência de 1.115 ms. Substituir o enfileiramento por uma operação de resumo
  reduziria o modelo em cerca de 140 vezes — não implementado nem avaliado.
- **Não há classe para ausência de gesto.** O limiar de 70% é paliativo: no conjunto
  de teste, rejeita 104 acertos e deixa passar 30 erros.
- **Depende de PSRAM**, o que impede o transporte direto para microcontroladores sem
  memória externa.
- **Não houve busca sistemática de hiperparâmetros.** Os valores foram adotados como
  configuração inicial e mantidos após a obtenção de um modelo funcional.
- Os resultados dos ensaios físicos valem para **fundo uniforme e iluminação
  controlada**, e não se generalizam para ambientes livres.

---

## Histórico de branches

| Branch | Responsável | Tarefa |
|---|---|---|
| `main` | Ambos | Estrutura base e integração |
| `branch/kaiki-resize` | Kaiki | Redimensionamento (`resize.py`) |
| `branch/pedro-grayscale` | Pedro | Conversão para cinza (`grayscale.py`) |
