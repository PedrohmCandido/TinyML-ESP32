#include <Arduino.h>
#include <esp_camera.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "modelo_gestos.h"

#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// ---- Streaming da imagem para visualizacao/debug via Python ----
// 1 = envia cada frame capturado pela Serial (para o script de visualizacao)
// 0 = desativa (usar durante a MEDICAO DE FPS, pois transmitir 9.216
//     bytes por quadro afeta a taxa do ciclo completo)
#define ENVIAR_IMAGEM_SERIAL 0

// ---- Coleta de metricas para o artigo ----
// 1 = mede latencia de inferencia, consumo da arena e taxa de quadros
// 0 = desativa toda a instrumentacao (volta ao firmware original)
#define MEDIR_METRICAS 1

// Pausa entre inferencias. Existe apenas para o log serial ficar legivel
// e NAO representa o limite de desempenho do dispositivo.
// Durante a medicao de FPS este valor deve ser 0.
#define DELAY_LOOP_MS 0

const char* CLASSES[] = {"paper", "rock", "scissors"};
const int NUM_CLASSES = 3;

constexpr int TENSOR_ARENA_SIZE = 200 * 1024;
uint8_t* tensor_arena = nullptr;

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

#if MEDIR_METRICAS
// Acumuladores das estatisticas. unsigned long porque micros() retorna
// esse tipo e o valor acumulado cresce rapidamente.
unsigned long soma_latencia_us  = 0;   // soma do tempo de Invoke()
unsigned long min_latencia_us   = 4294967295UL;
unsigned long max_latencia_us   = 0;
unsigned long soma_ciclo_us     = 0;   // soma do loop() completo
unsigned long total_inferencias = 0;

// A cada quantas inferencias o resumo eh impresso.
const int INTERVALO_RELATORIO = 20;
#endif

#if ENVIAR_IMAGEM_SERIAL
// Marcador de sincronizacao: sequencia improvavel de aparecer em texto ASCII normal,
// usada pelo script Python para saber onde comeca cada frame binario.
const uint8_t MARCADOR_FRAME[4] = {0xAA, 0x55, 0xAA, 0x55};

void enviar_frame_serial(camera_fb_t* fb) {
  Serial.write(MARCADOR_FRAME, sizeof(MARCADOR_FRAME));
  Serial.write(fb->buf, fb->len);
}
#endif

void setup() {
  Serial.begin(115200);

  if (!psramFound()) {
    Serial.println("PSRAM não encontrada!");
    return;
  }

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_GRAYSCALE;
  config.frame_size   = FRAMESIZE_96X96;
  config.jpeg_quality = 12;
  config.fb_count     = 1;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Falha ao inicializar câmera: 0x%x\n", err);
    return;
  }

  Serial.println("Câmera inicializada!");

  tensor_arena = (uint8_t*) ps_malloc(TENSOR_ARENA_SIZE);
  model = tflite::GetModel(modelo_gestos);

  static tflite::MicroMutableOpResolver<7> resolver;
  resolver.AddQuantize();
  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddReshape();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddDequantize();

  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, TENSOR_ARENA_SIZE
  );
  interpreter = &static_interpreter;

  interpreter->AllocateTensors();

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Modelo carregado com sucesso!");

#if MEDIR_METRICAS
  // O consumo da arena so fica disponivel APOS o AllocateTensors(),
  // que eh quem planeja o uso da memoria. A diferenca entre alocado e
  // utilizado mostra a margem de seguranca adotada.
  size_t arena_usada = interpreter->arena_used_bytes();
  Serial.println();
  Serial.println(F("=== CONSUMO DE MEMORIA ==="));
  Serial.print(F("Arena alocada  : "));
  Serial.print(TENSOR_ARENA_SIZE);
  Serial.println(F(" bytes"));
  Serial.print(F("Arena utilizada: "));
  Serial.print(arena_usada);
  Serial.println(F(" bytes"));
  Serial.print(F("Ocupacao       : "));
  Serial.print((float) arena_usada / TENSOR_ARENA_SIZE * 100.0f, 1);
  Serial.println(F("%"));
  Serial.print(F("Margem livre   : "));
  Serial.print(TENSOR_ARENA_SIZE - arena_usada);
  Serial.println(F(" bytes"));
  Serial.println(F("=========================="));
  Serial.println();
#endif
}

void loop() {
#if MEDIR_METRICAS
  unsigned long t_ciclo = micros();
#endif

  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Falha ao capturar imagem");
    return;
  }

#if ENVIAR_IMAGEM_SERIAL
  enviar_frame_serial(fb);
#endif

  for (int i = 0; i < 96 * 96; i++) {
    input->data.f[i] = fb->buf[i] / 255.0f;
  }

  esp_camera_fb_return(fb);

#if MEDIR_METRICAS
  // Mede exclusivamente o Invoke(), sem captura nem normalizacao.
  // Corresponde a definicao usual de "tempo de inferencia" na
  // literatura de TinyML, permitindo comparacao com outros trabalhos.
  unsigned long t_inicio = micros();
  interpreter->Invoke();
  unsigned long latencia_us = micros() - t_inicio;

  soma_latencia_us += latencia_us;
  if (latencia_us < min_latencia_us) min_latencia_us = latencia_us;
  if (latencia_us > max_latencia_us) max_latencia_us = latencia_us;
  total_inferencias++;
#else
  interpreter->Invoke();
#endif

  int classe_detectada = 0;
  float maior_confianca = output->data.f[0];

  for (int i = 1; i < NUM_CLASSES; i++) {
    if (output->data.f[i] > maior_confianca) {
      maior_confianca = output->data.f[i];
      classe_detectada = i;
    }
  }

  if (maior_confianca > 0.70f) {
    Serial.print("Gesto: ");
    Serial.print(CLASSES[classe_detectada]);
    Serial.print(" | Confiança: ");
    Serial.print(maior_confianca * 100);
    Serial.println("%");
  } else {
    Serial.println("Nenhum gesto detectado");
  }

#if MEDIR_METRICAS
  soma_ciclo_us += micros() - t_ciclo;

  if (total_inferencias % INTERVALO_RELATORIO == 0) {
    float media_lat = (float) soma_latencia_us / total_inferencias;
    float media_ciclo = (float) soma_ciclo_us / total_inferencias;

    Serial.println();
    Serial.println(F("=== METRICAS DE DESEMPENHO ==="));
    Serial.print(F("Amostras            : "));
    Serial.println(total_inferencias);
    Serial.println(F("-- Inferencia (Invoke) --"));
    Serial.print(F("Latencia media      : "));
    Serial.print(media_lat / 1000.0f, 2);
    Serial.println(F(" ms"));
    Serial.print(F("Latencia minima     : "));
    Serial.print(min_latencia_us / 1000.0f, 2);
    Serial.println(F(" ms"));
    Serial.print(F("Latencia maxima     : "));
    Serial.print(max_latencia_us / 1000.0f, 2);
    Serial.println(F(" ms"));
    // Limite superior teorico: quantas inferencias caberiam em 1 s se o
    // dispositivo executasse APENAS o Invoke(). Nao eh a taxa observada.
    Serial.print(F("Limite teorico      : "));
    Serial.print(1000000.0f / media_lat, 1);
    Serial.println(F(" inferencias/s"));
    Serial.println(F("-- Ciclo completo --"));
    Serial.print(F("Tempo medio do ciclo: "));
    Serial.print(media_ciclo / 1000.0f, 2);
    Serial.println(F(" ms"));
    Serial.print(F("FPS do sistema      : "));
    Serial.println(1000000.0f / media_ciclo, 1);
    // Avisos para nao se reportar um numero enganoso no artigo.
    if (DELAY_LOOP_MS > 0) {
      Serial.print(F("AVISO: delay de "));
      Serial.print(DELAY_LOOP_MS);
      Serial.println(F(" ms ativo -- o FPS acima nao"));
      Serial.println(F("       reflete a capacidade real do dispositivo."));
    }
#if ENVIAR_IMAGEM_SERIAL
    Serial.println(F("AVISO: streaming serial ativo -- transmitir 9.216"));
    Serial.println(F("       bytes por quadro reduz o FPS do sistema."));
#endif
    Serial.println(F("=============================="));
    Serial.println();
  }
#endif

  if (DELAY_LOOP_MS > 0) {
    delay(DELAY_LOOP_MS);
  }
}