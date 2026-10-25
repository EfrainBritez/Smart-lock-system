/*
  Wio Terminal - receptor de resultados de reconocimiento facial (etapa A)

  Recibe por USB serial lineas enviadas por face_system.py:
    OK,<nombre>,<score>     persona reconocida
    UNKNOWN,,<score>        rostro no registrado
    NONE                    no hay rostro en cuadro

  Requisitos:
    - Placa: "Seeed Wio Terminal" (paquete Seeed SAMD Boards)
    - Libreria: Seeed_Arduino_LCD (incluye TFT_eSPI)
*/
#include <TFT_eSPI.h>

TFT_eSPI tft;

enum State { WAITING, NO_FACE, KNOWN, UNKNOWN_FACE };
State state = WAITING;
String currentName = "";
String currentScore = "";
String line = "";

void drawScreen() {
  uint16_t bg;
  switch (state) {
    case KNOWN:        bg = TFT_DARKGREEN; break;
    case UNKNOWN_FACE: bg = TFT_MAROON;    break;
    default:           bg = TFT_BLACK;     break;
  }
  tft.fillScreen(bg);
  tft.setTextColor(TFT_WHITE, bg);
  tft.setTextDatum(MC_DATUM);

  switch (state) {
    case WAITING:
      tft.drawString("Esperando a la PC...", 160, 120, 4);
      break;
    case NO_FACE:
      tft.drawString("Sin rostro", 160, 120, 4);
      break;
    case KNOWN:
      tft.drawString("Bienvenido/a", 160, 80, 4);
      tft.drawString(currentName, 160, 125, 6);
      tft.drawString("similitud " + currentScore, 160, 190, 2);
      break;
    case UNKNOWN_FACE:
      tft.drawString("Desconocido", 160, 110, 4);
      tft.drawString("similitud " + currentScore, 160, 170, 2);
      break;
  }
}

void handleLine(const String &msg) {
  State previous = state;
  String previousName = currentName;

  if (msg.startsWith("OK,")) {
    int c1 = msg.indexOf(',');
    int c2 = msg.indexOf(',', c1 + 1);
    if (c2 < 0) return;
    currentName = msg.substring(c1 + 1, c2);
    currentScore = msg.substring(c2 + 1);
    state = KNOWN;
  } else if (msg.startsWith("UNKNOWN,")) {
    int c2 = msg.indexOf(',', msg.indexOf(',') + 1);
    currentScore = (c2 >= 0) ? msg.substring(c2 + 1) : "";
    currentName = "";
    state = UNKNOWN_FACE;
  } else if (msg == "NONE") {
    state = NO_FACE;
    currentName = "";
  } else {
    return;  // linea no reconocida
  }

  if (state != previous || currentName != previousName) {
    drawScreen();
    if (state == KNOWN) {
      tone(WIO_BUZZER, 1500, 150);
    } else if (state == UNKNOWN_FACE) {
      tone(WIO_BUZZER, 400, 400);
    }
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(WIO_BUZZER, OUTPUT);
  tft.begin();
  tft.setRotation(3);
  drawScreen();
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      line.trim();
      if (line.length() > 0) handleLine(line);
      line = "";
    } else if (c != '\r' && line.length() < 64) {
      line += c;
    }
  }
}
