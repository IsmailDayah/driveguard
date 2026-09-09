// DriveGuard - ESP32 alert unit firmware
//
// Listens on USB serial (115200 baud) for a single character '0'..'3' giving the
// fatigue risk stage, and drives the alert hardware accordingly:
//
//   '0' NORMAL    green LED steady, everything else off
//   '1' DROWSY    amber LED steady, one short chirp + one vibration pulse
//   '2' FATIGUE   amber LED blinking, double-beep pattern, vibration pulses
//   '3' CRITICAL  red LED fast blink, continuous siren beeping, vibration on
//   'x' STANDBY   everything off — sent by the host on shutdown so the unit
//                 goes dark when the detector exits (stage 0 still lights the
//                 green LED, which is not the same as "off")
//
// All patterns are non-blocking (millis-based) so serial stays responsive.

#include <Arduino.h>

const int PIN_RED = 25;
const int PIN_AMBER = 26;
const int PIN_GREEN = 27;
const int PIN_BUZZER = 14;
const int PIN_VIBRA = 32;

int stage = 0;
unsigned long tNow = 0;
unsigned long stageEntered = 0;

void allOff() {
  digitalWrite(PIN_RED, LOW);
  digitalWrite(PIN_AMBER, LOW);
  digitalWrite(PIN_GREEN, LOW);
  digitalWrite(PIN_BUZZER, LOW);
  digitalWrite(PIN_VIBRA, LOW);
}

void setup() {
  pinMode(PIN_RED, OUTPUT);
  pinMode(PIN_AMBER, OUTPUT);
  pinMode(PIN_GREEN, OUTPUT);
  pinMode(PIN_BUZZER, OUTPUT);
  pinMode(PIN_VIBRA, OUTPUT);
  allOff();
  digitalWrite(PIN_GREEN, HIGH);
  Serial.begin(115200);
  Serial.println("DriveGuard alert unit ready (send 0-3)");
}

void loop() {
  // ---- read stage commands ----
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c >= '0' && c <= '3') {
      int newStage = c - '0';
      if (newStage != stage) {
        stage = newStage;
        stageEntered = millis();
        allOff();
        Serial.print("stage=");
        Serial.println(stage);
      }
    } else if (c == 'x' || c == 'X') {
      // STANDBY: go completely dark. stage = -1 matches no case in the
      // switch below, so nothing re-lights the outputs after allOff().
      stage = -1;
      stageEntered = millis();
      allOff();
      Serial.println("stage=off");
    }
  }

  tNow = millis();
  unsigned long inStage = tNow - stageEntered;

  switch (stage) {
    case 0: {  // NORMAL — green steady
      digitalWrite(PIN_GREEN, HIGH);
      break;
    }
    case 1: {  // DROWSY — amber steady; chirp + vibration pulse on entry (600 ms)
      digitalWrite(PIN_AMBER, HIGH);
      bool entryPulse = inStage < 600;
      digitalWrite(PIN_BUZZER, entryPulse && (inStage < 200) ? HIGH : LOW);
      digitalWrite(PIN_VIBRA, entryPulse ? HIGH : LOW);
      break;
    }
    case 2: {  // FATIGUE — amber blink 500 ms; double beep + vibration each 2 s
      digitalWrite(PIN_AMBER, (tNow / 500) % 2 ? HIGH : LOW);
      unsigned long ph = tNow % 2000;                 // 2-second pattern
      bool beep = (ph < 120) || (ph >= 240 && ph < 360);
      digitalWrite(PIN_BUZZER, beep ? HIGH : LOW);
      digitalWrite(PIN_VIBRA, ph < 400 ? HIGH : LOW);
      break;
    }
    case 3: {  // CRITICAL — red fast blink; siren beeping; vibration on
      digitalWrite(PIN_RED, (tNow / 150) % 2 ? HIGH : LOW);
      unsigned long ph = tNow % 600;                  // urgent 600 ms siren cycle
      digitalWrite(PIN_BUZZER, ph < 300 ? HIGH : LOW);
      digitalWrite(PIN_VIBRA, HIGH);
      break;
    }
  }
}
