#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <ESP32Servo.h>

// ----------- REDE / ENDPOINTS -----------
#define WIFI_SSID   "Wokwi-GUEST"
#define WIFI_PASS   ""
#define API_BASE    "http://192.168.0.73:5000"
#define URL_STATUS   API_BASE "/status-irrigacao"
#define URL_LEITURAS API_BASE "/leituras/batch"

// -------------- PINAGEM -----------------
// Sensores analógicos de temperatura e umidade do ar (substituindo DHT22)
#define PIN_TEMP_AR_ADC  36  // Sensor de temperatura analógico
#define PIN_UMID_AR_ADC  39  // Potenciômetro simulando umidade do ar

// LM35 (substituindo DS18B20 - analógico no GPIO33)
#define PIN_LM35_ADC  33

// LDR (divisor)
#define PIN_LDR_ADC   34
// Umidade Solo (Potenciômetro simulando sensor capacitivo)
#define PIN_SOIL_ADC  35
// MQ-135 (A0)
#define PIN_MQ135_ADC 32
// Servo (janela)
#define PIN_SERVO_PWM 14
// Relé (irrigação)
#define PIN_RELE_IN   16

// LEDs (opcional)
#define LED_RELE_PIN     27
#define LED_BOMBA_PIN     2
#define LED_FOSFORO_PIN  26
#define LED_POTASSIO_PIN 25

// ------------- OBJETOS ---------------
// LCD I2C - endereço 0x27, 16 colunas, 2 linhas
LiquidCrystal_I2C lcd(0x27, 16, 2);
Servo servoJanela;

// --------- CÓDIGOS DE SENSOR (BD) ---------
#define COD_TEMP_AR   "DHT22_TEMP_AR_ESP32_01"
#define COD_UMID_AR   "DHT22_UMID_AR_ESP32_01"
#define COD_TEMP_SOLO "LM35_TEMP_SOLO_01"
#define COD_UMID_SOLO "CAP_SOLO_ESP32_01"
#define COD_LUX       "LDR_LUX_ESP32_01"
#define COD_CO2       "MQ135_CO2_ESP32_01"

// --------- PARÂMETROS / REGRAS ----------
const uint32_t INTERVALO_MS   = 5000;
const float    TH_SOIL_PCT    = 55.0;
const float    TH_TEMP_AR_ABR = 30.0;
const float    TH_CO2_ABR     = 1200.0;

unsigned long ultimoCiclo = 0;

// ---------- CONVERSÕES SIMPLES ----------
float mq135_to_ppm(int adc) { return map(adc, 0, 4095, 350, 2000); }
float soil_to_percent(int adc) {
  float pct = (adc / 4095.0f) * 100.0f;
  return constrain(pct, 0.0f, 100.0f);
}
float ldr_to_index(int adc) {
  float pct = (adc / 4095.0f) * 100.0f;
  return constrain(pct, 0.0f, 100.0f);
}
float lm35_to_celsius(int adc) {
  float voltage = (adc / 4095.0f) * 3.3f;
  float temperature = voltage / 0.01f;
  
  if (temperature > 50.0f) {
    temperature = 20.0f + ((temperature - 50.0f) * 0.3f);
  }
  
  return constrain(temperature, 15.0f, 35.0f);
}
float temp_sensor_to_celsius(int adc) {
  float temp = map(adc, 0, 4095, 15, 40);
  return constrain(temp, 15.0f, 40.0f);
}
float humidity_to_percent(int adc) {
  float humidity = map(adc, 0, 4095, 20, 90);
  return constrain(humidity, 20.0f, 90.0f);
}

// -------------- WIFI / LCD --------------
void conectaWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("Conectando ao Wi-Fi");
  for (int i=0; i<60 && WiFi.status()!=WL_CONNECTED; i++) { delay(250); Serial.print("."); }
  Serial.println(WiFi.status()==WL_CONNECTED? " OK!" : " FALHA (offline).");
}

void lcdInit() {
  Serial.println("Inicializando LCD I2C...");
  
  // Inicializa LCD I2C (sem Wire.begin explícito - já é chamado automaticamente)
  lcd.init();
  lcd.backlight();
  lcd.clear();
  delay(100);
  
  // Mostra mensagem inicial
  lcd.setCursor(0, 0);
  lcd.print("Estufa Intelig.");
  lcd.setCursor(0, 1);
  lcd.print("Inicializando...");
  
  Serial.println("LCD I2C inicializado - deve aparecer no display");
  delay(3000);
}

void lcdResumo(float umAr, float phFake, bool irrig, float tAr) {
  // Display virtual no Serial Monitor (mais confiável)
  Serial.println("==================== DISPLAY ====================");
  Serial.printf("║ Temp AR: %2.0f°C    Umidade AR: %2.0f%%         ║\n", tAr, umAr);
  Serial.printf("║ Irrigacao: %-3s     pH Solo: %-4.1f          ║\n", 
                irrig ? "SIM" : "NAO", phFake);
  Serial.println("==================================================");
  
  // Tenta atualizar LCD físico também (caso funcione)
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("U:");
  if (umAr > 0) {
    lcd.print((int)umAr);
  } else {
    lcd.print("--");
  }
  lcd.print("% T:");
  if (tAr > 0) {
    lcd.print((int)tAr);
  } else {
    lcd.print("--");
  }
  lcd.print("C");
  
  lcd.setCursor(0, 1);
  lcd.print("IRR:");
  lcd.print(irrig ? "SIM" : "NAO");
  lcd.print(" pH:");
  lcd.print(phFake, 1);
}

// -------------- LEITURAS --------------
bool lerSensoresAnalogicos(float &tAr, float &uAr) {
  float sumTemp = 0, sumHum = 0;
  int validReadings = 0;
  
  for (int i = 0; i < 5; i++) {
    int tempRaw = analogRead(PIN_TEMP_AR_ADC);
    int humRaw = analogRead(PIN_UMID_AR_ADC);
    
    if (tempRaw >= 0 && tempRaw <= 4095 && humRaw >= 0 && humRaw <= 4095) {
      sumTemp += temp_sensor_to_celsius(tempRaw);
      sumHum += humidity_to_percent(humRaw);
      validReadings++;
    }
    delay(20);
  }
  
  if (validReadings > 0) {
    tAr = sumTemp / validReadings;
    uAr = sumHum / validReadings;
    return true;
  }
  return false;
}

bool lerLM35(float &t) {
  float sum = 0;
  int validReadings = 0;
  
  for (int i = 0; i < 10; i++) {
    int adc = analogRead(PIN_LM35_ADC);
    if (adc >= 0 && adc <= 4095) {
      sum += lm35_to_celsius(adc);
      validReadings++;
    }
    delay(10);
  }
  
  if (validReadings > 0) {
    t = sum / validReadings;
    return true;
  }
  return false;
}

// -------------- API AUX ----------------
bool podeIrrigarAgora() {
  if (WiFi.status()!=WL_CONNECTED) return true;
  HTTPClient http; http.begin(URL_STATUS);
  int code=http.GET(); if(code!=200){ http.end(); return true; }
  StaticJsonDocument<128> doc;
  if (deserializeJson(doc, http.getString())) { http.end(); return true; }
  http.end();
  return (bool)(doc["pode_irrigar"] | true);
}

void enviaBatch(float tAr, float uAr, float tSolo, float uSoloPct, float luxIdx, float co2ppm) {
  if (WiFi.status()!=WL_CONNECTED) { Serial.println("Wi-Fi off, nao enviando."); return; }
  StaticJsonDocument<512> doc; JsonArray arr = doc.to<JsonArray>();
  auto add=[&](const char* cod, float valor, const char* unit){
    JsonObject o=arr.createNestedObject(); o["codigo_sensor"]=cod; o["valor"]=valor;
    JsonObject raw=o.createNestedObject("raw_data"); raw["unit"]=unit;
  };
  add(COD_TEMP_AR, tAr, "C");
  add(COD_UMID_AR, uAr, "%");
  add(COD_TEMP_SOLO, tSolo, "C");
  add(COD_UMID_SOLO, uSoloPct, "%");
  add(COD_LUX,      luxIdx, "idx");
  add(COD_CO2,      co2ppm, "ppm");

  String payload; serializeJson(arr, payload);
  HTTPClient http; http.begin(URL_LEITURAS); http.addHeader("Content-Type","application/json");
  int code=http.POST(payload); String resp=http.getString(); http.end();
  Serial.printf("POST %s -> %d | %s\n", URL_LEITURAS, code, resp.c_str());
}

// ------------------ SETUP -------------------
void setup() {
  Serial.begin(115200); 
  delay(1000);
  
  Serial.println("=== SISTEMA DE ESTUFA INTELIGENTE ===");
  Serial.println("Versão com sensores analógicos");
  Serial.println("=====================================");

  // Configura pinos de saída
  pinMode(PIN_RELE_IN, OUTPUT); digitalWrite(PIN_RELE_IN, LOW);
  pinMode(LED_RELE_PIN, OUTPUT); pinMode(LED_BOMBA_PIN, OUTPUT);
  pinMode(LED_FOSFORO_PIN, OUTPUT); pinMode(LED_POTASSIO_PIN, OUTPUT);
  digitalWrite(LED_RELE_PIN, LOW); digitalWrite(LED_BOMBA_PIN, LOW);
  digitalWrite(LED_FOSFORO_PIN, LOW); digitalWrite(LED_POTASSIO_PIN, LOW);

  // Inicializa sensores analógicos
  Serial.println("Configurando sensores analógicos...");
  pinMode(PIN_TEMP_AR_ADC, INPUT);
  pinMode(PIN_UMID_AR_ADC, INPUT);
  pinMode(PIN_LM35_ADC, INPUT);
  pinMode(PIN_LDR_ADC, INPUT);
  pinMode(PIN_SOIL_ADC, INPUT);
  pinMode(PIN_MQ135_ADC, INPUT);
  
  // Teste dos sensores analógicos
  Serial.println("Testando sensores de ar...");
  int testTempAr = analogRead(PIN_TEMP_AR_ADC);
  int testHumAr = analogRead(PIN_UMID_AR_ADC);
  Serial.printf("Sensores AR - Temp ADC: %d, Umid ADC: %d\n", testTempAr, testHumAr);

  // Inicializa Servo
  Serial.println("Inicializando Servo...");
  ESP32PWM::allocateTimer(0); ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2); ESP32PWM::allocateTimer(3);
  servoJanela.setPeriodHertz(50);
  servoJanela.attach(PIN_SERVO_PWM, 500, 2400);
  servoJanela.write(0);

  // Inicializa LCD I2C (biblioteca inicializa I2C automaticamente)
  Serial.println("Inicializando LCD I2C...");
  lcdInit();
  
  // Conecta WiFi
  Serial.println("Conectando WiFi...");
  conectaWiFi();
  
  Serial.println("=== SETUP CONCLUÍDO ===\n");
}

// ------------------- LOOP -------------------
void loop() {
  if (millis()-ultimoCiclo < INTERVALO_MS) { 
    delay(50);
    return; 
  }
  ultimoCiclo = millis();

  Serial.println("=== NOVO CICLO DE LEITURAS ===");

  float umAr, tAr, tSolo;
  bool okAr = lerSensoresAnalogicos(tAr, umAr);
  bool okLM35 = lerLM35(tSolo);

  int ldrRaw  = analogRead(PIN_LDR_ADC);
  int soilRaw = analogRead(PIN_SOIL_ADC);
  int mqRaw   = analogRead(PIN_MQ135_ADC);
  int lm35Raw = analogRead(PIN_LM35_ADC);
  int tempArRaw = analogRead(PIN_TEMP_AR_ADC);
  int humArRaw = analogRead(PIN_UMID_AR_ADC);
  
  Serial.printf("RAW -> LDR=%d | SOIL=%d | MQ=%d | LM35=%d | TempAr=%d | HumAr=%d\n", 
                ldrRaw, soilRaw, mqRaw, lm35Raw, tempArRaw, humArRaw);
                
  float luxIdx   = ldr_to_index(ldrRaw);
  float uSoloPct = soil_to_percent(soilRaw);
  float co2ppm   = mq135_to_ppm(mqRaw);

  bool abrirJanela = (okAr && (tAr > TH_TEMP_AR_ABR)) || (co2ppm > TH_CO2_ABR);
  servoJanela.write(abrirJanela ? 90 : 0);

  bool seco  = (uSoloPct < TH_SOIL_PCT);
  bool pode  = podeIrrigarAgora();
  bool ok    = okAr && okLM35;
  bool irrigar = seco && pode && ok;
 
  digitalWrite(PIN_RELE_IN, irrigar ? HIGH : LOW);
  digitalWrite(LED_RELE_PIN, irrigar ? HIGH : LOW);
  digitalWrite(LED_BOMBA_PIN, irrigar ? HIGH : LOW);
  digitalWrite(LED_FOSFORO_PIN, abrirJanela ? HIGH : LOW);
  digitalWrite(LED_POTASSIO_PIN, seco ? HIGH : LOW);

  float phFake = 6.5 + (random(-5,6) / 10.0f);
  lcdResumo(ok?umAr:0, phFake, irrigar, ok?tAr:0);

  Serial.printf("Sensores AR -> %s\n",
    okAr ? (String("T=")+String(tAr,1)+"C H="+String(umAr,0)+"%").c_str() : "ERRO");
  Serial.printf("LM35 -> %s\n", 
    okLM35 ? (String("Tsolo=")+String(tSolo,1)+"C").c_str() : "ERRO");

  if (ok) {
    enviaBatch(tAr, umAr, tSolo, uSoloPct, luxIdx, co2ppm);
  } else {
    Serial.println("Leitura invalida; nao enviando.");
  }
  
  Serial.println("=== FIM DO CICLO ===\n");
}