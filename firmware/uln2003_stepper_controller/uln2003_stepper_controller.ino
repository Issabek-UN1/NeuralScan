#include <Arduino.h>
#include <Stepper.h>

// --- Параметры шаговика ---
const int STEPS_PER_REV = 2048;      // 2048 шагов на 1 полный оборот
const int SCAN_STEPS_COUNT = 48;      // 48 ракурсов (по 7.5 градусов)
const int MOTOR_SPEED_RPM = 12;       // Скорость 12 об/мин (быстро и с высоким моментом)

// --- Пин лазера KY-008 ---
const uint8_t PIN_LASER = 7;

// --- Проверенная инициализация Stepper ---
Stepper myStepper(STEPS_PER_REV, 8, 10, 9, 11);

static long current_step_pos = 0;
static int  current_scan_view = 0;

// Обесточивание катушек в покое (мотор и плата остаются холодными)
inline void deenergize_coils() {
    digitalWrite(8, LOW);
    digitalWrite(9, LOW);
    digitalWrite(10, LOW);
    digitalWrite(11, LOW);
}

// Шаг сканирования на 7.5 градусов
void execute_scan_view_step(bool forward = true) {
    if (forward) {
        current_scan_view = (current_scan_view + 1) % SCAN_STEPS_COUNT;
    } else {
        current_scan_view = (current_scan_view - 1 + SCAN_STEPS_COUNT) % SCAN_STEPS_COUNT;
    }

    long target_step_pos = ((long)current_scan_view * STEPS_PER_REV) / SCAN_STEPS_COUNT;
    long delta = target_step_pos - current_step_pos;

    if (forward && delta <= 0) delta += STEPS_PER_REV;
    if (!forward && delta >= 0) delta -= STEPS_PER_REV;

    myStepper.step(delta);
    current_step_pos = target_step_pos;

    deenergize_coils();
}

// Полный оборот 360°
void execute_full_revolution() {
    myStepper.step(STEPS_PER_REV);
    current_step_pos = 0;
    current_scan_view = 0;
    deenergize_coils();
}

// Возврат в начальную нулевую точку
void return_to_zero() {
    if (current_step_pos > 0) {
        myStepper.step(-current_step_pos);
        current_step_pos = 0;
        current_scan_view = 0;
        deenergize_coils();
    }
}

void setup() {
    pinMode(PIN_LASER, OUTPUT);
    digitalWrite(PIN_LASER, LOW);

    myStepper.setSpeed(MOTOR_SPEED_RPM);
    deenergize_coils();

    Serial.begin(115200);
    while (!Serial) { ; }

    Serial.println(F("BOOT: NEURALSCAN_CONTROLLER_READY"));
}

void loop() {
    if (Serial.available() > 0) {
        char cmd = (char)Serial.read();

        switch (cmd) {
            case 'S': // Шаг вперед на 7.5°
                execute_scan_view_step(true);
                Serial.print(F("OK: STEP "));
                Serial.println(current_scan_view);
                break;

            case 's': // Шаг назад
                execute_scan_view_step(false);
                Serial.print(F("OK: STEP_REV "));
                Serial.println(current_scan_view);
                break;

            case 'R': // Полный оборот
                execute_full_revolution();
                Serial.println(F("OK: ROTATE 360"));
                break;

            case 'H': // Возврат в ноль
            case 'Z':
                return_to_zero();
                Serial.println(F("OK: HOME_ZERO"));
                break;

            case 'L': // Включить лазер
                digitalWrite(PIN_LASER, HIGH);
                Serial.println(F("OK: LASER ON"));
                break;

            case 'l': // Выключить лазер
            case 'O':
                digitalWrite(PIN_LASER, LOW);
                Serial.println(F("OK: LASER OFF"));
                break;

            case 'P': // Пинг
                Serial.println(F("PONG: NEURALSCAN_READY"));
                break;

            case 'E': // Аварийный сброс
                deenergize_coils();
                digitalWrite(PIN_LASER, LOW);
                Serial.println(F("OK: EMERGENCY_CUTOFF"));
                break;

            default:
                break;
        }
    }
}