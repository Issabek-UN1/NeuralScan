/*
 * ============================================================================
 * NeuralScan: Edge-AI Hybrid 3D Metrology Scanner
 * Microcontroller Firmware for Arduino Mega 2560 + L298P Dual Motor Shield
 * 
 * Hardware Target:
 *  - Board: Arduino Mega 2560 (ATmega2560)
 *  - Driver: L298P Dual Full-Bridge Shield
 *  - Stepper Motor: Mitsumi M35SP-7T (Bipolar, 7.5 deg/step, 48 steps/rev, 10 Ohm/phase)
 *  - Laser Module: KY-008 650nm line laser (TTL controlled)
 *
 * CRITICAL SAFETY REGULATION:
 *  At 5-7V input, holding current in the 10-Ohm coil exceeds 0.5-0.7A, causing
 *  catastrophic thermal runaway and motor burn-out within 15 seconds.
 *  EVERY step sequence MUST de-energize the H-bridge coils immediately
 *  following the 55ms mechanical impulse:
 *      digitalWrite(PIN_ENA, LOW);
 *      digitalWrite(PIN_ENB, LOW);
 *
 * Serial Protocol: 115200 Baud, 8N1
 * Commands:
 *  'S' -> Step forward 1 step (7.5 deg), de-energize coils
 *  's' -> Step backward 1 step (7.5 deg), de-energize coils
 *  'R' -> Full 360-degree rotation (48 steps), de-energize coils
 *  'L' -> Laser ON
 *  'l' / 'O' -> Laser OFF
 *  'P' -> Ping / Health Check (Returns "PONG: NEURALSCAN_READY")
 *  'E' -> Emergency cutoff (Disables all outputs)
 * ============================================================================
 */

#include <Arduino.h>

// --- Pin Definitions (L298P Shield on Arduino Mega) ---
const uint8_t PIN_ENA  = 10; // Bridge A Enable / Speed (PWM)
const uint8_t PIN_ENB  = 11; // Bridge B Enable / Speed (PWM)
const uint8_t PIN_DIRA = 12; // Bridge A Direction
const uint8_t PIN_DIRB = 13; // Bridge B Direction

// --- Laser Module Control Pin ---
const uint8_t PIN_LASER = 7; // KY-008 TTL trigger (Active HIGH)

// --- Stepper Timing & Geometry Constants ---
const uint16_t STEP_PULSE_MS = 55;   // Pulse width strictly 50-60 ms to prevent coil overheating
const uint16_t INTER_STEP_MS = 30;   // Cooldown between continuous steps in full rotation
const uint8_t  STEPS_PER_REV = 48;   // 360 deg / 7.5 deg = 48 steps

// --- Bipolar Stepper 4-State Full-Step Phase Table ---
// Phase 0: A+ (DIRA=HIGH), B+ (DIRB=HIGH)
// Phase 1: A- (DIRA=LOW),  B+ (DIRB=HIGH)
// Phase 2: A- (DIRA=LOW),  B- (DIRB=LOW)
// Phase 3: A+ (DIRA=HIGH), B- (DIRB=LOW)
const uint8_t STEP_SEQUENCE[4][2] = {
    {HIGH, HIGH},
    {LOW,  HIGH},
    {LOW,  LOW},
    {HIGH, LOW}
};

static int8_t current_phase = 0;
static long   step_counter  = 0;

/**
 * Immediate coil cutoff to enforce zero idle dissipation.
 */
inline void deenergize_coils() {
    digitalWrite(PIN_ENA, LOW);
    digitalWrite(PIN_ENB, LOW);
}

/**
 * Executes a single 7.5-degree step with safe coil pulse & immediate cutoff.
 * @param forward true for clockwise (positive theta), false for counter-clockwise
 */
void execute_single_step(bool forward = true) {
    if (forward) {
        current_phase = (current_phase + 1) & 0x03; // modulo 4
        step_counter = (step_counter + 1) % STEPS_PER_REV;
    } else {
        current_phase = (current_phase - 1 + 4) & 0x03;
        step_counter = (step_counter - 1 + STEPS_PER_REV) % STEPS_PER_REV;
    }

    // 1. Set directional H-bridge states
    digitalWrite(PIN_DIRA, STEP_SEQUENCE[current_phase][0]);
    digitalWrite(PIN_DIRB, STEP_SEQUENCE[current_phase][1]);

    // 2. Energize coils for calibrated movement impulse
    digitalWrite(PIN_ENA, HIGH);
    digitalWrite(PIN_ENB, HIGH);
    delay(STEP_PULSE_MS);

    // 3. MANDATORY SAFETY CUTOFF: De-energize coils to prevent thermal overload
    deenergize_coils();
}

/**
 * Executes a full 360-degree rotation (48 steps).
 */
void execute_full_revolution() {
    for (uint8_t i = 0; i < STEPS_PER_REV; i++) {
        execute_single_step(true);
        delay(INTER_STEP_MS);
    }
}

/**
 * Returns turntable to initial zero position.
 */
void return_to_zero() {
    long steps_back = step_counter % STEPS_PER_REV;
    if (steps_back > 0) {
        for (long i = 0; i < steps_back; i++) {
            execute_single_step(false);
            delay(INTER_STEP_MS);
        }
        step_counter = 0;
    }
    deenergize_coils();
}

void setup() {
    // Configure motor control pins as outputs
    pinMode(PIN_ENA, OUTPUT);
    pinMode(PIN_ENB, OUTPUT);
    pinMode(PIN_DIRA, OUTPUT);
    pinMode(PIN_DIRB, OUTPUT);

    // Configure laser pin
    pinMode(PIN_LASER, OUTPUT);

    // HARD INITIAL SAFETY STATE: zero power to coils & laser
    deenergize_coils();
    digitalWrite(PIN_LASER, LOW);

    // Initialize USB UART
    Serial.begin(115200);
    while (!Serial) {
        ; // Wait for serial port connection (Mega native)
    }

    Serial.println(F("BOOT: NEURALSCAN_CONTROLLER_READY"));
    Serial.println(F("CONFIG: M35SP-7T 48STEPS L298P ACTIVE_CUTOFF=ON"));
}

void loop() {
    if (Serial.available() > 0) {
        char cmd = (char)Serial.read();

        switch (cmd) {
            case 'S': // Step Forward
                execute_single_step(true);
                Serial.print(F("OK: STEP "));
                Serial.println(step_counter);
                break;

            case 's': // Step Reverse
                execute_single_step(false);
                Serial.print(F("OK: STEP_REV "));
                Serial.println(step_counter);
                break;

            case 'R': // Full 360-degree rotation
                execute_full_revolution();
                Serial.println(F("OK: ROTATE 360"));
                break;

            case 'H': // Home / Return to Zero
            case 'Z':
                return_to_zero();
                Serial.println(F("OK: HOME_ZERO"));
                break;

            case 'L': // Laser ON
                digitalWrite(PIN_LASER, HIGH);
                Serial.println(F("OK: LASER ON"));
                break;

            case 'l': // Laser OFF
            case 'O':
                digitalWrite(PIN_LASER, LOW);
                Serial.println(F("OK: LASER OFF"));
                break;

            case 'P': // Ping / Keepalive
                Serial.println(F("PONG: NEURALSCAN_READY"));
                break;

            case 'E': // Emergency cutoff
                deenergize_coils();
                digitalWrite(PIN_LASER, LOW);
                Serial.println(F("OK: EMERGENCY_CUTOFF"));
                break;

            case '\r':
            case '\n':
            case ' ':
                // Ignore whitespace
                break;

            default:
                Serial.print(F("ERR: UNKNOWN_CMD '"));
                Serial.print(cmd);
                Serial.println(F("'"));
                break;
        }
    }
}
