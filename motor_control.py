"""
Motor Control — Pi5
Birebir test/direksiyon.py + test/dcmotorTest.py baz alındı.

Servo:  GPIO12  → AngularServo (gpiozero, lgpio factory)
Motor:  GPIO23→IN1, GPIO24→IN2, GPIO18→ENA
"""

import os, atexit
os.environ["GPIOZERO_PIN_FACTORY"] = "lgpio"

from gpiozero import Motor, DigitalOutputDevice, AngularServo
import warnings
warnings.filterwarnings("ignore")


# ── Pin tanımları (test dosyalarından birebir) ──
SERVO_PIN  = 12   # BCM12 (fiziksel pin 32)
MOTOR_FWD  = 23   # IN1
MOTOR_BWD  = 24   # IN2
MOTOR_ENA  = 18   # ENA

# Servo mekanik ayar
SERVO_OFFSET = 4.0   # derece — servo deadband asimi + sag tarafli mekanik bias
                     # Hala sol sapiyorsa: 'x' tusu ile artir (1 adim = 1 derece)
                     # Sag sapmaya baslarsa: 'z' tusu ile azalt
SERVO_MAX_ANGLE = 75  # derece — virajda tam kilitlenme icin artirildi (60 → 75)
SERVO_SLEW_RATE = 12.0  # derece/frame — ani ziplamalar onlenir (yavas gecis)


class SteeringServo:
    """Yazilimsal PWM veya lgpio uzerinden yuksek torklu AngularServo.
    Slew rate limiter ile ani aci degisimlerini onler (mekanik stres azalir).
    offset: calisma sirasinda '[' ve ']' tuslari ile ayarlanabilir."""

    def __init__(self):
        # Pulse width: AngularServo dahili -90..+90 map'i kullanir
        # Gercek kullanim +-60 derece ile sinirli (SERVO_MAX_ANGLE)
        # Pulse width daraltirsak -90 bile gonderilse fiziksel limit daha dar olur
        # 0.0007 - 0.0023 => Fiziksel olarak ~-75..+75 arasi, yazilim +-60 ile sinirlar
        self.servo = AngularServo(SERVO_PIN, min_pulse_width=0.0003, max_pulse_width=0.0027)
        self._value = 0.0
        self._angle = 0.0
        self.offset = SERVO_OFFSET   # Instance degiskeni — '[' ']' tuslari ile ayarlanabilir
        self.center()                # offset ONCE tanimlanmali, center() onu kullaniyor

    def set(self, value):
        """value: -1.0 (full left) ... 0 (duz) ... +1.0 (full right)"""
        value = max(-1.0, min(1.0, float(value)))
        target = value * SERVO_MAX_ANGLE + self.offset
        target = max(-SERVO_MAX_ANGLE, min(SERVO_MAX_ANGLE, target))

        # Slew rate limiter: ani ziplamayı onle
        diff = target - self._angle
        if abs(diff) > SERVO_SLEW_RATE:
            target = self._angle + SERVO_SLEW_RATE * (1.0 if diff > 0 else -1.0)

        self.servo.angle = target
        self._value = value
        self._angle = target

    def center(self):
        self.set(0.0)

    def detach(self):
        """Titremeyi önlemek için bağlantı kesilir."""
        try:
            self.servo.detach()
        except Exception:
            pass

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, deg):
        """Dogrudan derece ile ayarla (hailo_app uyumu)."""
        deg = max(-SERVO_MAX_ANGLE, min(SERVO_MAX_ANGLE, float(deg)))
        target_deg = deg + self.offset
        target_deg = max(-SERVO_MAX_ANGLE, min(SERVO_MAX_ANGLE, target_deg))

        # Slew rate limiter
        diff = target_deg - self._angle
        if abs(diff) > SERVO_SLEW_RATE:
            target_deg = self._angle + SERVO_SLEW_RATE * (1.0 if diff > 0 else -1.0)

        self.servo.angle = target_deg
        self._angle = target_deg
        self._value = target_deg / SERVO_MAX_ANGLE if SERVO_MAX_ANGLE > 0 else 0.0

    def close(self):
        self.detach()


class DriveMotor:
    """ENA sabit HIGH — hız IN1/IN2 üzerinden Motor.forward(speed)."""

    def __init__(self):
        self.motor_yon = Motor(forward=MOTOR_FWD, backward=MOTOR_BWD)
        self.motor_ena = DigitalOutputDevice(MOTOR_ENA)
        self._speed = 0.0
        self.motor_ena.off()

    def forward(self, speed=0.5):
        speed = max(0.0, min(1.0, speed))
        self._speed = speed
        self.motor_ena.on()
        self.motor_yon.forward(speed)

    def backward(self, speed=0.5):
        speed = max(0.0, min(1.0, speed))
        self._speed = -speed
        self.motor_ena.on()
        self.motor_yon.backward(speed)

    def stop(self):
        self._speed = 0.0
        self.motor_yon.stop()
        self.motor_ena.off()

    def brake(self):
        self.stop()

    @property
    def speed(self):
        return self._speed

    def close(self):
        self.stop()
        self.motor_ena.close()


class MotorController:
    def __init__(self):
        self.steering = SteeringServo()
        self.drive = DriveMotor()
        atexit.register(self._cleanup)

    def set_steering(self, value):
        self.steering.set(value)

    def set_speed(self, speed):
        if speed >= 0:
            self.drive.forward(speed)
        else:
            self.drive.backward(abs(speed))

    def stop(self):
        self.steering.center()
        self.drive.stop()

    def emergency_stop(self):
        self.steering.center()
        self.drive.brake()

    def _cleanup(self):
        try:
            self.drive.stop()
            self.steering.detach()
        except Exception:
            pass

    def close(self):
        self._cleanup()
