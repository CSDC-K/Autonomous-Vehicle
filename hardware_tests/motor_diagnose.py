"""
Motor teşhis — direkt lgpio (gpiozero yok, thread yok)
Önceki testlerde L298N soğuk + her iki testte ~5s'de durdu
→ gpiozero/lgpio thread zaman aşımı şüphesi
"""
import lgpio, time

MOTOR_IN1 = 23   # ileri
MOTOR_IN2 = 24   # geri
MOTOR_ENA = 18   # enable

def test():
    h = lgpio.gpiochip_open(0)
    for pin in [MOTOR_IN1, MOTOR_IN2, MOTOR_ENA]:
        lgpio.gpio_claim_output(h, pin, 0)

    print("\n═══ TEST: Direkt lgpio — gpiozero/thread yok ═══")
    print("IN1=HIGH, IN2=LOW, ENA=HIGH → motor ileri")
    lgpio.gpio_write(h, MOTOR_IN2, 0)
    lgpio.gpio_write(h, MOTOR_ENA, 1)
    lgpio.gpio_write(h, MOTOR_IN1, 1)

    input(">>> Motor döndü mü? — Enter ile devam: ")

    print("Motor 60 saniye çalışacak, her 5 saniyede gözlem sorusu...")
    t0 = time.time()
    stopped_at = None
    for i in range(1, 13):
        time.sleep(5)
        elapsed = time.time() - t0
        r = input(f">>> {elapsed:.0f}s — motor hâlâ dönüyor mu? (e/h): ")
        if r.strip().lower() == 'h':
            stopped_at = elapsed
            print(f"  ✗ {elapsed:.0f}. saniyede durdu → HARDWARE SORUNU (güç kaynağı / motor sargısı)")
            break
    else:
        print("  ✓ 60 saniye boyunca hiç durmadı → gpiozero/lgpio thread problemi")

    # Pinleri sıfırla
    for pin in [MOTOR_IN1, MOTOR_IN2, MOTOR_ENA]:
        lgpio.gpio_write(h, pin, 0)
    lgpio.gpiochip_close(h)

    print()
    if stopped_at:
        input(">>> Motor sargısına (motor gövdesi) dokunun — ısınıyor mu? (e/h): ")
        print("\nSONUÇ: Güç kaynağı voltajını motor çalışırken ölçün.")
        print("       Beklenen: ≥7V. Düşüyorsa akü/adaptör yetersiz.")
    else:
        print("SONUÇ: Problem gpiozero/lgpio threadlerindeydi.")
        print("       motor_control.py zaten direkt lgpio'ya geçirilecek.")

    print("\nTEST TAMAMLANDI")

test()
