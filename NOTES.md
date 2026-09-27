# Boshlang'ich pipeline — nima ishlaydi, nima TODO

## G'oya
Kamera qo'zg'almas bo'lgani uchun sahnani **bir marta** kalibrovka qilamiz
(yo'l chegarasi, piyoda o'tish joylari, har bir lane va uning yo'nalishi),
keyin har bir videoda pretrained YOLO + ByteTrack bilan mashina/piyodalarni
kuzatib, ularning traektoriyasidan (tezlik, yo'nalish, joylashuv) qoidalar
orqali hodisalarni chiqaramiz. Hech qanday model o'qitilmagan — hammasi
tayyor og'irliklar + geometriya.

## Ishga tushirish tartibi

```bash
pip install -r requirements.txt

# 1) samples papkangizni ko'rib chiqing
python tools/eda_videos.py samples/

# 2) sahnani KALIBROVKA qiling — BIR MARTA, istalgan sample video bilan
python tools/calibrate_scene.py samples/<clip>.mp4
# -> scene_config.json yaratiladi (repo ildizida saqlanadi)

# 3) YOLO og'irligini bir marta yuklab oling (keyin weights/ ga ko'chiring)
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
mkdir -p weights && mv yolo11n.pt weights/

# 4) sinab ko'ring
python run_submission.py --videos samples --out predictions_samples.json --team your-team
python evaluate.py --pred predictions_samples.json --validate-only
```

O'zingiz `samples/` videolarni qo'lda belgilab (`my_labels.json`,
`examples/ground_truth.json` bilan bir xil formatda), haqiqiy F1 ni
ko'rish uchun:
```bash
python evaluate.py --pred predictions_samples.json --gt my_labels.json --per-video
```

## Hozir ishlaydigan klasslar (`src/events.py`)
- **stopped_vehicle** — mashina yo'l ustida 10s+ deyarli qo'zg'almasa
- **congestion** — yo'ldagi mashinalarning katta qismi bir vaqtda sekin/to'xtagan
- **wrong_way** — mashina tezligi lane'ning belgilangan yo'nalishiga qarama-qarshi
  (lane kalibrovkasi shart)
- **jaywalking** — piyoda yo'l ustida, lekin crosswalk tashqarisida
- **accident** — ikki mashina bbox'i bir-biriga tegib qoladi
- **near_miss** — ikki mashina juda yaqinlashadi va biri keskin tormoz beradi

Bularning barchasi **kuchli boshlang'ich (baseline)** — aniqlik uchun
`src/events.py` tepasidagi tunable konstantalarni (masalan
`STOPPED_SPEED_PX_S`, `NEAR_CLOSE_PX`) o'z sample videolaringizga qarab
sozlashingiz kerak bo'ladi.

## Hali qilinmagan klasslar (va nega)
Bular sof track pozitsiyasi/tezligidan chiqmaydi, qo'shimcha signal kerak:

| Klass | Nima kerak |
|---|---|
| `red_light` | svetofor rangini aniqlash (kichik crop + rang klassifikatori) + stop-line kesish |
| `stop_line` | svetofor holati + mashina stop-line oldida to'xtashi |
| `illegal_u_turn`, `illegal_turn` | burilish geometriyasi (traektoriya burchagi) + qaysi burilish taqiqlanganini bilish (belgi/yo'l chizig'i) |
| `solid_line_crossing` | yo'l chizig'ini aniqlash (masalan Hough transform yoki segmentatsiya) |
| `road_obstacle` | statik, harakatlanmaydigan g'ayrioddiy ob'ekt — alohida klass yoki anomaly-detection kerak |
| `fire_smoke` | rang/tutun tekstura klassifikatori (frame-level) |

Ustuvorlik tavsiyasi: avval yuqoridagi 6 ta klassni **o'z videolaringizda
yaxshi ishlaydigan** darajaga yetkazing (real F1 macro'ga eng katta ta'sir
shu yerdan keladi), so'ng vaqt qolsa `red_light` va `solid_line_crossing`ni
qo'shing — ular nisbatan sodda (rang tekshirish, chiziq aniqlash).

## Tezlik haqida
- Part A: `solution.py` dagi `FRAME_STRIDE` (hozir 2) — kerak bo'lsa 3-5 ga
  oshiring, vaqt byudjeti = video davomiyligining 3 baravari.
- Part B: `src/risk.py` dagi `DETECT_EVERY_N` (hozir 3) xuddi shu maqsadda —
  har kadrda emas, har N kadrda YOLO ishlaydi, orasida tezlik bo'yicha
  ekstrapolatsiya qilinadi.
- Yakuniy topshiriqda internet yo'q — `weights/yolo11n.pt` repo ichida
  bo'lishi SHART (yoki `weights/download.sh`).

## Determinizm
`ultralytics` va `bytetrack` odatda deterministik, lekin `README.md`da
seed'laringizni aniq yozing (tasodifiylik ishlatsangiz).
