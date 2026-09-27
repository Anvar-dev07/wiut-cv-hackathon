Pandora — Traffic Event Detection (WIUT Hackathon 2026, CV Track)
Fixed-camera trafik videolarida hodisalarni aniqlash (Part A) va avariya xavfini oldindan
baholash (Part B, bonus) uchun yechim.
O'rnatish va ishga tushirish
Bash
Model tashqi (pretrained) og'irliklardan foydalanmaydi — barcha logika qoidaga asoslangan
(rule-based), shuning uchun weights/ papkasi va download.sh skripti talab qilinmaydi.
python run_submission.py --videos /data/test --out predictions.json
python evaluate.py --pred predictions.json --validate-only
Namuna videolardagi natijalar predictions_samples.json faylida keltirilgan.
Yondashuv
Pipeline to'liq rule-based (o'rgatilgan model ishlatilmagan):
Fon ayirish (background subtraction) — OpenCV BackgroundSubtractorMOG2 yordamida
har kadrdagi harakatlanuvchi ob'ektlar (foreground mask) aniqlanadi.
Tozalash — morfologik open/close amallari bilan shovqin olib tashlanadi.
Ob'ekt aniqlash — mask'dagi konturlar (contours) orqali bounding box'lar chiqariladi,
maydoni bo'yicha kichik shovqinlar filtrlanadi.
Qoidaga asoslangan klassifikatsiya — har bir ob'ektning pozitsiyasi, o'lchami va
kadrlar orasidagi o'zgarishi asosida (masalan, yo'l markazidagi kichik ob'ekt →
jaywalking, uzoq vaqt harakatsiz yirik ob'ekt → stopped_vehicle, keskin piksel farqi →
accident kabi) hodisa segmentlari yig'iladi.
Birlashtirish — bir xil klassdagi yaqin segmentlar birlashtiriladi, juda qisqa
(sub-second) bo'laklar tashlab yuboriladi.
Part B (RiskEstimator) — kadrlar orasidagi harakat intensivligi, "brakelash" signali
(harakat tezligining keskin pasayishi) va yo'lning markaziy hududidagi zichlik asosida
0–1 oralig'ida xavf balli hisoblanadi; faqat o'tgan kadrlardan foydalanadi (causal).
Tashqi datasetlar/modellar: ishlatilmagan — barcha mantiq OpenCV'ning klassik
computer vision funksiyalari asosida qo'lda yozilgan.
Determinizm
Kod tasodifiy (random) qiymatlardan foydalanmaydi — barcha bosqichlar (fon ayirish,
kontur aniqlash, qoidalar) deterministik. Bir xil video ikki marta ishga tushirilganda
bir xil natija beradi (float darajasidagi hisoblash farqlaridan tashqari).
Jamoa
Anvar Turdiyev — captain, model/CV logikasi (detection pipeline)
Komiljon Matmurodov — ma'lumotlar va testlash (video annotatsiya, evaluate.py)
Akbarbek Xaytbayev — website va hujjatlar (live demo, README)
GitHub: [link] | Sayt: [live demo linki]
Ma'lumot haqida eslatma
Namuna videolar (samples/*.mp4) tashkilotchilar ko'rsatmasiga ko'ra ushbu repositoryga
yuklanmagan.
