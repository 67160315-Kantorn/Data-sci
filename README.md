# BMW Used Car Price Prediction

โปรเจกต์ Regression เพื่อทำนายราคารถ BMW มือสอง (`price`) จากข้อมูลรถ เช่น ปีรถ เลขไมล์ รุ่นรถ ประเภทเกียร์ ประเภทน้ำมัน ภาษี MPG และขนาดเครื่องยนต์

## วิธีรัน
```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```

## โครงสร้างไฟล์
```bash
bmw_regression_project/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── data/
│   └── bmw.csv
└── models/
    ├── best_model.pkl
    └── metadata.json
```
