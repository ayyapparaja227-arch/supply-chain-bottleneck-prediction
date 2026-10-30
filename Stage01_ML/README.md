# ML Stage — Supply Chain Logistics Control Tower

**Purpose:** shipment delay-risk classification

This stage follows the five-role pattern in the supplied OncologyPrecisionMedicine_MiniDemo:
1. `01_data_engineer.py`
2. `02_eda_engineer.py`
3. `03_ml_engineer.py`
4. `04_evaluation_engineer.py`
5. `05_integration_engineer.py`

Run from this folder, in numeric order:
```bash
python 01_data_engineer.py
python 02_eda_engineer.py
python 03_ml_engineer.py
python 04_evaluation_engineer.py
python 05_integration_engineer.py
```

Install the common dependencies: `pip install pandas numpy scikit-learn matplotlib joblib`.

The sample telemetry is synthetic/demo data, not live port data. This educational mini demo is not connected to live logistics systems.
