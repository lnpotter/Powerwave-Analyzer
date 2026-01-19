# PowerWave Analyzer

PowerWave Analyzer is a Python tool for offline analysis of electrical waveforms stored as CSV files. It computes FFT, harmonic spectrum, RMS values, and Total Harmonic Distortion (THD) for voltage or current waveforms.

## Features

- Load time-domain waveform from CSV file.
- Compute FFT and magnitude spectrum.
- Detect fundamental frequency and harmonic components.
- Compute RMS and THD for the waveform.

## Installation & Usage

```bash
git clone https://github.com/lnpotter/Powerwave-Analyzer.git
cd Powerwave-Analyzer

python -m venv venv
# Linux/Mac: source venv/bin/activate
# Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

# Usage (example CLI/terminal):
python cli.py --file sine.csv --column voltage --sampling-rate 1000 --fundamental 60 --lang en

# Usage (example Streamlit):
streamlit run app.py
```
