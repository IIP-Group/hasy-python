# Alias-Free Oscillator Synchronization via Additive Synthesis

This repository provides supplementary material for the paper "Alias-Free Oscillator Synchronization via Additive Synthesis" presented at DAFx26:

* [`add_sync.ipynb`](add_sync.ipynb) : reference implementation in Python of the proposed method, provided as Jupyter Notebook.
* [`audio_examples/`](audio_examples/) : example wav files, generated using the Python implementation.

### Setup (macOS)

Create virtual environment and install requirements:
```bash
python3 -m venv addsync_env
source addsync_env/bin/activate
pip install -r requirements.txt
```

If desired, install additional development requirements:
```bash
pip install -r requirements-dev.txt
nbstripout --install
```

### Development Testing

```bash
pytest tests/
```
