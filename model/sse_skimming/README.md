# Skimming Project – Extract Significant Sentences

This code can be used to extract significant sentences from a pdf.

## Installation

```bash
pip install poetry
cd sse_skimming
poetry install
```

## Usage of `example.py`


```bash
python -m sse_skimming \
    src=path/to/file.pdf \
    dst=path/to/output/dir/for/visualization
```

For example, if ran with the [Longchecker pdf](https://arxiv.org/pdf/2112.01640v1), it should return the following outut:

![Image of the output for the longchecker PDF](longchecker.png)
