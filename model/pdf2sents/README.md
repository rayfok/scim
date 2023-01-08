# PDF2Sents

This repository contains a command line tool that leverages [mmda](https://github.com/allenai/mmda) to convert PDF files to sentences. Command line usage:

```bash
python -m pdf2sents \
    src=path/to/pdf_file.pdf \  # Can be a URL or a local path
    dst=pdf_file.jsonl \        # Where to save output
    mode=sent
```

## Installation

You can install pdf2sents from S2 internal PyPI repo:

```bash
pip install -i https://pip.s2.allenai.org/simple/ pdf2sents
```


## Release

Run the following commands (this will only work while on VPN).

```bash
poetry publish --build -r s2-pypi
```
