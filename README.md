## Scim: Intelligent Skimming Support for Scientific Papers

By Raymond Fok, Hita Kambhamettu, Luca Soldaini, Jonathan Bragg, Kyle Lo, Andrew Head, Marti A. Hearst, and Daniel S. Weld

Abstract: Researchers need to keep up with immense literatures, though it is time-consuming and difficult to do so. In this paper, we investigate the role that intelligent interfaces can play in helping researchers skim papers, that is, rapidly reviewing a paper to attain a cursory understanding of its contents. After conducting formative interviews and a design probe, we suggest that skimming aids should aim to thread the needle of highlighting content that is simultaneously diverse, evenly-distributed, and important. We introduce \scim{}, a novel intelligent skimming interface that reifies this aim, designed to support the skimming process by highlighting salient paper contents to direct a skimmer's focus. Key to the design is that the highlights are faceted by content type, evenly-distributed across a paper, with a density configurable by readers at both the global and local level. We evaluate \scim{} with an in-lab usability study and deployment study, revealing how skimming aids can support readers throughout the skimming experience and yielding design considerations and tensions for the design of future intelligent skimming tools.

## UI
`reader/` includes implementation of the main Scim user interface. Skimming functionality is built upon infrastructure for a previous augmented reading interface, previously known as ScholarPhi [Github](https://github.com/allenai/scholarphi).

## Model
`model/sse_skimming` implements a data processing pipeline including PDF processing and tokenization, salient sentence extraction and classification, and other heuristics to improve highlight quality before rendering in the Scim UI.

## Weak Supervision
`model/sse_skimming/weak_label` implements a weak supervision paradigm using Snorkel for faceted highlight extraction, including a set of labeling functions designed over research papers in NLP.

## Citation
If you use our work, please cite our preprint

```
@misc{dec-2022-scim,
    title={Scim: Intelligent Skimming Support for Scientific Papers},
    author={Fok, Raymond and Kambhamettu, Hita and Soldaini, Luca and Bragg, Jonathan and Lo, Kyle and Head, Andrew and Hearst, Marti A. and Weld, Daniel S.},
    url = {https://arxiv.org/abs/2205.04561},
    publisher = {arXiv},
    year = {2022}
}
```
