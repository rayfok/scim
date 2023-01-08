# Scholar Reader

The user interface, API, and data processing scripts for an
augmented PDF reader application.

This repository hosts code for three subprojects: the user
interface, API, and data processing scripts.  To learn about
each of these projects and how to run the code for each of
them, see the `README.md` file in the relevant directory.

Key directories include:

* `api/`: the web API that provides data about entities in
papers and bounding boxes for those entities.
* `data-processing/`: a set of data processing scripts that
extract entities and their bounding boxes from papers.
* `ui/`: the user interface for the augmented PDF reader.

Note: Only the ui/ directory is updated and required for the skimming application Scim.

## Quick Start

In order to run the reader app locally (pointing to the production api) you only need to do

```bash
cd ui
npm install
npm start
```

See [ui/README.md](ui/README.md) for more details on how to use the reader.
