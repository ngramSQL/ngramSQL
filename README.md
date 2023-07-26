##

We plan on making a web-hosted instance of ngramSQL available in the following days. We will publish access information here.

## Installation

To install ngramSQL locally (on a Linux machine), please perform the following steps:

  - Make sure to have ```poetry``` installed.

  - Clone this repository to your machine.
  - Extract the ```corpora.zip``` into a new directory named ```corpora```.
  - Navigate to the ```ngramsql``` folder, it contains the ```pyproject.toml``` file.
  - Run ```poetry install```.
  - Run ```poetry shell```.
  - Run ```streamlit run main.py```. The first startup might take a while, the following ones will be faster.
  - Navigate to the *Guided Tour* tab and explore the system.
