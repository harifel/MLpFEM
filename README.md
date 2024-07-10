# MLpFEM - towards Machine Learning based parameter calibration for Finite Element Modelling

This repository contains code for the research project between [NGI](https://www.ngi.no/eng) and the Graz University of Technology
[Institute of Soil Mechanics, Foundation Engineering and Computational Geotechnics](https://www.tugraz.at/en/institutes/ibg/home) to explore the possibility to automatically calibrate parameters for constitutive models for geotechnical Finite Element Modelling with Machine Learning Algorithms.

## Folder structure

```
MLpFEM
├── data                                  - data from simulated and real soil mechanical tests
├── graphics                              - saved graphics from running scripts in src
├── src                                   - folder that contains the python script files
│   ├── main.py                           - main script for now
├── environment.txt                       - dependency file to use
├── LICENSE                               - Github license file to specify the license of the repository 
├── README.md                             - repository description
```

## Requirements

The environment is set up using `python 3.11.1`.

To achieve this, create an environment named `venv` and label it as MLpFEM (or any other desired name).
```bash
C:\Users\haris\Documents\GitHub\MLpFEM>C:\Users\haris\AppData\Local\Programs\Python\Python311\python -m venv MLpFEM
```

Activate the new environment with:
```bash
MLpFEM\Scripts\activate
```

Then, install all packages using 'environment.txt', for example:
```bash
(MLpFEM) C:\Users\haris\Documents\GitHub\MLpFEM>py -m pip install -r environment.txt
```
If you encounter pip errors, install the libraries manually.
