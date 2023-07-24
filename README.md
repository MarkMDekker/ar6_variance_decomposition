# Variance decomposition of climate change mitigation scenarios
## Introduction
This code involves a variance decomposition analysis of climate change mitigation scenarios, in order to understand the spread in these scenarios. The scenario spread of variables such as primary energy carriers, energy use in end-used sectors, emissions and more is attributed to three drivers: (1) climate targets, (2) model differences, and (3) other scenario assumptions.

## Requirements
The code reads in the IPCC AR6 scenario database that is publicly available [here](https://zenodo.org/record/5886912) (version Nov 2022). Other required datafiles are already inside this Github repository. For the calculations, it only requires standard Python packages like `pandas`, `tqdm` and `yaml`. For plotting, slightly advanced packages such as `plotly` are used.

## Usage
#### Setup
The first step is to save the IPCC AR6 scenario data into a location, that you subsequently put into the `input.yml` file under the entry `location_ipcc`. In this file, you can also set other parameters: `threshold_dataremoval` dictates how many scenarios a model should at least have to be part of this analysis, `sample_size_per_ms` indicates the number of model-scenario pairs that is drawn for each model-climate category label when creating the samples, `resampling` is the number of times the analysis is redone (including sampling), in order to average out potential stochasticity, and `removal_c8` (either `yes` or `no`) indicates whether the C8 climate category should be included or not.

#### Calculations
The calculations are done in the `Main.ipynb`, which reads in a Class object from the file `ClassVarianceDecomp_v2.py` and step-wise reads in data, performs the decomposition, and saves the data into the `Data` folder (particularly in the `Variances.nc` netcdf file).

#### Plotting
Plotting scripts are provided in the folder `Plotting Scripts`, for each individual figure separately, annotated by the name of the file. At the top of each plotting script, which are jupyter notebooks, you can find a number of parameters to tune the plotting.

## References
For more information, please have a look at our preprint: https://www.researchsquare.com/article/rs-2073170/v1. Also, please cite this reference if you use this code or data in your own research.
