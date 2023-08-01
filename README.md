# Identifying the drivers of spread in climate change mitigation scenarios
### Introduction
This code involves a variance decomposition analysis of climate change mitigation scenarios, in order to understand the spread in these scenarios. The scenario spread of variables such as primary energy carriers, energy use in end-used sectors, emissions and more is attributed to three drivers: (1) climate targets, (2) model differences, and (3) other scenario assumptions.

### Requirements
The code reads in the IPCC AR6 scenario database that is publicly available [here](https://zenodo.org/record/5886912) (version Nov 2022). Both the full data as well as the metadata are required. These files can be stored in any file directory, but make sure that this directory is specified in the `input.yml` file. Other required datafiles are already inside this Github repository. For the calculations, it only requires standard Python packages like `pandas`, `tqdm` and `yaml`. For plotting, packages such as `plotly` are used.

### Usage
##### Setup
As mentioned, the first step is to save the IPCC AR6 scenario data into a location, that you subsequently put into the `input.yml` file under the entry `location_ipcc`. In this file, you can also set other parameters: `threshold_dataremoval` dictates how many scenarios a model should at least have to be part of this analysis (default `10`), `sample_size_per_ms` indicates the number of model-scenario pairs that is drawn for each model-climate category label when creating the samples (default `3000`), `resampling` is the number of times the analysis is redone (including sampling), in order to average out potential stochasticity (default `100`), and `removal_c8` (either `yes` or `no`) indicates whether the C8 climate category should be included or not (default `yes`).

##### Calculations
The calculations are done in the `Main.ipynb`, which reads in a Class objects from the files `class_datahandling.py` and `class_decomposition.py` and step-wise reads in data, performs the decomposition, and saves the data into the `Data` folder. Of particular interest is the subfolder `Data/Output_files`, where a netcdf file called `Variances.nc` is stored. This file includes the relative importance of the three drivers - the outcomes of this research.

##### Plotting
Plotting scripts are provided in the folder `Plotting Scripts`, for each individual figure separately, annotated by the name of the file. At the top of each plotting script, which are jupyter notebooks, you can find a number of parameters to tune the plotting.

### References
For more information, please have a look at our preprint: https://www.researchsquare.com/article/rs-2073170/v1. Also, please cite this reference if you use this code or data in your own research.
