# Model Swapping
### Authors: Patrick Lavin, Jonathan Beard

This repository contains all code and data needed to reproduce the results and plots from our PMBS 2020 submission. 

You can find information on using the simulator in `ModelSwapping.ipynb`. You can find the code to produce our plots in `CollectDataPMBS20.ipybb`. This simulator is derived from [SVE-Cachesim](https://github.com/ARM-software/Methodology_for_ArmIE_SVE/tree/master/sve-cachesim). 

You can find our custom version of [Meabo](https://github.com/ARM-software/meabo) in `trace-generation/`

You can find the Meabo traces (generated by DynamoRIO memtrace) in `traces/`.

You can find the simulator output used for this project in `DataPMBS20/`, as well out the plots themselves. 

You can find the plots for the paper in `DataPMBS20/`. This is also where simulation data should live, but it is too big for github. See `DataPMBS20/MISSINGFILES.md/` for more. 

### Some Notes

* Cloning this project requires `git lfs`. 

* Help information is available for most of the classes and functions in this project. For instance, you can use `help(PhaseDetector.PhaseDetector)` to get info on using the PhaseDetector class. 

* This project have some extra python dependencies but they should all be readily available with pip. You will need `numpy`, `pandas`, `numba`, and `bitvec`.
