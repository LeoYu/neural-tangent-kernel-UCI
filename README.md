# Performance of Nerual Tangent Kernel (NTK) on UCI datasets
This is code for the UCI experiment in paper "[Harnessing the Power of Infinitely Wide Deep Nets on Small-data Tasks](https://arxiv.org/abs/1910.01663)"
## Prerequisites
Python3, numpy, sklearn
### Setup
Download and decompress the pre-processed datasets used in paper "[Do we need hundreds of classifiers to solve real world classification problems?](http://jmlr.org/papers/volume15/delgado14a/delgado14a.pdf)" by running
```
bash setup.sh
```
## Running the tests
```
python UCI.py -max_tot N -max_dep dep -file output_file
```
Use option `-max_tot N` to skip datasets with size larger than `N`.

Use option `-max_dep dep` to set the maximum depth allowed for NTK.

Use option `-file output_file` to set the output file.
## Comparison
Compare with other classifiers using results reported by "[Do we need hundreds of classifiers to solve real world classification problems?](http://jmlr.org/papers/volume15/delgado14a/delgado14a.pdf)" from the link blow:
- http://persoal.citius.usc.es/manuel.fernandez.delgado/papers/jmlr/results.txt

Details are listed in paper "[Harnessing the Power of Infinitely Wide Deep Nets on Small-data Tasks](https://arxiv.org/abs/1910.01663)".
