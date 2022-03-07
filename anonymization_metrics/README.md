# Anonymization Metrics
This toolkit encapsulates multiple python implementations of anonymization metrics from the state-of-the-art.

It is also the implementation of our paper https://hal.inria.fr/hal-02907918 .
```
Mohamed Maouche, Brij Mohan Lal Srivastava, Nathalie Vauquier, Aurélien Bellet, Marc Tommasi, Emmanuel Vincent.
A comparative study of speech anonymization metrics. INTERSPEECH 2020, Oct 2020, Shanghai, China
```
## Requirements
requirements.txt is included in the project root node.
```py
matplotlib==3.1.2
numpy==1.18.1
pandas==1.0.5
seaborn==0.10.1
```

## Usage

```sh
python3 compute_metrics.py [-h] -s SCORE_FILE [-dc] [-dl] [-oc OUTPUT_FILE_CLLR] [-ol OUTPUT_FILE_LINK] [--omega OMEGA] [--tag TAG] [-wopt] [-oopt OUTPUT_FILE_OPT]
```
### Options
```
  -h, --help            show the help message and exit
  -s SCORE_FILE         path to score file
  -dc                   flag: draw the APE-plot
  -dl                   flag: draw the Linkability Plot
  -oc OUTPUT_FILE_CLLR  output path of the png and pdf file (default is ape_<score_file>
  -ol OUTPUT_FILE_LINK  output path of the png and pdf file (default is link_<score_file>
  --omega OMEGA         prior ratio for linkability metric (default is 1)
  --tag TAG             tag before the values in the order of linkaiblity,cllr,cmin,eer
  -wopt                 flag: Write the calibrated scores
  -oopt OUTPUT_FILE_OPT output path of the calibrated scores (default is opt_<score_file>
```

### Output
```
<tag>,matedMean,nonMatedMean,matedStd,nonMatedStd,linkaiblity,cllr,cmin,eer
```
### Simple example
```sh
echo "id,matedMean,nonMatedMean,matedStd,nonMatedStd,linkaiblity,cllr,cmin,eer" && python compute_metrics.py -s  examples/scores/scores_1 --tag "s1"
```
It outputs
```
id,matedMean,nonMatedMean,matedStd,nonMatedStd,linkaiblity,cllr,cmin,eer
s1,3.0077259531982308,1.4462666150080412,0.11097579928810934,0.7184156570301649,0.5063636363636363,1.2601658561658864,0.06332590373152217,0.016666666666666666
```
### Multi score-files example
```sh
cd examples
bash run_examples.sh
```


## Ackowledgement

This code is partly based on code-snippets of:
* [BOSARIS toolkit](https://sites.google.com/site/bosaristoolkit/)
* [cllr](https://gitlab.eurecom.fr/nautsch/cllr)

This work was supported by the French National Research Agency under project DEEP-PRIVACY (ANR-18-CE23-0018) and by the European Union’s Horizon 2020 Research and Innovation Program under Grant Agreement No.825081 COMPRISE (https://www.compriseh2020.eu/). Experiments presented in this paper were carried out using the Grid'5000 testbed, supported by a scientific interest group hosted by Inria and including CNRS, RENATER and several Universities as well as other organizations (see https://www.grid5000.fr).
