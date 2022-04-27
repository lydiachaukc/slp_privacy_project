## Install
run `./install.sh`

## Running the baseline system
1. `cd baseline` 
2. run `./run_baseline.sh`. In run.sh, to download models and data the user will be requested the password which is provided during the Challenge registration. Please visit the [challenge website](https://www.voiceprivacychallenge.org/) to register and get access to the baseline models.

## Running the proposed system
1. `cd baseline` 
2. To run the proposed model for cross gender voice anonymization, run `./run_cross_gender.sh`. In run.sh, to download models and data the user will be requested the password which is provided during the Challenge registration. Please visit the [challenge website](https://www.voiceprivacychallenge.org/) to register and get access to the baseline models.
3. To run the proposed model for same gender voice anonymization, run `./run_same_gender.sh`

#### Training data
The dataset for anonymization system traing consists of subsets from the following corpora*:
* [LibriTTS](http://www.openslr.org/60/) - train-clean-360
---------------------------------------------------------------------------

## Code References
1. [Data pipline and baseline system](https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2020)
2. [Evaluation metrics](https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2022)
3. [StarGANV2](https://github.com/clovaai/stargan-v2)
