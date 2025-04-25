# Musformer




Although misleading, but the name seemed nice. The goal of this project is to try out multiple source separation models in speechbrain. The idea is to replicate building a model in the waveform domain directly. We see that multiple models try to solve the music source separation as a problem and most state-of-the-art models reach an SiSDR ratio of 9 to tackle the same. 

Most models working on this problem came into existence as a solution to the SDX (Sound Demixing) challenge. A few of the existing solutions are as follows: 
- **MMDenseLSTM**: Combines dense blocks with LSTMs for lightweight waveform separation.
- **Demucs (v1/v2)**: U-Net with bidirectional LSTMs; later versions add transformers.
- **ConvTasNet**: Efficient temporal convolutional network with learned encoder/decoder.
- **DPRNNet**: Dual-path RNN with attention, offering SOTA results at higher compute costs.
- **BandSplitRNN**: Separates audio into frequency bands processed by independent RNNs before recombination.
- **Wave-U-Net**: Adapts medical imaging's U-Net architecture for waveform-based source separation with learned down/upsampling.
- **Open-Unmix**: Spectrogram-based separation model using three-layer BiLSTMs with industry-standard implementation.
- **ResUNetDecouple**: U-Net variant with residual connections that decouples magnitude and phase processing.
- **TDCN++**: Improved temporal convolutional network with global skip connections and stacked dilation patterns.
- **Spleeter**: Facebook's lightweight CNN-based separator using spectrogram masking with pretrained models.

Our goal in this mini-project is to focus on the waveform domain. We try to implement four of these in their basest versions in order to learn how these architectures come into life for vocal separation from music. Along with that, we use multiple attempts to enhance our working. 

This is a project done by Mohnish Mirchandani, under professor Mirco Ravanelli, at Concordia University. Tagalong code for this can be found at: https://github.com/mirchandani-mohnish/Musformer

## Disclaimers
This notebook refers to the tip of the iceberg of multiple approaches tried along the way in order to reach here. It provides for a summary of the underlying work done around trials and errors in speech separation. Additionally, the informal nature of the text in this report is to keep the content somewhat light-hearted in order to ensure engagement with the reader...

The Github Repository is a good starting point for anyone who dares to explore the rest of the iceberg...

This report is intended to be self explanatory and so all the data you need to learn about these models should be referenced in the report itself. Additionally, all the models presented here have their intended recipes which you can use to cook your own models... 

Bonne Apetit!

## Step1 : Gaining Context (Literature Review)

### Gimme the Data!!
First things first, lets talk about the dataset and the models a bit. MUSDB18 is the benchmark dataset for music source separation, containing 150 full-track recordings (100 for training, 50 for test) with isolated stems for vocals, drums, bass, and other instruments. It provides professionally mixed 44.1kHz stereo audio, enabling evaluation of waveform-domain separation models. The dataset covers diverse genres and production styles, making it ideal for testing real-world generalization. It has become the standard benchmark for models like Open-Unmix, Demucs, and D3Net, with SI-SDR and SDR as primary metrics. The included Python toolbox (musdb) provides data loading, evaluation, and stem mixing utilities. Musdb is however dependent on something known as STEM files i.e. music tensor files combined and compressed into one. We use the kaggle uploaded version of the dataset which saves us the trouble of downloading, unzipping and uploading the same. You can find the dataset here: https://www.kaggle.com/datasets/jakerr5280/musdb18-music-source-separation-dataset


### The MusDB Package
One of the most helpful things available in the community to enhance the friendliness of this challenge is the [package provided by sigsep](https://github.com/sigsep/sigsep-mus-db/tree/master/musdb). Given that all the files are STEM format files i.e. a format extension which clubs multiple vectors into a cohesive mp4 format, the musdb package helps us read those using [FFMPEG](https://ffmpeg.org/). This allows us to load the vectors directly into the file without the need of creating a manifest file to read the particular values.


## How We Doin...? (Methodology)

In order to learn and explore the different model approaches, we draw up each of the available models in speechbrain and try to retrofit them to a simple vocal separation task. The expectation being that the data more often than not, the model is able to reach state-of-the-art...ish levels. The steps we follow are as below:
1. Learn and draw up the model architecture using speechbrain modules wherever possible
2. Run them on a particular set of data values with particular hyperparameters
3. Compare the results

### Train.py
The `hparams` and `train.py` files both are heavily sourced from speechbrain recipes at : https://github.com/speechbrain/speechbrain/tree/develop/recipes

Speechbrain provides a framework to build ones model with a lot of functions. This provides modularity to our model development and as such a general description of our functions is as follows:

| Function                     | Description |
|------------------------------|-------------|
| `compute_forward()`          | Performs model forward pass for both training and inference |
| `compute_objectives()`       | Calculates loss between predictions and targets |
| `fit_batch()`                | Handles one training batch (forward pass + backpropagation) |
| `evaluate_batch()`           | Processes validation/test batches without gradient computation |
| `on_stage_end()`             | Performs end-of-epoch operations (logging, checkpointing, LR scheduling) |
| `save_results()`             | Computes and saves separation metrics (SDR, SI-SNR) |
| `save_audio()`               | Saves sample audio outputs for qualitative evaluation |



## Generating Results

Ohh the sweet and savory.... Finally we get to discussing our results for the entire set of models. First things first, the below snippet of code generates a sweet little table for us to see how our models perform and improve across training and training instances. 

Before that.... context sil vous plait ;)

What exactly do these four terms mean...

### SiSNR  (Scale-Invariant Signal-to-Noise Ratio)

SI-SNR measures the quality of source separation by comparing the reconstructed signal to the original target. Higher values (closer to +inf dB) mean better separation, while low/negative values indicate poor reconstruction (noise dominates). Ideal values depend on the task, but >10 dB is typically good, while <0 dB suggests failure.


### SiSNR_i  (Scale-Invariant Signal-to-Noise Ratio - Improvement)
SI-SNRi compares the model’s output to the original mixture. Positive values mean the model improved separation, while negative values mean it performed worse than the input mix. A value of +3 dB or higher is desirable, while negative SI-SNRi indicates the model is degrading the signal.

### SDR (Source-to-Distortion Ratio)
SDR evaluates overall signal fidelity, considering distortions and artifacts. Higher SDR (>0 dB) means cleaner separation, while negative values indicate severe distortion. Values below -10 dB suggest unusable output.

### SDR_i (SDR Improvement)
SDRi measures how much better (or worse) the model is compared to the original mixture. Positive SDRi means improvement, while negative values imply the output is worse than the mix. A well-trained model should have SDRi > 0 dB, with higher values (e.g., +5 dB+) indicating strong performance.


Ok so with that in mind...(drumroll)... scroll down to see the verdict



