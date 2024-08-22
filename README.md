<h1>Budgerigar Spectrogram Generation</h1>
Budgerigars display an astounding range of vocal plasticity. To capture this natural vocalization behavior, our lab opted for using piezoelectric microphones for audio analysis. The benefit of doing so over traditional (ambient) microphones is that audio quality is almost perfectly preserved even in noisy environments.
<br/><br/>
The main drawback to relying on piezoelectric microphone data is its high signal to noise ratio, as it is very sensitive to any disturbances or vibrations (wing flapping, feeding, etc.). Bird audio recordings can range from hours to over a 24-hour period, with periods of singing representing a very small portion of the total recording duration. Thus, it is an arduious task to segment and label such audio files by hand.

<br/>
I have implemented a neural-network-based binary classifier to differentiate vocalizations from noise. This package will generate mel-scale spectrograms from an audio segment and its corresponding label file (event detections). These generated images are then used as the input to the neural network.


<div>
    <h3>Vocalization Examples:</h3>
    <img src="figures/voc1.jpg">
    <img src="figures/voc2.jpg">
    <img src="figures/voc3.jpg">
</div>

<div>
    <h3>Noise Examples:</h3>
    <img src="figures/noise1.jpg">
    <img src="figures/noise2.jpg">
    <img src="figures/noise3.jpg">
</div>


<h2>Usage</h2>
Change the audio and label paths to match that of your audio files in spec_options.toml, and then run the following command in the terminal. 

```console
python src/main.py spec_options.toml
```