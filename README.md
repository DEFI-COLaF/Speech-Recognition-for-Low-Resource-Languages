# Speech-Recognition-for-Low-Resource-Languages

This repository contains code to fine-tune Whisper models on custom datasets. It includes:
* `fine_tune_whisper.py`: The core script for fine-tuning, data preprocessing, training, and saving models.
* `fine_tuning_example.py`: An example script demonstrating how to use `fine_tune_whisper.py` for fine-tuning.
* `basque_fine_tuning.py`: An example script for fine-tuning Whisper on a Basque dataset and evaluating the model performance.

### 📦 Requirements
Install dependencies with:  
`pip install -r requirements.txt`

### ⚙️ Configuration

Update the following parameters at the top of `fine_tuning_example.py` to configure data paths, model selection, and training settings:

```python

# Save directory for checkpoints
SAVE_DIR = "./SAVE_DIR"

# Whisper model variant (options: "openai/whisper-base", "openai/whisper-medium", "openai/whisper-large", "openai/whisper-large-v3")
MODEL_NAME = "openai/whisper-medium"

# Target language (must match dataset and be supported by Whisper)
LANGUAGE = "Basque"

# Training parameters
BATCH_SIZE = 24                # batch size per GPU
MAX_EPOCHS = 30                 # total training epochs
ACCUMULATE_GRAD_BATCHES = 24    # gradient accumulation steps
LEARNING_RATE = 1e-4            # optimizer learning rate
```

Whisper supports a fixed set of languages. You can find the full list of supported languages [here](https://platform.openai.com/docs/guides/speech-to-text/supported-languages/#supported-languages).
If your language is not included, select the closest supported one. Otherwise, you may omit the `LANGUAGE` parameter to enable automatic detection.

Adjust `BATCH_SIZE` and `ACCUMULATE_GRAD_BATCHES` depending on your available GPU memory. Choose a `BATCH_SIZE` that balances memory usage with training stability. If you encounter memory limitations, reduce the batch size and use `ACCUMULATE_GRAD_BATCHES` to accumulate gradients over multiple steps, allowing you to simulate a larger effective batch size while avoiding memory overflow.

For large models (whisper-large, whisper-large-v3), it is recommended to set `LEARNING_RATE = 1e-5` to allow for more gradual updates, preventing abrupt changes that could disrupt pre-existing knowledge and ensuring a thorough exploration of the parameter space without skipping optimal solutions. 


### 🧾 Dataset Format Options
You can load your dataset using one of the following two methods:

#### 📚 Option 1: Hugging Face Dataset (Common Voice)
You can directly use [Common Voice datasets](https://huggingface.co/datasets/mozilla-foundation) available on Hugging Face. This method allows you to load and preprocess the data easily.

Example usage:

```python
 # load and preprocess data
    raw_dataset, processed_dataset= load_and_preprocess_data(
    "mozilla-foundation/common_voice_17_0",  # Dataset name on Hugging Face
    language="eu",                           # ISO 639-1 language code
    use_huggingface=True, 
    token="hf_token",                        # Your Hugging Face access token
    feature_extractor=feature_extractor, 
    tokenizer=tokenizer
	)
```

#### 📁 Option 2: Local Directory
If you prefer to use a custom dataset from a local directory, organize your files so that each audio file (.wav or .mp3) has a corresponding .txt transcription file with the same name.
Directory structure example:

```
your_dataset/
├── sample_001.wav  
├── sample_001.txt  
├── sample_002.wav  
├── sample_002.txt  
└── ... 
```
To load and preprocess the data:

```python
raw_dataset, processed_dataset = load_and_preprocess_data(path="data/")
```

##### Audio Requirements
* Maximum duration: 30 seconds per clip
* Sampling rate: 16 kHz (audio will be automatically resampled if necessary)

To comply with Whisper's 30-second audio limit, make sure your audio files are segmented and aligned with the transcriptions. This alignment can be achieved using alignment tools such as [Wav2Vec2 Forced Alignment](https://pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html?utm_source=chatgpt.com) or [Montreal Forced Aligner (MFA)](https://mfa-models.readthedocs.io/en/latest/).

##### Transcript Normalization
During preprocessing, all transcriptions are automatically:

* Fully lowercased
* Stripped of all punctuation

This normalization ensures the model focuses solely on phonetic and linguistic content, without being distracted by case or punctuation.

### 🚀 Training
Once parameters and data are configured, launch training with:
```python
python fine_tuning_example.py
```
✅ The best-performing model will be automatically saved as `model.pt` in the specified `SAVE_DIR`.


### 📈 Fine-tuning Results:
We have focused on a selection of low-resource languages spoken in France, including:

* Basque
* Alsatian
* Shimaore

The table below summarizes the results obtained after fine-tuning Whisper models on these languages. Key metrics include the size of the training data, duration of fine-tuning, and transcription performance:

| Language | Base Model       | Training Data (hrs) | Training Duration | WER (%) | CER (%) |
|----------|------------------|---------------------|-------------------|---------|---------|
| Basque   | whisper-medium   | 116h20              | 13h08             |  8.7    |   1.6   |
| Alsatian | whisper-large-v3 | 9h36                | 18min             | 47.2    |  26.1   |
| Shimaore | whisper-large-v3 | 1h28                | 5min              | 69.2    |  29.1   |

Experiments were conducted using two Nvidia Tesla L40S GPUs, each with 45 GiB of memory. 

For Basque, the model was fine-tuned using the [CommonVoice Basque dataset](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0), while private datasets were used for the other languages

To reproduce the Basque results, run:
```python
python basque_fine_tuning.py
```
This script fine-tunes the Whisper model on the Basque dataset and displays the model's performance metrics (WER and CER) upon completion.


### 🛠️ Work in Progress
We are actively working on expanding support to additional regional and Low-Resource-Languages. 
