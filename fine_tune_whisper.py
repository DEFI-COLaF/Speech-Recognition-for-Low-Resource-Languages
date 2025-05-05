import torch
import evaluate
import re
import os
from transformers import (WhisperProcessor, WhisperForConditionalGeneration,
                          WhisperFeatureExtractor, WhisperTokenizer)

from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Any, List, Dict, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Timer,ModelCheckpoint,EarlyStopping

from pytorch_lightning import Trainer


from datasets import load_dataset,Dataset,DatasetDict,Audio

from torch.utils.data import IterableDataset
from tqdm import tqdm 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. DATA PREPARATION
def clean_text(text: str) -> str:
    """ Clean text: remove punctuation and convert to lowercase. """
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\s+", " ", text)     # Normalize whitespace
    return text.strip().lower()

class WhisperStreamingDataset(IterableDataset):
    """
    A streaming dataset class for processing large Hugging Face audio datasets without loading them entirely into memory.

    This class reads samples one by one, applies audio feature extraction and tokenizes transcriptions on-the-fly.
    Ideal for memory-efficient training and preprocessing of large-scale speech datasets.
    """
    def __init__(self, hf_dataset, feature_extractor, tokenizer):
        self.dataset = hf_dataset
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    def __iter__(self):
        for sample in self.dataset:
            transcription = clean_text(sample["sentence"])
            input_features = self.feature_extractor(
                sample["audio"]["array"], sampling_rate=16000
            ).input_features[0]
            labels = self.tokenizer(transcription).input_ids
            yield {
                "input_features": input_features,
                "labels": labels
            }


def load_and_preprocess_data(path,language=None, use_huggingface=False,token=None ,feature_extractor=None, tokenizer=None):
    """
    Loads and preprocesses data by extracting audio features and tokenizing transcriptions,
    from the specified source (Hugging Face or local directory).
    
    Parameters:
    -----------
    - path (str): The path to the dataset. If `use_huggingface=True`, this should be the name of a dataset 
                  available on Hugging Face (e.g., "mozilla-foundation/common_voice_17_0"). If `use_huggingface=False`, 
                  this should be a path to a local directory containing audio files (.wav) and transcription files (.txt).
    - use_huggingface (bool): If True, loads the dataset from Hugging Face. If False, loads the files from a 
                              local directory.
    - language (str, optional): The language code for the dataset. This parameter is required if `use_huggingface=True`
                                and specifies the language of the dataset (e.g., "eu" for Basque).
    - token (str, optional): Hugging Face authentication token (required for private datasets).

    - feature_extractor (Wav2Vec2FeatureExtractor or other): The feature extractor used to transform audio data into 
                                                            numerical representations.
    - tokenizer (PreTrainedTokenizer): The tokenizer used to transform transcriptions into token IDs (input_ids).
    
    Returns:
    --------
    - dataset (DatasetDict): A dictionary of datasets containing preprocessed training and validation data, 
                              with extracted audio features and tokenized transcriptions.
                              
    Detailed Description:
    ----------------------
    This function loads and preprocesses data from two possible sources:
    1. **Hugging Face**: If `use_huggingface=True`, it loads the dataset from the Hugging Face platform (e.g., "mozilla-foundation/common_voice_17_0").
       The `language` parameter specifies the language of the dataset (e.g., "eu" for Basque). The audio data is cast to a 16 kHz sampling rate, 
       and the transcriptions are cleaned (using the `clean_text` function). The function then uses an audio feature extractor to convert 
       each audio file into features, and a tokenizer to convert each transcription into a sequence of token IDs.
       
    2. **Local Directory**: If `use_huggingface=False`, the function assumes that `path` points to a local directory containing 
       audio files (with `.wav` extension) and transcription files (with `.txt` extension). Each audio file is paired with a corresponding 
       transcription file. The audio data is cast to a 16 kHz sampling rate, and the transcriptions are cleaned and tokenized.
       
    The function returns a `DatasetDict` containing preprocessed datasets for training and validation, 
    where each sample is a dictionary with audio features and tokenized transcription labels.
    """
    
    if use_huggingface:
        # Load datasets from Hugging Face
        train_dataset = load_dataset(path, language, split="train", streaming=True,token=token)
        dev_dataset = load_dataset(path, language, split="validation", streaming=True,token=token)
        test_dataset = load_dataset(path, language, split="test", streaming=True,token=token)

        # Cast the 'audio' column to 16kHz
        train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
        dev_dataset = dev_dataset.cast_column("audio", Audio(sampling_rate=16000))
        test_dataset = dev_dataset.cast_column("audio", Audio(sampling_rate=16000))
        
        dataset = DatasetDict({
            "train": train_dataset,
            "dev": dev_dataset,
            "test":test_dataset
        })
        train_dataset = WhisperStreamingDataset(train_dataset, feature_extractor, tokenizer)
        dev_dataset = WhisperStreamingDataset(dev_dataset, feature_extractor, tokenizer)
        processed_dataset = DatasetDict({
            "train": train_dataset,
            "dev": dev_dataset
        })


    else:
        # Loading data from a local directory
        def audio_text_generator(data_dir):
            """
            Generates a dictionary for each text file (.txt) by finding a matching audio file
            (.wav or .mp3) with the same base name.

            This function is useful for pairing audio and transcription data for speech processing tasks.
            """
            for file_name in os.listdir(data_dir):
                if file_name.endswith(".txt"):
                    base_name = os.path.splitext(file_name)[0]
                    txt_file_path = os.path.join(data_dir, file_name)

                    # Chercher le fichier audio correspondant (.wav ou .mp3)
                    audio_file_path = None
                    for ext in [".wav", ".mp3"]:
                        candidate_path = os.path.join(data_dir, base_name + ext)
                        if os.path.exists(candidate_path):
                            audio_file_path = candidate_path
                            break

                    if audio_file_path:
                        with open(txt_file_path, "r", encoding="utf-8") as f:
                            transcription = f.read().strip()

                        yield {
                            "audio": audio_file_path,
                            "sentence": transcription
                        }

        def prepare_dataset(batch, feature_extractor, tokenizer):
            """ Prepare the data from local files. """
            audio_batch = batch["audio"]
            input_features = []
            for audio in audio_batch:
                features = feature_extractor(audio['array'], sampling_rate=16000)
                input_features.append(features.input_features[0])

            batch["input_features"] = input_features
            batch["labels"] = tokenizer(batch["sentence"]).input_ids
            return batch

        # Check if the provided path is a valid directory
        if not os.path.isdir(path):
            raise ValueError(f"The path '{path}' is not a valid directory.")

        # Load local data
        dataset = Dataset.from_generator(lambda: audio_text_generator(path))
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        
        # Split the dataset into train, dev and test
        dataset_split = dataset.train_test_split(test_size=0.2, shuffle=True, seed=2)
        dev_test_split = dataset_split["test"].train_test_split(test_size=0.5, shuffle=True, seed=2)
        dataset =DatasetDict( {
                "train": dataset_split["train"],  # 80%
                 "dev": dev_test_split["train"],  # 10% 
                "test": dev_test_split["test"]    # 10% 
                })

        
        # Apply the transformation with the prepare_dataset function
        processed_dataset = dataset.map(lambda batch: prepare_dataset(batch, feature_extractor, tokenizer),
                                           batched=True,
                                           remove_columns=dataset["train"].column_names)


    return dataset, processed_dataset 


# 3. MODEL PREPARATION
def initialize_model_and_processors(model_name, language):
    """Initialize feature extractor, tokenizer, and model processor."""
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.generation_config.task = "transcribe"
    model.generation_config.language=language
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name, task="transcribe", language=language)
    tokenizer = WhisperTokenizer.from_pretrained(model_name, task="transcribe", language=language)
    processor = WhisperProcessor.from_pretrained(model_name, task="transcribe", language=language)
    return model,feature_extractor, tokenizer, processor




# 3. COLLATOR FOR BATCHING
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """ Custom collator for batching audio and transcription data. """
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch




# 4. LightningModule: fine-tuning logic for Whisper
class WhisperFineTuner(pl.LightningModule):
    """
    PyTorch Lightning module to fine-tune OpenAI’s Whisper model for speech-to-text transcription.

    - Loads a pretrained Whisper model and adapts its generation config for the target language.
    - Implements training and validation steps, logging both loss and Word Error Rate (WER).
    - Uses a custom data collator to pad variable-length audio feature inputs and label sequences.
    - Provides a generate() helper that wraps beam‐search decoding with no‐repeat-ngram and nucleus sampling.
    - Configures the AdamW optimizer and supports gradient accumulation, mixed-precision, and DeepSpeed stage 3.
    """
    def __init__(self, model, processor,feature_extractor=None, tokenizer=None, lr=None,batch_size=None,dataset=None):
        super().__init__()
        self.model = model
        self.lr = lr
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.processor =processor
        self.data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)
        self.dataset= dataset
        self.batch_size = batch_size
        
        
    def forward(self, input_features, labels):
        """
        Forward pass: computes model outputs including loss.
        """
        return self.model(input_features=input_features, labels=labels)

    
    def training_step(self, batch):
        """
        Training step: calculate loss and log training metrics.
        """
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    
    def validation_step(self, batch):
        """
        Validation step: compute loss and WER for model performance evaluation.
        """
        outputs = self.model(**batch)
        val_loss = outputs.loss
        self.log("val_loss", val_loss, prog_bar=True,sync_dist=True)

        # Compute WER
        pred_ids = outputs.logits.argmax(dim=-1)  # Get predicted tokens (logits -> predicted ids)
        label_ids = batch["labels"]

        label_ids[label_ids == -100] = self.tokenizer.pad_token_id
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        metric = evaluate.load("wer")
        wer = metric.compute(predictions=pred_str, references=label_str)
        self.log("val_wer", wer, prog_bar=True, sync_dist=True)  # Log the WER metric after each validation step
        return val_loss
    
    def train_dataloader(self):
        print("[INFO] Preparing train_DataLoader...")
        return DataLoader(self.dataset["train"], batch_size=self.batch_size, collate_fn=self.data_collator,num_workers=7)

    def val_dataloader(self):
        print("[INFO] Preparing val_dataLoader...")
        return DataLoader(self.dataset["dev"], batch_size=self.batch_size, collate_fn=self.data_collator,num_workers=7)


    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
        
    def generate(self, input_features, **kwargs):
        return self.model.generate(input_features,
                                    num_beams=5,
                                    no_repeat_ngram_size=2,  
                                    **kwargs)
# 6. Trainer
def setup_trainer(save_dir, max_epochs, accumulate_grad_batches, n_gpus):
    """Setup the Trainer and its callbacks."""
    timer = Timer()
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", 
        dirpath=save_dir,    
        filename="best_model", 
        save_top_k=1,       
        mode="min",         
        verbose=True
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",  
        patience=3, 
        mode="min"
    )
    trainer = Trainer(
        max_epochs=max_epochs,
        devices=n_gpus if n_gpus > 1 else 1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy="deepspeed_stage_3",
        log_every_n_steps=50,
        callbacks=[timer,checkpoint_callback,early_stop_callback], 
        default_root_dir=save_dir,
        accumulate_grad_batches=accumulate_grad_batches
    )
    return trainer


def generate_predictions(dataset, model):
    model.to(device)
    predictions = []
    references = []
    for sample in tqdm(dataset, desc="Processing samples", unit="sample", dynamic_ncols=True):
        input_speech = sample["audio"]
        reference_text = sample["sentence"]

        # Preprocess audio
        input_features = model.processor(input_speech["array"], sampling_rate=input_speech["sampling_rate"], return_tensors="pt").input_features.to(device)

        # Generate token ids
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        
        # Decode token ids into text
        predicted_text = model.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        # Append prediction and reference to their respective lists
        predictions.append(predicted_text)
        references.append(reference_text)

    return predictions, references
    

