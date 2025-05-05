import torch
import numpy as np
import jiwer
from fine_tuning_script import load_and_preprocess_data, initialize_model_and_processors, WhisperFineTuner,setup_trainer,generate_predictions,clean_text
from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
import re 

# Configuration
SAVE_DIR = "./SAVE_DIR"
MODEL_NAME = "openai/whisper-medium"
LANGUAGE = "Basque"
LEARNING_RATE = 1e-4
BATCH_SIZE = 24
MAX_EPOCHS = 30
ACCUMULATE_GRAD_BATCHES = 24

N_GPUS = torch.cuda.device_count()

print(f"N_gpu: {N_GPUS}")
print(f"Gradient Accumulation: {ACCUMULATE_GRAD_BATCHES}")
print(f"BATCH_SIZE: {BATCH_SIZE}")




def main():

    # Initialize feature extractor, tokenizer, and processor
    whisper_model,feature_extractor, tokenizer, processor = initialize_model_and_processors(MODEL_NAME, LANGUAGE)

    # load and preprocess data
    raw_dataset, processed_dataset = load_and_preprocess_data(
    "mozilla-foundation/common_voice_17_0", 
    language='eu',
    use_huggingface=True, 
    token="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",
    feature_extractor=feature_extractor, 
    tokenizer=tokenizer
	)


    # Initialize model
    basque_model = WhisperFineTuner(whisper_model,processor,feature_extractor, tokenizer,LEARNING_RATE,BATCH_SIZE,processed_dataset)

    # Setup Trainer
    trainer = setup_trainer(
        save_dir=SAVE_DIR,
        max_epochs=MAX_EPOCHS,
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
        n_gpus=N_GPUS
    )

    # Start training
    trainer.fit(basque_model)
    output_model = SAVE_DIR+"/model.pt"

    # Lightning Deepspeed has saved a directory instead of a file
    convert_zero_checkpoint_to_fp32_state_dict(SAVE_DIR+"/best_model.ckpt", output_model)
    print(f"[INFO] Training completed. Best model saved at {SAVE_DIR}")

     
if __name__ == "__main__":
    main()


