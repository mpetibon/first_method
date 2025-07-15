# 1 Encoder decoder Notebook

This notebook provides a complete, step-by-step pipeline for training, evaluating, and testing a convolutional autoencoder for audio super-resolution. All code is contained in a single, well-annotated notebook, making it easy to follow and adapt to your own data.

## Features

- **Data Pairing:** Functions to pair degraded and clean audio files for supervised learning.
- **Custom Dataset:** PyTorch Dataset and DataLoader for efficient spectrogram loading and batching.
- **Configurable Training:** Train the autoencoder with customizable hyperparameters and save the model to any path.
- **Evaluation & Output Generation:** Evaluate the model on test data and automatically generate and save example outputs for inspection.
- **Modular Functions:** All major steps (data preparation, training, evaluation, output generation) are implemented as reusable, well-documented Python functions.


## Main Imports

The notebook uses the following Python libraries:
- `torch`
- `torch.nn`
- `torchaudio`
- `torch.nn.functional`
- `os`
- `tqdm` (including `tqdm.notebook`)
- `numpy`
- `shutil`
- `torch.utils.data`

## Requirements

Install the required packages with:
```bash
pip install torch torchaudio tqdm numpy
```

## Usage

1. **Prepare Data Pairs:**  
   Use the provided function to pair degraded and clean audio files.

2. **Create Dataset and DataLoader:**  
   Instantiate the dataset and dataloader for training and testing.

3. **Train the Model:**  
   Call the training function, specifying hyperparameters and the model save path.

4. **Evaluate and Save Examples:**  
   Use the evaluation function to compute test loss and save example outputs for listening or analysis.

## Notes

- All steps are contained in a single notebook for clarity and reproducibility.
- Paths, hyperparameters, and output directories are easily configurable.
- Example outputs are automatically saved for inspection.

-----------------------------------------------------------

# 2 test_trained_model Notebook
This notebook is designed to **test a trained audio autoencoder model** on a set of audio files.  
You provide the path to your trained model (`.pth` file) and the directory containing your audio data.  
The notebook will process the data, run the model, and save the reconstructed audio outputs for listening or further analysis.

## What does this notebook do?

- Loads a trained PyTorch autoencoder model from a `.pth` file.
- Loads and preprocesses audio files from a user-specified directory.
- Runs the model on each audio file to generate reconstructed outputs.
- Saves each reconstructed audio file with a filename that includes the original name.

## How to use

1. **Set the model path:**  
   Change the `model_path` variable in the notebook to the path of your trained model file (e.g. `trained_autoencoder_simple.pth`).

2. **Set the data directory:**  
   Change the `audio_dir` variable to the folder containing your audio files (e.g. `MusicGen_data`).

3. **Run the notebook:**  
   The notebook will:
   - Load your model and its weights
   - Preprocess each audio file
   - Run the model to generate reconstructed outputs
   - Save each reconstructed audio file as `reconstructed_<originalfilename>.wav` in the current directory

## Requirements

- Python 3.7+
- torch
- torchaudio
- tqdm

Install dependencies with:
```bash
pip install torch torchaudio tqdm
```

## Output

- For each input audio file, a reconstructed `.wav` file will be saved in the current directory, prefixed with `reconstructed_`.
  - Example:  
    - Input: `A_flamenco_guitar_improvisation_with_pal.wav`  
    - Output: `reconstructed_A_flamenco_guitar_improvisation_with_pal.wav`

## Notes
- The model class definition in the notebook must match the architecture used during training.
- This notebook is for inference only; it does not perform training or fine-tuning.

-------------------------------------------------
# 3 Trained model
The weights of the trained model are available here:
https://github.com/mpetibon/first_method/releases/download/trained_model/trained_encoder_decoder_simple.pth
