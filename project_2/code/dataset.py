import os
import pickle
import zipfile
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset # Hugging Face datasets library
from transformers import AutoTokenizer, DataCollatorWithPadding
# Need kaggle api if you want to use the download function
# from kaggle.api.kaggle_api_extended import KaggleApi

# --- Custom Dataset for the Competition Test File ---
class AGNewsTestDataset(Dataset):
    """
    Custom dataset for AGNEWS competition test text data.

    Args:
        pkl_file (str): Path to the pickle file containing the test data.
        tokenizer (callable): Tokenizer instance (e.g., from Hugging Face)
        max_length (int): Maximum sequence length for tokenization.
    """
    def __init__(self, pkl_file, tokenizer, max_length=512):
        try:
            with open(pkl_file, 'rb') as f:
                # Load data - *ASSUMPTION*: pkl contains a list of text strings
                # If it's a dictionary (e.g., {'text': [...]}), adjust access: data_dict['text']
                self.texts = pickle.load(f)
                if isinstance(self.texts, dict):
                    # Common pattern: dictionary with a key like 'text' or 'data'
                    possible_keys = ['text', 'data', 'description'] # Add other likely keys if needed
                    data_key = next((k for k in possible_keys if k in self.texts), None)
                    if data_key:
                         print(f"Assuming text data is under key '{data_key}' in the pkl file.")
                         self.texts = self.texts[data_key]
                    else:
                        raise ValueError(f"Could not find text data in pkl dictionary. Keys found: {list(self.texts.keys())}")

                if not isinstance(self.texts, list):
                     raise TypeError(f"Expected pkl file to contain a list of texts (or a dict with a text list), but got {type(self.texts)}")

        except FileNotFoundError:
            print(f"Error: Test pickle file not found at {pkl_file}")
            raise
        except Exception as e:
            print(f"Error loading or processing pickle file {pkl_file}: {e}")
            raise

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        # Ensure text is a string
        if not isinstance(text, str):
            text = str(text)

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=False, # Padding will be handled by the collator
            max_length=self.max_length,
            return_tensors=None, # Return python lists/ints, collator handles tensor conversion
        )

        # Return the tokenized inputs and the original index for submission mapping
        # Remove 'token_type_ids' if your model doesn't use them (like RoBERTa)
        item = {k: v for k, v in encoding.items() if k != 'token_type_ids'}
        item['index'] = index # Include original index

        return item

# --- Data Module for AGNEWS ---
class AGNewsDataModule:
    """
    Data module for AGNEWS dataset (train/val from Hugging Face, test from competition file).

    Args:
        model_name_or_path (str): Identifier for the tokenizer (e.g., "roberta-base").
        data_dir (str): Directory to potentially store data (less critical when using `datasets`).
        competition_name (str): Name of the Kaggle competition for downloading test data.
        batch_size (int): Training batch size.
        test_batch_size (int): Testing/Validation batch size.
        num_workers (int): Number of workers for data loading.
        max_seq_length (int): Maximum sequence length for tokenizer.
        val_split_percentage (float): Percentage of training data to use for validation (0 to disable).
    """
    def __init__(self,
                 model_name_or_path="roberta-base",
                 data_dir="./data_agnews",
                 competition_name="deep-learning-spring-2025-project-2", # UPDATE IF NEEDED
                 batch_size=16,
                 test_batch_size=32,
                 num_workers=2,
                 max_seq_length=512,
                 val_split_percentage=0.1): # Use 10% of train for validation

        self.model_name_or_path = model_name_or_path
        self.data_dir = data_dir
        self.competition_name = competition_name
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.max_seq_length = max_seq_length
        self.val_split_percentage = val_split_percentage

        # Paths for competition data
        self.competition_path = os.path.join(self.data_dir, self.competition_name)
        self.zip_path = os.path.join(self.competition_path, f"{self.competition_name}.zip")
        self.test_pkl = os.path.join(self.competition_path, "test_unlabelled.pkl") # Correct filename

        # Initialize tokenizer and data collator
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        # Data collator handles dynamic padding within each batch
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        self.train_dataset = None
        self.val_dataset = None
        self.predict_dataset = None

    def _tokenize_function(self, examples):
        # Tokenize the text field. AGNEWS uses 'text'.
        # Padding is false here; collator handles it later.
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=self.max_seq_length
        )

    def prepare_data(self):
        """Downloads competition data if needed."""
        # Download standard AGNEWS train/test via `datasets` library automatically on first use.
        print("Checking/downloading AGNEWS dataset from Hugging Face...")
        load_dataset("ag_news", cache_dir=os.path.join(self.data_dir, "hf_cache"))
        print("Checking/downloading competition test data...")
        self.download_competition_data()

    def setup(self, stage=None):
        """Loads and preprocesses datasets."""
        # Load AGNEWS dataset
        dataset = load_dataset("ag_news", cache_dir=os.path.join(self.data_dir, "hf_cache"))

        # Tokenize dataset
        tokenized_dataset = dataset.map(self._tokenize_function, batched=True)

        # Remove original text column, select necessary columns
        tokenized_dataset = tokenized_dataset.remove_columns(["text"])
        # Rename 'label' to 'labels' if required by the model/trainer framework
        # tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
        tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

        if stage == "fit" or stage is None:
            ag_train_data = tokenized_dataset["train"]
            if self.val_split_percentage > 0:
                split = ag_train_data.train_test_split(test_size=self.val_split_percentage)
                self.train_dataset = split['train']
                self.val_dataset = split['test']
                print(f"Using {len(self.train_dataset)} samples for training, {len(self.val_dataset)} for validation.")
            else:
                # Use standard AGNEWS test set as validation if no split % is given
                self.train_dataset = ag_train_data
                self.val_dataset = tokenized_dataset["test"]
                print(f"Using {len(self.train_dataset)} samples for training, {len(self.val_dataset)} (standard test set) for validation.")


        if stage == "validate" or stage is None:
             if self.val_dataset is None: # If setup wasn't called with 'fit'
                 # Load validation data (standard AGNEWS test set)
                 self.val_dataset = tokenized_dataset["test"]
                 print(f"Loaded {len(self.val_dataset)} (standard test set) for validation.")


        if stage == "test" or stage is None:
            # Setup competition test dataset
             print(f"Setting up competition test dataset from: {self.test_pkl}")
             self.predict_dataset = AGNewsTestDataset(
                 self.test_pkl,
                 self.tokenizer,
                 self.max_seq_length
             )
             print(f"Loaded {len(self.predict_dataset)} samples for competition prediction.")


    def get_train_loader(self):
        if not self.train_dataset:
            self.setup("fit")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.data_collator # Use collator for dynamic padding
        )

    def get_val_loader(self):
        if not self.val_dataset:
            self.setup("validate") # Or 'fit' if you always run setup completely
        return DataLoader(
            self.val_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.data_collator # Use collator for dynamic padding
        )

    def get_competition_test_loader(self):
        """Gets the DataLoader for the competition's unlabelled test set."""
        if not self.predict_dataset:
            self.setup("test")
        return DataLoader(
            self.predict_dataset,
            batch_size=self.test_batch_size,
            shuffle=False, # Important: Keep order for submission
            num_workers=self.num_workers,
            collate_fn=self.data_collator # Use collator for dynamic padding - it handles dicts well
        )

    def download_competition_data(self):
        """Downloads and extracts competition test data using Kaggle API."""
        if not os.path.exists(self.test_pkl):
            print(f"Competition test file not found at {self.test_pkl}. Attempting download...")
            os.makedirs(self.competition_path, exist_ok=True)
            try:
                from kaggle.api.kaggle_api_extended import KaggleApi
                api = KaggleApi()
                api.authenticate() # Make sure kaggle.json is set up
                api.competition_download_files(self.competition_name, path=self.competition_path)

                if os.path.exists(self.zip_path):
                    print(f"Extracting {self.zip_path}...")
                    with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                        zip_ref.extractall(self.competition_path)
                    os.remove(self.zip_path) # Clean up the zip file
                    print("Extraction complete.")
                else:
                     print(f"Warning: Zip file {self.zip_path} not found after download attempt.")

            except ImportError:
                print("Warning: 'kaggle' library not found. Cannot download competition data automatically.")
                print("Please download the 'test_unlabelled.pkl' manually from the Kaggle competition page")
                print(f"and place it in: {self.competition_path}")
            except Exception as e:
                print(f"An error occurred during Kaggle download/extraction: {e}")
                print("Please check your Kaggle API setup and competition name.")

        if not os.path.exists(self.test_pkl):
            # Raise error only after attempting download
            raise FileNotFoundError(
                f"Competition test file '{os.path.basename(self.test_pkl)}' not found in '{self.competition_path}'. "
                "Please ensure it is downloaded and extracted correctly."
            )
        else:
            print(f"Competition test file found: {self.test_pkl}")


# --- Example Usage ---
if __name__ == '__main__':
    # Example configuration
    MODEL_ID = "roberta-base"
    COMPETITION_ID = "deep-learning-spring-2025-project-2" # Double-check this ID
    DATA_DIR = "./agnews_data"
    BATCH_SIZE = 8 # Small batch size for demo
    TEST_BATCH_SIZE = 16
    MAX_LEN = 128 # Shorter length for faster demo processing

    # Instantiate the data module
    data_module = AGNewsDataModule(
        model_name_or_path=MODEL_ID,
        data_dir=DATA_DIR,
        competition_name=COMPETITION_ID,
        batch_size=BATCH_SIZE,
        test_batch_size=TEST_BATCH_SIZE,
        max_seq_length=MAX_LEN,
        num_workers=0 # Set to 0 for easier debugging in __main__
    )

    # --- Prepare Data (Download) ---
    # Normally you might call this separately or rely on a framework like PyTorch Lightning
    # data_module.prepare_data() # Downloads HF data and competition data if needed

    # --- Setup Data (Load and Process) ---
    # Call setup for the stages you need. Calling with None sets up all.
    # data_module.setup() # Sets up train, val, and test

    # --- Get DataLoaders ---
    print("\n--- Getting Train Loader ---")
    try:
        train_loader = data_module.get_train_loader()
        print(f"Train loader created. Number of batches: {len(train_loader)}")
        # Inspect a batch
        batch = next(iter(train_loader))
        print("Sample batch keys:", batch.keys())
        print("Input IDs shape:", batch['input_ids'].shape)
        print("Labels shape:", batch['labels'].shape)
        # Decode an example (Optional)
        print("Sample decoded text:", data_module.tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True))
        print("Sample label:", batch['labels'][0].item())

    except Exception as e:
        print(f"Error creating/iterating train loader: {e}")


    print("\n--- Getting Validation Loader ---")
    try:
        val_loader = data_module.get_val_loader()
        print(f"Validation loader created. Number of batches: {len(val_loader)}")
        batch = next(iter(val_loader))
        print("Sample batch keys:", batch.keys())
        print("Input IDs shape:", batch['input_ids'].shape)
        print("Labels shape:", batch['labels'].shape)
    except Exception as e:
        print(f"Error creating/iterating validation loader: {e}")


    print("\n--- Getting Competition Test Loader ---")
    try:
        comp_test_loader = data_module.get_competition_test_loader()
        print(f"Competition test loader created. Number of batches: {len(comp_test_loader)}")
        batch = next(iter(comp_test_loader))
        print("Sample batch keys:", batch.keys()) # Should include 'input_ids', 'attention_mask', 'index'
        print("Input IDs shape:", batch['input_ids'].shape)
        print("Index shape:", batch['index'].shape)
         # Decode an example (Optional)
        print("Sample decoded test text:", data_module.tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True))
        print("Sample original index:", batch['index'][0].item())

    except FileNotFoundError as e:
         print(f"\nError getting competition test loader: {e}")
         print("Please ensure 'test_unlabelled.pkl' is downloaded to the correct location.")
    except Exception as e:
         print(f"Error creating/iterating competition test loader: {e}")