from . import Generator
from .tabu.tabula import Tab
from .tabump.tabula import TabMP
from synthtab.utils import console

import os
import pandas as pd
import torch
import typing as tp
from huggingface_hub import hf_hub_download

REPO_ID = "omaralvarez/tabula"
FILENAME = "model.pt"


class Tabula(Generator):
    """Tabula Class

    The Tabula class handles the whole generation flow. It is used to fine-tune a large language model for tabular data,
    and to sample synthetic tabular data.

    Attributes:
        llm (str): HuggingFace checkpoint of a pretrained large language model, used a basis of our model
        tokenizer (AutoTokenizer): Tokenizer, automatically downloaded from llm-checkpoint
        model (AutoModelForCausalLM): Large language model, automatically downloaded from llm-checkpoint
        experiment_dir (str): Directory, where the training checkpoints will be saved
        epochs (int): Number of epochs to fine-tune the model
        batch_size (int): Batch size used for fine-tuning
        train_hyperparameters (dict): Additional hyperparameters added to the TrainingArguments used by the
         HuggingFaceLibrary, see here the full list of all possible values
         https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments
        columns (list): List of all features/columns of the tabular dataset
        num_cols (list): List of all numerical features/columns of the tabular dataset
        categorical_columns (dict | list): only for int that are categories? did not work with conditional_col
        maybe only for other categorical labels
    """

    def __init__(
        self,
        dataset,
        llm: str = "distilgpt2",
        experiment_dir: str = "trainer_tabula",
        epochs: int = 100,
        batch_size: int = 8,
        max_tries_per_batch: int = 512,
        resume_from_checkpoint: tp.Union[bool, str] = False,
        encode_categories: bool = False,  # Reduce token length using int categories
        # Generation options
        start_col: tp.Optional[str] = "",
        start_col_dist: tp.Optional[tp.Union[dict, list]] = None,
        temperature: float = 0.7,
        k: int = 100,
        max_length: int = 100,
        device: str = "cuda",
        trained_model: str = None,
        n_samples: int = 1338,
        middle_padding: bool = False,
        random_initialization: bool = False,
    ) -> None:
        super().__init__(dataset, batch_size, max_tries_per_batch)
        self.data = self.dataset.get_single_df()
        self.categorical_columns = self.categorical_columns if encode_categories else []
        self.llm = llm
        self.experiment_dir = experiment_dir
        self.resume_from_checkpoint = resume_from_checkpoint
        self.epochs = epochs
        self.start_col = start_col
        self.start_col_dist = start_col_dist
        self.temperature = temperature
        self.k = k
        self.max_length = max_length
        self.device = device
        self.trained_model = trained_model
        self.n_samples = n_samples

        if middle_padding:
            self.model = TabMP(
                llm=self.llm,
                experiment_dir=self.experiment_dir,
                batch_size=self.batch_size,
                epochs=self.epochs,
                categorical_columns=self.categorical_columns,
            )
        else:
            self.model = Tab(
                llm=self.llm,
                experiment_dir=self.experiment_dir,
                batch_size=self.batch_size,
                epochs=self.epochs,
                categorical_columns=self.categorical_columns,
            )

        if self.trained_model is None:
            if not middle_padding and not random_initialization:
                path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
                self.model.model.load_state_dict(torch.load(path), strict=False)
        else:
            self.model.model.load_state_dict(
                torch.load(self.trained_model), strict=False
            )

    def preprocess(self) -> None:
        return super().preprocess()

    def train(self) -> None:
        if self.trained_model is None:
            self.model.fit(
                data=self.data,
                conditional_col=self.dataset.config["y_label"],
                resume_from_checkpoint=self.resume_from_checkpoint,
            )
            torch.save(
                self.model.model.state_dict(),
                os.path.join(self.experiment_dir, "model_" + str(self.dataset) + ".pt"),
            )
        else:
            self.model.init(
                data=self.data,
                conditional_col=self.dataset.config["y_label"],
            )

    def sample(self) -> pd.DataFrame:
        gen = self.model.sample(
            n_samples=self.n_samples,
            start_col=self.start_col,
            start_col_dist=self.start_col_dist,
            temperature=self.temperature,
            k=self.k,
            max_length=self.max_length,
            device=self.device,
            max_tries=self.max_tries_per_batch,
        )

        return gen
