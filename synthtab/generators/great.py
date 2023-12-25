from . import Generator
from synthtab.utils import console

from be_great import GReaT as GR
import pandas as pd
import typing as tp


class GReaT(Generator):
    """GReaT Class

    The GReaT class handles the whole generation flow. It is used to fine-tune a large language model for tabular data,
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
        conditional_col (str): Name of a feature/column on which the sampling can be conditioned
        conditional_col_dist (dict | list): Distribution of the feature/column specified by condtional_col
    """

    def __init__(
        self,
        dataset,
        llm: str = "distilgpt2",
        experiment_dir: str = "trainer_great",
        epochs: int = 100,
        batch_size: int = 8,
        max_tries_per_batch: int = 1338,
        efficient_finetuning: str = "",
        start_col: tp.Optional[str] = "",
        start_col_dist: tp.Optional[tp.Union[dict, list]] = None,
        temperature: float = 0.7,
        k: int = 100,
        max_length: int = 100,
        drop_nan: bool = False,
        device: str = "cuda",
        n_samples: int = 1338,
    ) -> None:
        super().__init__(dataset, batch_size, max_tries_per_batch)
        self.data = self.dataset.get_single_df()
        self.llm = llm
        self.experiment_dir = experiment_dir
        self.epochs = epochs
        self.efficient_finetuning = efficient_finetuning
        self.start_col = start_col
        self.start_col_dist = start_col_dist
        self.temperature = temperature
        self.k = k
        self.drop_nan = drop_nan
        self.max_length = max_length
        self.device = device
        self.n_samples = n_samples

        self.model = GR(
            llm=self.llm,
            experiment_dir=self.experiment_dir,
            batch_size=self.batch_size,
            epochs=self.epochs,
            efficient_finetuning=self.efficient_finetuning,
        )

    def preprocess(self) -> None:
        return super().preprocess()

    def train(self) -> None:
        self.model.fit(data=self.data)

    def sample(self) -> pd.DataFrame:
        return self.model.sample(
            n_samples=self.n_samples,
            start_col=self.start_col,
            start_col_dist=self.start_col_dist,
            temperature=self.temperature,
            k=self.k,
            max_length=self.max_length,
            device=self.device,
            drop_nan=self.drop_nan,
        )
