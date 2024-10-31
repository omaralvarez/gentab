from . import Evaluator
from gentab.utils import console, SPINNER, REFRESH, DEVICE

from types import ModuleType
from typing import Type
from typing_extensions import List, Union, Callable, Self

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch import Tensor
from sklearn.metrics import accuracy_score
from skorch.classifier import NeuralNetClassifier
from skorch.callbacks import EarlyStopping, EpochScoring
from skorch.helper import predefined_split


def reglu(x: Tensor) -> Tensor:
    """The ReGLU activation function from [1].
    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


def geglu(x: Tensor) -> Tensor:
    """The GEGLU activation function from [1].
    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)


class ReGLU(nn.Module):
    """The ReGLU activation function from [shazeer2020glu].

    Examples:
        .. testcode::

            module = ReGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return reglu(x)


class GEGLU(nn.Module):
    """The GEGLU activation function from [shazeer2020glu].

    Examples:
        .. testcode::

            module = GEGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return geglu(x)


def _make_nn_module(module_type: ModuleType, *args) -> nn.Module:
    return (
        (
            ReGLU()
            if module_type == "ReGLU"
            else GEGLU() if module_type == "GEGLU" else getattr(nn, module_type)(*args)
        )
        if isinstance(module_type, str)
        else module_type(*args)
    )


class MLPGorish(nn.Module):
    """The MLP model used in [gorishniy2021revisiting].

    The following scheme describes the architecture:

    .. code-block:: text

          MLP: (in) -> Block -> ... -> Block -> Linear -> (out)
        Block: (in) -> Linear -> Activation -> Dropout -> (out)

    Examples:
        .. testcode::

            x = torch.randn(4, 2)
            module = MLPGorish.make_baseline(x.shape[1], [3, 5], 0.1, 1)
            assert module(x).shape == (len(x), 1)

    References:
        * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
    """

    class Block(nn.Module):
        """The main building block of `MLP`."""

        def __init__(
            self,
            *,
            d_in: int,
            d_out: int,
            bias: bool,
            activation: ModuleType,
            dropout: float,
        ) -> None:
            super().__init__()
            self.linear = nn.Linear(d_in, d_out, bias)
            self.activation = _make_nn_module(activation)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: Tensor) -> Tensor:
            return self.dropout(self.activation(self.linear(x)))

    def __init__(
        self,
        *,
        d_in: int,
        d_layers: List[int],
        dropouts: Union[float, List[float]],
        activation: Union[str, Callable[[], nn.Module]],
        d_out: int,
    ) -> None:
        """
        Note:
            `make_baseline` is the recommended constructor.
        """
        super().__init__()
        if isinstance(dropouts, float):
            dropouts = [dropouts] * len(d_layers)
        assert len(d_layers) == len(dropouts)
        assert activation not in ["ReGLU", "GEGLU"]

        self.blocks = nn.ModuleList(
            [
                MLPGorish.Block(
                    d_in=d_layers[i - 1] if i else d_in,
                    d_out=d,
                    bias=True,
                    activation=activation,
                    dropout=dropout,
                )
                for i, (d, dropout) in enumerate(zip(d_layers, dropouts))
            ]
        )
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)

    @classmethod
    def make_baseline(
        cls: Type["MLP"],
        d_in: int,
        d_layers: List[int],
        dropout: float,
        d_out: int,
    ) -> "MLPGorish":
        """Create a "baseline" `MLP`.

        This variation of MLP was used in [gorishniy2021revisiting]. Features:

        * :code:`Activation` = :code:`ReLU`
        * all linear layers except for the first one and the last one are of the same dimension
        * the dropout rate is the same for all dropout layers

        Args:
            d_in: the input size
            d_layers: the dimensions of the linear layers. If there are more than two
                layers, then all of them except for the first and the last ones must
                have the same dimension. Valid examples: :code:`[]`, :code:`[8]`,
                :code:`[8, 16]`, :code:`[2, 2, 2, 2]`, :code:`[1, 2, 2, 4]`. Invalid
                example: :code:`[1, 2, 3, 4]`.
            dropout: the dropout rate for all hidden layers
            d_out: the output size
        Returns:
            MLP

        References:
            * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
        """
        assert isinstance(dropout, float)
        if len(d_layers) > 2:
            assert len(set(d_layers[1:-1])) == 1, (
                "if d_layers contains more than two elements, then"
                " all elements except for the first and the last ones must be equal."
            )
        return MLPGorish(
            d_in=d_in,
            d_layers=d_layers,  # type: ignore
            dropouts=dropout,
            activation="ReLU",
            d_out=d_out,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.float()
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        return x


class MLPClassifier:
    def __init__(
        self,
        input_features,
        num_classes,
        *args,
        epochs: int = 800,
        layers: List[int] = [128, 128],
        dropout: float = 0.1,
        batch_size: int = 8192,
        seed: int = 42,
        lr: float = 1e-5,
        weight_decay: float = 1e-6,
        device: str = DEVICE,
        **kwargs,
    ) -> None:
        self.input_features = input_features
        self.layers = layers
        self.dropout = dropout
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.lr = lr
        self.weigth_decay = weight_decay
        self.epochs = epochs
        self.device = device
        self.seed = seed
        self.acc = accuracy_score
        self.es = EarlyStopping(monitor="train_loss", patience=16)

    def fit(self, X, y) -> Self:
        self.model = MLPGorish.make_baseline(
            self.input_features, self.layers, self.dropout, self.num_classes
        )
        self.net = NeuralNetClassifier(
            self.model,
            criterion=CrossEntropyLoss,
            optimizer=AdamW,
            lr=self.lr,
            optimizer__weight_decay=self.weigth_decay,
            batch_size=self.batch_size,
            max_epochs=self.epochs,
            train_split=None,
            iterator_train__shuffle=True,
            device=self.device,
            callbacks=[self.es, EpochScoring(self.acc, lower_is_better=False)],
            verbose=0,
        )

        self.net.fit(X=X.to_numpy(), y=y)

        return self

    def predict(self, X):
        predictions = self.net.predict_proba(X.to_numpy())

        # Keep original 2D array and get pred. class
        return predictions.argmax(axis=1, keepdims=True)


class MLP(Evaluator):
    def __init__(
        self,
        generator,
        *args,
        layers: List[int] = [128, 128],
        dropout: float = 0.1,
        epochs: int = 1000,
        lr: float = 1e-5,
        batch_size: int = 1024,
        **kwargs,
    ) -> None:
        super().__init__(generator, **kwargs)
        self.model = MLPClassifier(
            self.dataset.num_features(),
            self.dataset.num_classes(),
            *args,
            layers,
            dropout,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            seed=self.seed,
            **kwargs,
        )

    def preprocess(self, X, y, X_test, y_test):
        X = self.dataset.encode_categories(X)
        X = self.dataset.get_normalized_features(X)
        X_test = self.dataset.encode_categories(X_test)
        X_test = self.dataset.get_normalized_features(X_test)

        y = self.generator.dataset.label_encoder_ohe.transform(y)
        y_test = self.generator.dataset.label_encoder_ohe.transform(y_test)

        return X, y, X_test, y_test

    def postprocess(self, pred):
        return self.dataset.decode_labels(pred)
