API Reference
=============

Config
------

The **Config** module reads JSON configuration files that define
parameters for downloading and using datasets. These include:

- Paths for train and test splits.

- Dataset-specific settings for the implemented generation models.

- Column types (e.g., categorical, binary or continuous).

- The downstream task (e.g., binary or multiclass classification).

- Additional parameters as needed.

.. code-block:: python
   :caption: Example: Loading the Adult dataset configuration in a Python interactive session

   >>> from gentab.data import Config
   >>> config = Config("configs/adult.json")
   ✅ Config configs/adult.json loaded...

Dataset
-------

The **Dataset** component handles datasets based on the provided
configuration settings. Its primary responsibilities include loading
the data into memory and preprocessing it to ensure compatibility with
the implemented generative models.

This module offers functionalities such as:

- Creating structured data frames tailored for different generators.

- Performing random under-sampling to balance class distributions.

- Reducing memory consumption for improved efficiency, especially when
  working with larger datasets.

- Assessing the quality of synthetic datasets, focusing on fidelity
  and privacy.

Generator
---------

Tuner
-----

Evaluation
----------
