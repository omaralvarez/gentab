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
   :caption: Loading the Adult dataset configuration in a Python interactive session

   >>> from gentab.data import Config
   >>> config = Config("configs/adult.json")
   ✅ Config configs/adult.json loaded...

Dataset
-------

Generator
---------

Tuner
-----

Evaluation
----------
