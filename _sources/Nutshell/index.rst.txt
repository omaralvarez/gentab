In a nutshell
=============

GenTab defines 5 main entities: Config, Dataset, Generator, Evaluator,
and Tuner. The following diagram illustrates their typical dataflow
within GenTab:

.. figure:: /_static/figs/architecture.svg
   :align: center
   :width: 70%

   System overview of GenTab

Sample code to implement a basic GenTab workflow:

.. code-block:: Python

                from gentab.generators import AutoDiffusion
                from gentab.evaluators import LightGBM
                from gentab.tuners import AutoDiffusionTuner
                from gentab.data import Config, Dataset

                config = Config("configs/adult.json")

                dataset = Dataset(config)
                dataset.merge_classes({
                  "<=50K": ["<=50K."], ">50K": [">50K."]
                })
                dataset.reduce_mem()

                trials = 10

                generator = AutoDiffusion(dataset)
                evaluator = LightGBM(generator)

                tuner = AutoDiffusionTuner(evaluator, trials)
                tuner.tune()

                tuner.save_to_disk()

In the code we

1. Parse a Config
2. Create a Dataset and preprocess it
3. Create a Generator
4. Create an Evaluator
5. Create a Tuner and perform hyperparameter tuning for the desired generator
6. Store the best dataset and model parameters obtained after ten tries

