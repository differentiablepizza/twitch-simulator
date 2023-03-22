# Introduction

These are scripts for treatment, training, evaluation of a fine tuned version of GPT-Neo from data of Twitch Chat logs and streamer transcribed conversations. Currently, the successfully trained models are being saved on [HuggingFace](https://huggingface.co/Differentiable).

# Instalation

This project is known to work with Python 3.10+. To install all necessary packages, open a terminal and run the following command in the root directory:

```bash
pip install -r requirements.txt
```

Depending on your PIP installation, you'll need to use `pip3` isntead of `pip`. This usually occurs when you have more than one version of Python installed on your computer (2.7 and 3.10 for example).

# Training

The model can be trained with the file `model-neo.py`, the way it is currently set up requires a GPU with Cuda enabled. In case you don't have a capable GPU or the model ran out of memory, you can also try to run the model directly on your CPU at the cost of a longer training time. For that, comment out lines with `.to('cuda')` and set `no_cuda` to `True` and comment out the parameter `fp16` on the `TrainingArguments` constructor.

Then, execute the trainer as a notebook or do

```bash
python model-neo.py
```

# Evaluation

The file `evaluator.py` is able to generate text based on the input model. Make sure that `model_name` is set to the correct path and model version that you want to use. This script doesn't require a GPU to function. The input prompt is the text contained `prompt`, which you can change at will. The response is printed out at the end.

For evaluation, execute the file as a notebook or do

```bash
python evaluator.py
```
