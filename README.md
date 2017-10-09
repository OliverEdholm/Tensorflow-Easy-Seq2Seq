# Tensorflow-Easy-Seq2Seq
A tool that allows you to easily train a Seq2Seq, get the embeddings and the outputs without having much knowledge about deep learning.


## Requirements
* Python 3.*
* Tensorflow
* SciPy
* Scikit-Learn
* tqdm
* matplotlib
* Numpy
* Six


## Adjusting parameters
Before you create your dataset file and train your model it's adviseable that you adjust the parameters first. This you can do by editing the file ```config.py```. If you don't understand a parameter it's best to leave it alone.

When running a script you can always pass the parameters directly and override the ```config.py``` parameters. To get to know what the parameters are you can type ```python3 script_name.py -h```.


## Creating dataset file
Firstly put your data in the **data** folder. Put your inputs in a file called inputs.txt and your outputs in a file called outputs.txt. Each line in the two files corresponds to one datapoint.

For example, chatbot dataset:

**data/inputs.txt**
```
Hello, my name is Oliver!
How are you?
Where do you come from?
Do you like carrots?
What do you think about North Korea?
```
 
**data/outputs.txt**
```
Hi, nice to meet you, my name is Fredrik.
I'm fine thank you!
I come from France.
No, I don't like carrots.
I don't like North Korea, what about you?
```

Thereafter you can run ```python3 create_data_set.py```. This will create a file called ```data_set.pkl``` in a folder called ```models```. This script does everything that you'll need, creating vocabulary and encoding the data to the correct format for training.


## Training
To train a model you firstly have to **create a dataset file**. After that it's as easy as running the script ```python3 train.py```


## Get model outputs
**Evaluating outputs**

To get an output from the Seq2Seq model you can write:
```python3 evaluate_outputs.py --eval_text="WRITE YOUR INPUT HERE"```

**Evaluating embeddings**

To plot embeddings you can run write:
```python3 evalute_embeddings --inputs_file_path=embedding_inputs.txt```
The embeddings will be dimensionality reducted with TSNE to two dimensions and plotted for you with matplotlib. The **inputs_file_path** is the path with a file that has the inputs written in the same way as in normal **inputs.txt**.

**Get outputs and embeddings from model programmatically**

In the ```src``` folder there's the files ```embedding.py``` and ```predicting.py```, import from these to do it programmatically.


### Todos
* Tensorboard.
* Example data that you can download.
* TFRecord data!
* Other Seq2Seq models than embedding attention Seq2Seq.


### Other
Made by Oliver Edholm, 15 years old.
