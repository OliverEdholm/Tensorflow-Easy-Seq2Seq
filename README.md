# Tensorflow-Easy-Seq2Seq
A tool that allows you to easily train a Seq2Seq, get the embeddings and the outputs without having much knowledge about deep learning.


### Requirements
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



### Creating dataset
Firstly put your data in the **data** folder. Put your inputs in a file called inputs.txt and your outputs in a file called outputs.txt. Each line in the two files corresponds to one datapoint.

For example, chatbot dataset:

**inputs.txt**
```
Hello, my name is Oliver!
How are you?
Where do you come from?
Do you like carrots?
What do you think about communism?
```
 
**outputs.txt**
```
Hi, nice to meet you, my name is Fredrik.
I'm fine thank you!
I come from France.
No, I don't like carrots.
I don't like communism, what about you?
```

Thereafter you can run ```python3 create_data_set.py```. This will create a file called ```data_set.pkl``` in a folder called ```models```. This script does everything that you'll need, creating vocabulary and encoding the data to the correct format for training.


**Embedding images and saving them**

**Get embedding from trained Convolutional autoencoder**
To train a Convolutional autoencoder to vectorize images do this command:
```
python3 autoencoder_training.py
```
You can get a look at the hyperparameters using.
```
python3 autoencoder_training.py --help
```
The same principles follow in all the other scripts.

**Embedding with autoencoder**
Just do this command.
```
python3 vectorize_autoencoder.py
```


**Get embedding from pretrained models**
Just do this command.
```
python3 vectorize_pretrained.py --model_path=<model_path> --model_type=<model_type> --layer_to_extract=<layer_to_extract>
```
What does these arguments mean?

**model_path**: Path to pretrained model. e.g ./inception_v4.ckpt

**model_type**: Type of model, either VGG16, VGG19, InceptionV3 or InceptionV4. e.g InceptionV4

**layer_to_extract**: Which layer to take vector from. e.g Mixed_7a

This command will save the vectors in a file in the vectors folder and will print out the path to the vectors for later
use or evaluation at the end of the program.


**Evaluating**
To evaluate your vectors you can do this command.
```
python3 evaluation.py --vectors_path=<vectors_path> --image_path=<image_path>
```
What does these arguments mean?

**vectors_path**: Where vectors are saved. e.g vectors/vectors_1

**image_path**: Image to evaluate on, i.e the image to check nearest neighbour on. e.g img.jpg


### Todos
* Tensorboard.
* Example data that you can download.
* TFRecord data!
* Other Seq2Seq models than embedding attention Seq2Seq.


### Other
Made by Oliver Edholm, 15 years old.
