![](https://i.imgur.com/joqFojF.png)
# Flair-Rap
A simple language model for generating polish rap lyrics based on flair library

This project supports word level and char level language model that generates polish rap sequences based on texts from genius.com
## Requirements
All requirements are located in the `requirements.txt` file 
## Usage
### Creating dictionaries
Both word and char model needs a dictionary of tokens that it uses. In order to generate new dictionary you can use create_char_dict() and crate_word_dict() functions
```Python
from flair_rap import Flair_Rap

Flair_Rap.create_char_dict(files=['corpus/full_text.txt'], filename= 'dicts/char_dict')
Flair_Rap.create_word_dict(files=['corpus/full_text.txt'], filename= 'dicts/word_dict')
```
* `files` is an array of file to generate the dict form a 
* `filename` is a name of the file to be generated
### Learing 
In order to teach the model you can use `learn_word_model()` and `learn_char_model()` functions

```python
from flair_rap import Flair_Rap

Flair_Rap.learn_char_model(dict_file = 'dicts/char_dict',is_forward_lm = True, hidden_size=128, nlayers=2, sequence_length=15, mini_batch_size=16, max_epochs =20)
Flair_Rap.learn_word_model(dict_file = 'dicts/word_dict',is_forward_lm = True, hidden_size=128, nlayers=2, sequence_length=10, mini_batch_size=16, max_epochs =20)
```
Parameters are pretty self explanatory

The corpus that the model learns form must be organized according to flair corpus structure : https://github.com/zalandoresearch/flair/blob/master/resources/docs/TUTORIAL_9_TRAINING_LM_EMBEDDINGS.md
Models will be saved in the `/traning/` folder 
### Generating sequence
In orger to generate sequence you can use the `predict_sequence()` function 

```python
from flair_rap import Flair_Rap

Flair_Rap.predict_sequence(dataset = 'models/word-lm.pt', seq_length = 100, mode='word')
```
* `dataset` is the path to your trained model

* `seq_length` is the length of the sequence ot be generated(if `mode` is `"word"` it fill be number of words if `mode` is `"char"` it will be number of chars)

* `mode` is the mode to be used (`"word"` or `"char"`)
