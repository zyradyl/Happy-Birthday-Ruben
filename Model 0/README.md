# Model Definition
This is the first model I generated for this iteration of the project. The goal
was to create a model purely trained on Ruben's writing. This is a character
level model which means it actually has learned how to write one letter at a
time. Because of this, it both copies Ruben's writing style (Character models
trained on Shakespeare learned a rudimentary form of iambic pentameter) and
can create novel words on its own. At higher temperature levels this can lead
to bizarre non-sense words.

The biggest benefit of this type of model is that it can generally be used
without any degree of post-processing. Post-processing will undoubtedly make
it even better, but the output at least looks like English. You can contrast
this with the unprocessed output of Model 1.

# Configuration Settings
```
model_cfg = {
    'rnn_size': 128,
    'rnn_layers': 4,
    'rnn_bidirectional': True,
    'max_length': 40,
    'max_words': 10000,
    'dim_embeddings': 100,
    'word_level': False,
}

train_cfg = {
    'line_delimited': False,
    'num_epochs': 50,
    'gen_epochs': 10,
    'batch_size': 1024,
    'train_size': 0.8,
    'dropout': 0.0,
    'max_gen_length': 300,
    'validation': True,
    'is_csv': False
}
```

This model uses a 4 layer LSTM (Long Short-Term Memory) network where each
layer is composed of 128 cells. TensorFlow also has a final "attention" layer
that combines the capabilities of all layers before it.

Since English spelling follows, however vaguely, certain rules, the model was
trained bidirectionally. This should prevent the model from outputting constant
strings of consonants or vowels.

The model was ran for 50 epochs, with each Epoch requiring approximately 150
seconds on a Google Compute Platform equipped with a Tesla K80. Total time to
create the model was approximately 3 hours. Once I figure out how to utilize
GCP's Tensor Processing Units I may recreate these models with further training.

Training Size was set to 80% of the file, chosen randomly, to prevent the model
from cheating and regurgitating full strings found in the corpus. Since we had
a fair amount of data (approximately 400k character sequences) dropout was set
to 0 to let the model self adjust.

Finally, since we were running a large number of Epochs, validation was enabled
so that we could detect a potential over-fitting of the model to the data.

# Run this Model
Navigate to the Model Files directory and open a python shell. Once there type
or copy the following:

```
from textgenrnn import textgenrnn
textgen = textgenrnn(weights_path='model_0_weights.hdf5',
                       vocab_path='model_0_vocab.json',
                       config_path='model_0_config.json')
```

Once those commands have been ran, Model 0 is loaded into memory. To generate
text at fairly conservative creativity and have it output to your screen, you
can use:

```
textgen.generate_samples(max_gen_length=1000, temperature=0.2)
```

Simply increase or decrease the `max_gen_length` to change how much output you
get. You can make the model more creative by increasing the `Temperature`
variable.

If you would like to shave the output to a file, you can use the following
string. Again, adjust `max_gen_length` to change how much you get, and
`Temperature` to change the creativity.

```
textgen.generate_to_file('textgenrnn_texts.txt', max_gen_length=1000, temperature=0.2)
```
