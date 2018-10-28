# Model Definition
This is the second model generated for this project. It uses the same corpus
as Model 0, except this time it was trained to learn on the word level instead
of the character level. Because of this, it does not "write" in the same way
as Model 0. This one outputs whole words at a time, and then runs the last
word generated through the model to try and find the most useful following word.

As mentioned in the description for Model 0, this type of network has one rather
significant downside. The model reads punctuation as a whole "word," which means
that it generally requires rather significant post-processing to be usable in
any fashion. You can see an example of the raw output under the stories
directory.

However, since this model learns entire words, and "thinks" in entire words, it
cannot make spelling errors. The grammar _can_ be a little bit better as well,
but not always. Additionally, learning entire words means that the model's
training Epochs run __MUCH__ faster than character Epochs, which means the model
can train more intensely in the same approximate amount of time.

# Configuration Settings
```
model_cfg = {
    'rnn_size': 128,
    'rnn_layers': 4,
    'rnn_bidirectional': True,
    'max_length': 10,
    'max_words': 10000,
    'dim_embeddings': 100,
    'word_level': True,
}

train_cfg = {
    'line_delimited': False,
    'num_epochs': 500,
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

Similar to spelling, English grammar has vague rules that it is possible to
learn based on word order in a sentence. Because of this, the model learns
bidirectionally. However, because a word has a significant number of characters,
we reduce the `max_length` variable. The value of `10` is roughly the maximum
number of words we would expect to see in a correct english sentence.

Finally, the model was trained at the Word level.

As word level training is __MUCH__ faster, we were able to run the model for
500 epochs. Each Epoch required approximately 18 seconds to run. As it was ran
on the same Google Compute Platform as Model 0, it too took approximately 3
hours to run.

Training Size was set to 80% of the file, chosen randomly, to prevent the model
from cheating and regurgitating full strings found in the corpus. While there
was slightly less word-level data than I would have preferred to have, I still
kept dropout at 0 to see what the model would do on its own.

We are genuinely running a massive number of Epochs this time, so validation
was required. Once again this setting detects and prevents over-fitting of the
model to the corpus data.

# Run this Model
Navigate to the Model Files directory and open a python shell. Once there type
or copy the following:

```
from textgenrnn import textgenrnn
textgen = textgenrnn(weights_path='model_1_weights.hdf5',
                       vocab_path='model_1_vocab.json',
                       config_path='model_1_config.json')
```

Once those commands have been ran, Model 1 is loaded into memory. To generate
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
