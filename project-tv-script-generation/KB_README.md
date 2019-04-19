# Project Submission Notes

The project directory contains several Jupyter notebook files with similar names:

 * `dlnd_tv_script_generation_KB.ipynb`: The original notebook that I have developed __before__ moving my code to the GPU-enabled Project Workspace. So it has some extras, e.g. "side notes". However, it is not completed: there is nothing added to it beyond the "Training" step. 
 * `dlnd_tv_script_generation_v1.ipynb`: The is the first notebook I downloaded from my Project Workspace. It contains the results of the first training run. Ultimately, we had to abandon the trained model created in that run: there weas an indexing issue caused by "off-by-one" indexing (`enumerate(vocab, 1)`) I had applied to the `vocab_to_int` dictionary (see the `create_lookup_tables` function).
 * `dlnd_tv_script_generation.ipynb`: This is __the notebook__. I downloaded it from my Project Workspace upon the successful completion of the entire assignment. It contains __everything__, including the results of the last training run, as well as one of the generated TV scripts.
 
 # Offical Review from Udacity

## Meets Specifications

> Excellent work with the project! :smile: 👏
> 
> I hope the review helped you. If you feel there's something more that you would have preferred from this review please leave a comment. That would immensely help me to improve feedback for any future reviews I conduct including for further projects. Would appreciate your input too. Thanks!
> 
> Congratulations on finishing the project! :smile: :udacious:

## All Required Files and Tests
### The project submission contains the project notebook, called “dlnd_tv_script_generation.ipynb”.

### All the unit tests in project have passed.

## Pre-processing Data
### The function create_lookup_tables create two dictionaries:

 * Dictionary to go from the words to an id, we'll call vocab_to_int
 * Dictionary to go from the id to word, we'll call int_to_vocab
The function create_lookup_tables return these dictionaries as a tuple (vocab_to_int, int_to_vocab).

> Good work! You correctly established the two dictionaries.
> 
> Also, here's a good resource if you wish to understand word embeddings a bit more as well - http://veredshwartz.blogspot.in/2016/01/representing-words.html

### The function token_lookup returns a dict that can correctly tokenizes the provided symbols.

> Good job.
> 
> Here's a good resource discussing more preprocessing steps that you can try - http://datascience.stackexchange.com/questions/11402/preprocessing-text-before-use-rnn

## Batching Data
### The function batch_data breaks up word id's into the appropriate sequence lengths, such that only complete sequence lengths are constructed.

> Nice work!
> 
> You utilized a for loop to create your array and fill the indices with the required values.
> 
> Do you think you could complete this step without using a for loop and only using numpy? Numpy is optimized to handle complex operations so that you don't usually require loops and is faster as a result. So, try it out and see if you can optimize your code or not :)

### In the function batch_data, data is converted into Tensors and formatted with TensorDataset.

> Great job!
> 
> For your own benefit, you could also try to create your own function which utilizes Python Generators. If you haven't worked with Generators before, I highly recommend you try them out for batching the data instead of using TensorDataset and DataLoader. Generators probably don't offer any "distinct" advantages over the two, but it's a good python concept to be familiar with as per me.
> 
> Here's a couple of resources on generators that I found useful when I was trying to learn them -
> 
 * https://www.youtube.com/watch?v=bD05uGo_sVI
 * https://jeffknupp.com/blog/2013/04/07/improve-your-python-yield-and-generators-explained/

### Finally, batch_data returns a DataLoader for the batched training data.

## Build the RNN
### The RNN class has complete __init__, forward , and init_hidden functions.

> Well done!
> 
> Excellent work!
> 
> I have some questions for you to think about that might benefit you -
> 
> Does your nn.Linear() layer require any activation function?
> If you wanted to expand your class to be more generic such that you could have any number of layers you wanted, how would you go about that? Is that required (not just for this project, but for a general implementation which you could use anywhere for example)
> Are you familiar with Batch Normalization, yet? It's alright if not as I think you will come across this in the Classroom soon. But look it up and see how it compares to dropouts and whether they are useful for RNNs.
> The reason for the above questions is to try to make sure you are ready to explore alternatives. Often, Udacity's helper code tends to tie us down a bit and we restrict ourselves from exploring further. So try to think about the above and get the answers even if they don't seem valid for this project! :)

### The RNN must include an LSTM or GRU and at least one fully-connected layer. The LSTM/GRU should be correctly initialized, where relevant.

> Nicely done!
> 
> Especially for using the Embedding Layer. Have you checked out the documentation for it? Check it out - https://pytorch.org/docs/stable/nn.html#embedding and try to play around with some of the additional parameters for this method.
> 
> Also, here's a good resource if you wish to understand word embeddings a bit as well - http://veredshwartz.blogspot.in/2016/01/representing-words.html

## RNN Training
 * Enough epochs to get near a minimum in the training loss, no real upper limit on this. Just need to make sure the training loss is low and not improving much with more training.
 * Batch size is large enough to train efficiently, but small enough to fit the data in memory. No real “best” value here, depends on GPU memory usually.
 * Embedding dimension, significantly smaller than the size of the vocabulary, if you choose to use word embeddings
 * Hidden dimension (number of units in the hidden layers of the RNN) is large enough to fit the data well. Again, no real “best” value.
 * n_layers (number of layers in a GRU/LSTM) is between 1-3.
 * The sequence length (seq_length) here should be about the size of the length of sentences you want to look at before you generate the next word.
 * The learning rate shouldn’t be too large because the training algorithm won’t converge. But needs to be large enough that training doesn’t take forever.
> You selected a good set of hyperparameters.
> 
> * You get a low training loss with the number of epochs, which is good. We should tend to select the number of epochs such that you get a low training loss which reaches kind of a steady-state (not much change in value beyond a point).
> * I would recommend that you use a smaller batch size. Try to observe what happens if you lower it or increase it, first. Smaller batch sizes take too long to train. Larger batch sizes speed up the training but can degrade the quality of the model. Here is a useful resource to understand this better - http://stats.stackexchange.com/questions/164876/tradeoff-batch-size-vs-number-of-iterations-to-train-a-neural-network Also, do you think the batch size should be a power of 2 (like 16, 32, 64, etc) or your value of 300 is fine? Do you think these DL frameworks and tools are optimized to use values that are powers of 2 because of the hardware?
> * Your current RNN size (number of layers) fits the data well. What results do you think you will get when you increase or decrease that? First try increasing and see if your model converges better and has a lower loss or not. Changing this is limited to your system configuration as well, so you might not be able to have a very high value either.
> * Usually, we select the sequence length so that it matches the structure of data we are working with. There isn't really a rule of thumb though -http://stats.stackexchange.com/questions/158834/what-is-a-feasible-sequence-length-for-an-rnn-to-model. For a tougher challenge, try to think of having variable sequence lengths. https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e might help in this regard.
> * Good selection for the learning rate!
> 
> Couple of good things for you to try out - a slightly smaller learning rate with more epochs. As you can see your loss seems to still decrease, that means your model seems to still be learning so you can train for more epochs. But at the same time your learning rate results in the loss being a bit spiky (increases and decreases a bit). That usually means a higher learning rate. So you can try to reduce that as well.
> 
> Good work!

### The printed loss should decrease during training. The loss should reach a value lower than 3.5.

> Excellent job! You can still experiment more with your hyperparameters, if you'd like to. But you are getting good results.

### There is a provided answer that justifies choices about model size, sequence length, and other parameters.

> Well explained. You are starting to get an intuitive understanding of working with RNNs/LSTMs 🙂

> But I would recommend focussing less on expediting the training and more on understanding the fundamentals around hyperparameter tuning as well.

## Generate TV Script
### The generated script can vary in length, and should look structurally similar to the TV script in the dataset.

### It doesn’t have to be grammatically correct or make sense.

> Your generated script looks awesome! :clap:
> 
> Question for you to think about - How do you think you could design your model so that you can control how many words per dialogue your script has? Do you think that's possible with just deep learning?