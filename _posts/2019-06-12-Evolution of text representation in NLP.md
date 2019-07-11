---
layout:            post
title:             "Evolution of text representation in NLP"
menutitle:         "Evolution of text representation in NLP"
category:          Blog posts
author:            asallab
tags:              
---
# The Journey to BERT 
_"A unique journey of text representations"_

The research in text and natural language processing can be summarized in the journey to answer the question: "What is the best representation of language". In fact, this is the most important question for any ML task, for all media: text, speech and image.

# NLP tasks
- Classification
Sentiment, Spam
- Sequence2Sequence (same as semantic segmentation)
MT, QA

# Text representation
Computer and Neural Nets can only understand digital forms.

Images are represented as pixel values, mapped in 2D spatial map, with each pixel value represent color intensity as that location in the 2D image space.

Speech signals are represented as the voltage activation of the microphone signal recording after being digitized with an ADC.

_What about text?_

Words are digitized as indices in vocabulary vectors.

Words as indices is not good since higher index means higher value, which is not true.

## Word and Sentence Word representations
__Image data analogy__
In image data, the smallest unit of representation is the pixel. A pixel is usually represented by its intensity on a 0 to 1 scale, which is called grey-scale if we are to neglect colors. For colored images, we represent each pixel with 3 numbers, referred to as channels, each representing an intensity component for Red, Green or Blue colors, and their mix represents a color in the color space.

Given the pixel representation, an image is usually a 2D matrix made of such pixels, hence the minimum representation is 2D matrix of intensities, or 3D tensor including the colors components.

If we have video data, then we have many of such 2D images, called frames, and hence we have 4D tensor representations.

In text, the smallest unit of representation can be a word or characters, which are known as word or character level representations. For simplicity, let's start our analogy by assuming that the smallest language representation unit is the word. 

A word can be represented in many ways, which we will discuss below. The range or of words representation is a function of the dimension of what is called "vocabulary" or vocab for short, which defines the list of all language words (or the subset we are interested in). Using this vocabulary, there are many representations possible to represent each word. In most cases, a word is represented using 1D vector.

Due to the existence of vacabulary, most of the time we don't keep track of all possible vocabulary words, or in the best case, we might ignore some cosmetic inflections of the words, like suffix, prefix,...etc. Keeping only a subset of the vocabulary creates another issue called Out-Of-Vocabulary (OOV) words, where we might encounter words that are not in the vocabulary vector. One of the easiest approaches to overcome this is to use character level instead of word level representations.

Since language is a sequence of words, we have another dimension to account for the sequence, same as we had in case of video. However, for text, we can have variable length sequence to represent a sentence for example (not all sentences are of the same length). For now we will assume we have a maximum length that we will pad all sentences to, and hence we have 2D matrix. You can think of character level representation of words in same way we go from words to sentences, with the exception that the range of characters is limited by the alphabet size of the language, and no existence of vocabulary in this case.

In addition, we might define the text structure in a more hierarichal way, where a document is made of paragraphs, and paragraphs made of sentences, and sentences made of words,..etc. Hence, sometimes we can represent the text in 3D or 4D tensors.

In most of NLP research we are either concerned of word or sentence representations. Sentences are sometimes referred to as segments. Also in some cases, sentence representations are called document representation, assuming that the document is made of one long sequence of words, without any further structure. 

Next we will review some methods for both cases.



# Word representatins:
## One-Hot-Encoding (OHE)

Each index is mapped to a OHE vector

Issues:

- No similarity relation between words vectors. Dot product is a measure of simialrity. Since OHE gives orthogonal vecotrs by definition, we have 0 value in all cases:

[0 0 0 1 0 0] T . [1 0 0 0 0 0 0] = 0

- Also, its a very sparse vecor; vocab can reach 13M words.


## Distributional similarity representation: Word vectors
To encode similarity of words in the word representation, we better use the context of the word. In this way, words that occur in similar context will have near representation. Also, in this way their dot products will not be 0 as in case of OHE, but will have high value if they are similary, and low otherwise. This is the objective.

Learn a dense representation of words, say a vector of 50 or 100 dimensions of real numbers. Such vectors represent a projection of the word in a space called the "Embedding" space. This vector is called an "Embedding" for short.

If we stack all the vectors in one table we have an "Embedding table", or a Look-up-table (LUT).

The table is 2D; the first dimension is the vocabulary size. The 2nd dimension is the embedding vector dimensionality we want to project to (50, 100, typically 300).

As a digression, it's better to keeps the stems in the vocabulary, and for some applications, it might be also good to keep separate entries for the suffix, prefix,..etc (in general: morphemes), instead of keeping separate entries for the different morphologies of the same word, because simply it's hard to keep all of them!
As an example, consider the word incorrect, you better keep an entry for "in" and "correct", because you might enounter other morphemes of "correct", like "correctly", "correctness",..etc. This method reduces the OOV.

If we represent the word as OHE, then dot product of the word with the LUT gives the word vector.
In that sense, the LUT can be thought as a _weight matrix_.

_How to find the values in the LUT?_

As we said the LUT can be thought as weights matrix.

__We want to have vectors that encode similarity__, in a way that synonyms are near to each others, same as in wordnet.

To do that, we come up with an artificial task that encode similarity. The most trivial task is the "validity" task as follows: stick every 2 neighboring words vectors together, and train a network to produce "1" for that pair, meaning that this pair can occur together. And then flip one of them with any random word, and train the net to produce "0" for the new pair, marking that as "invalid". 

Of course such trivial task by itself is not useful in anything, but after we finish, we have a LUT that encodes words that occur together in near vectors in the embedding space.

There are two famous other tasks (at least in word2vec):
- Skip-gram:
x1, x2 <-- x3 --> x4

Predict the context words given the center one. We ask the net to predict the precedding and secceeding n words in a window, given a center word.

It's important to note that, the loss in this case is the probability of all context words given the center word.
See CS224n, lec 2, 00:26:00.

$J(theta) = Pi_t=1..T(Pi_j=-m..m, j!= 0(p(wt+j | wt;theta))$

Where wt+j are the context words, while wt is the center one. Each p(wt+j | wt;theta) represents the similarity between each context word wt+j and the cetner word wt. theta is the weights of the word vector of wt, when it's a center word.

This similarity can be calculated by a softmax, say u is the word vector of wt+j and v is the vector of wt:

p(wt+j | wt;theta) = softmax(wt+j, wt) = exp(u.v)/sum_vocab(exp(u_i.v)

Note we sum over the vocab, to get a prob over all possibilities, where the sample space in this case is all vocab words.

In this case, a word can take two roles: as a target word (center word), as a context word (outside word). We keep two vectors in our LUT for each word, one for "outside" role, and another one for the "center" role. You could keep one vector per word, but empirically keeping 2 works better. In this case we have 2 LUT's.

See CS224n, lec 2, 00:30:00.

The position of the word in the window is not considered.

- CBOW:
x1, x2--> x3 <-- x4

Predict the center word given the context words. The context words are given as bag-of-words (position is dropped).


In some sense this encodes the context.

The position of the word in the window is not considered.

## GloVe


## FastText


## ELMO
The word2vec provides the same vector regardless the context of the word. This prevents word sense disambiguation
ELMO provides contextualized vecotrs, learned by mixing the current word vecotr and its negithbors

ELMO can work at char level, which reduces the OOV rate.
 
# Sentence or document representation
We are not interested in words represntation, but in sequence of words; document or at least sentence represenations.

## Bag of Words (BoW)

More compact representation is to encode the vocab vector for each sentence, and mark the locations of the words of vocab appearing in that sentence as 1, while all other locations are 0.

It's still a sparse vector, but the sentence representation is more compact (one 13M vector for sentence/document instead of word).

However, the order of words is lost.

## Bag of words vectors
The BoW idea words for sequence of words in the same manner described above. 
A more advanced version is to average the words vectors of the sentence words:

S = 1/N(W1 + W2 + ....+WN)

Still order is missing

## RNN

For sequence learning problems, either for classification or sequence production (seq2seq), what we want to do is to summarize the sequence in one vector. It is also desirable to encode the order of words somehow.

RNN's can act as a state machine, keeping an internal hidden state that starts with some value with the start of the sequence. The state evolves every time a word of the sequence is fed to the RNN, in addition to its previous state (recurrence). In some sense, the state evolution is conditioned on the previous state, like in Bayes filters. After all the sequence is fed, the final state encodes and summarizes the whole sequence information in order.

RNN's remained the SoTA till 2017 and still widely used. The main disadv is their tendency to forget and diverge with longer sequences, which is handled with gated units like in LSTMs and GRU's and state reset.


## RAE


NLP community is familiar to "parse trees, which originate from linguistic origins, where a sentence is read/parsed according to a certain order that encodes it's syntactic struture.
In the same fashion, RAE's aim to compute a sentence representation by computing a representation of 2 vectors at a time. THe choice of which words vectors to parse is done according to a binary parse tree.
The tree could be a known syntactic tree obtained by linguistic rules, or it can be "guessed".
The way "guessing" happens is through an AE. The AE takes 2 vecors, encodes them, then reconstruct them again, scoring an error between the original and reconstructed vecrtors. This process is repeated for the whole sequence. The pair giving min reconstruction error is picked.
When a pair is parsed, they are replaced with their AE bottle neck representation. This make the tree shrinks with every parsing step.
This process is repeated in a tree fashion until only one vector remains, that represent the sequence.

Unlike RNN's, Recursive models do not keep internal state, but instead, the same computational block is re-called recursively on every new input.

## Convs2s
The recursive and recurrent models are sequential by nature; we need to wait the output of the previous step to start the next. Such sequential nature is not suitable for parallelization, which prevents us from using GPU's. 
ConvS2S models aims at using conv1d kernels to summarize the sequence. Conv kernel has a "FoV". the FoV of deeper feature maps neurons span larger part of the input. This provides a good hierarichal representation, with the depther controls the span of the representation.
This technique is way faster than recurrent or recursive models.

However, since the sequence length requires deeper models, it makes it difficult to learn dependencies between distant words (see https://arxiv.org/pdf/1706.03762.pdf)

## Attention
A trending techniue is to add gates that "Selects" to encode or discard vectors in the final representation.
Like in the BoW vectors, where the vectors are averaged, we can think of a weighted average.
The weights are "attention" gates. This is called "attention mechanism".
The weights can be "soft" or "hard" attention.
In "hard" attention, we keep the max weight vector
In "soft" attention, we use a "softmax" over the weights.

V = softmax(W's).V's



The two main techniques of attention are Bahddanau attention https://arxiv.org/abs/1409.0473 or Luong attention https://arxiv.org/pdf/1508.04025.pdf. While the first acts on the hiddent LSTM states, the latter works on the LSTM outputs.

There are few ways to learn the weights https://arxiv.org/pdf/1508.04025.pdf:
- Dot: W_i = W_i.W's: the higher the simialiry of word vectors, the higher the weight
- General: W_i.W_learnable.W's: learn the weights
- Concat: W_learnable.[W_i;W_j]

Like in BoW, the order is lost. That's why attention is mixed with recurrence as in https://arxiv.org/abs/1409.0473.

Another way is to use positional encoding (Trans and Convs2s); either learnable or constant (Trans=sine/cosine): http://jalammar.github.io/illustrated-transformer/

In Bahdanau, seq2seq is done by summarizing the input sequence in one vector coming out of the encoder. The decoder task is to produce unaralled tokens, conditioned on the sumamrized encoder state, and the previous decoded token. Attention comes into play to increase dependency not only on the last state of the encoder, but also the previous ones, in addition to the decoded states, not only the previous one.

## HATT
Text is a sequence of words, but this sequence has a hierarichal structure beyond only a flat sequence. We can have document, structured into paragraphs, then into sentences.
In HATT, similar idea to Bahddanau attention is used, but first the sequence is tokenized into sentences, then words.
High score on IMDB

## Transformer
http://jalammar.github.io/illustrated-transformer/
To get rid of recurrence, the transformer fully relies on attention gates. The attention weights are calculated using dot, in parallel for all sequence words, producing vectors for all the words in parallel.
The process is repeated multiple steps as needed (layers).
The final representation is a number of refined vectors. 
To summarize into one vectors, we can add 1 attention weight gates at the end. In BERT a special symbol is introduced for that.

The decoder also depends on attention, and condition on the encoder states in addition to "only" the previous decoded tokens, same as in seq2seq with RNN.

__Positional Embeddings__
https://mlexplained.com/2019/06/30/paper-dissected-xlnet-generalized-autoregressive-pretraining-for-language-understanding-explained/
This is great, but there is one glaring flaw we haven’t yet addressed. A simple attention-based model cannot handle positional information. The solution to this problem used in the Transformer is simple: add positional embeddings to each word that express information regarding the position of each word in a sequence.

```
embeddings = word_embs + pos_embs
h = [embeddings] + [None for _ in attention_layers]

for i, attention in enumerate(attention_layers):
    h[i+1] = attention(queries=h[i], keys=h[i], values=h[i])
```

## Universal transformer
The universal transformer (UT) is a mix of RNN and Transfromer. On one hand, the state evolution is parallel-in-time (matrix multiplication with multi-head attention mechanism) as in the Transfromer. On the other hand, the evolution of the hidden states happen as an autoregression or recurrence in both the encoder and decoder. In that sense, the depth (Number of attention blocks) of the encoder or decoder is dynamic according to the recurrence time steps. Also, the dynamic depth is decided per position according to dynamic halting mechanism.

## XL-Transformer = Recurrent Transformer
The Xtra Long Transformer is an extension over the normal transformer.
The normal Transformer is based on tokenizing the input sequence into segments.
Such segmentation might lack long context modeling, especially for tasks like AR language models.
The XL-Trans adopts a recurrent decoding technique, accounting for a summary state vector for previous context.
Decoding of next tokens is conditioned on the current segment tokens, in addition to previous context state, under the full attention mechanism lf the transformer.

https://mlexplained.com/2019/06/30/paper-dissected-xlnet-generalized-autoregressive-pretraining-for-language-understanding-explained/

"Transformers were a game-changer in NLP due to their incredible performance and ease of training. However, they had a major drawback compared to RNNs: they had limited context.

__Not an issue for RNN__
Suppose you had a 50000-word long piece of text that you wanted to feed to a model. _Feeding this into any model all at once would be infeasible given memory constraints_. For an RNN you could work around this by simply chunking the text, then feeding the RNN one chunk at a time _without resetting the hidden state between chunks_. This works because the RNN is recurrent and as long as you keep the hidden state, the RNN can “remember” previous chunks, giving it a theoretically infinite memory.

For a Transformer, this is impossible because Transformers take fixed-length sequences as input have no notion of “memory”. All its computations are stateless (this was actually one of the major selling points of the Transformer: no state means computation can be parallelized)  so there is an upper limit on the distance of relationships a vanilla Transformer can model.

__Add recurrence to Trans at segment level__
The Transformer XL is a simple extension of the Transformer that seeks to resolve this problem. The idea is simple: what if we added recurrence to the Transformer? Adding recurrence at the word level would just make it an RNN. But what if we added recurrence at a “segment” level. In other words, what if we added state between consecutive sequences of computations? The Transformer XL accomplishes this by caching the hidden states of the previous sequence and passing them as keys/values when processing the current sequence. For example, if we had the consecutive sentences.

“I went to the store. I bought some cookies.”

we can feed “I went to the store.” first, cache the outputs of the intermediate layers, then feed the sentence “I bought some cookies.” and the cached outputs into the model.

```
memory = init_memory()
h = [embeddings] + [None for _ in attention_layers]

for i, attention in enumerate(attention_layers):
    ext_h_i = concat([h[i], memory[i])
    h[i+1] = Attention(queries=h[i], keys=ext_h_i, values=ext_h_i)

# save memory
memory = h
```

__Relative Positional Embeddings__
This idea is great, but there is one flaw: position. In the Transformer, we handled position using positional embeddings. The first word in a sentence would have the “first position” embedding added to it, the second word would have the “second position” embedding added, and so on. But with recurrence, what happens to the positional embedding of the first word in the previous segment? If we’re caching the Transformer outputs, what happens to the positional embedding of the first word in the current segment?

You can easily have sequences like [1,2,3,4], [1,2,3,4]--> [1,2,3,4,1,2,3,4]

To address these issues, the Transformer XL introduces the notion of relative positional embeddings. Instead of having an embedding represent the absolute position of a word, the Transformer XL uses an embedding to encode the relative distance between words. This embedding is used while computing the attention score between any two words: in other words, the relative positional embedding enables the model to learn how to compute the attention score for words that are n  words before and after the current word.

"


## ULMFiT

An intuitive, but new trend in NLP is transfer learning. ULMFiT started this wave by training an encoder for language modeling on wiki text, then fine tune by adding classification or decoder layers for the target tasks.
The idea is well established almost 5 years earlier in the CV community, where networks like VGG, ResNet,...etc are widely used after being trained on imagenet. Those are called backbones.

## GPT
The GPT mixes the idea of transfer learning with Transformer

https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf

BERT is based on GPT, but using AE LM: MLM and Next sentence

GPT = ULMFiT (AR LM) + Transformer

BERT = GPT + Bidir LM (MLM, Next sent)


## GPT2
https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf

GPT2 = Large GPT 

https://blog.floydhub.com/gpt2/

GPT-2 stands for “Generative Pretrained Transformer 2”:

- “Generative” means the model was trained to predict (or “generate”) the next token in a sequence of tokens in an unsupervised way. In other words, the model was thrown a whole lot of raw text data and asked to figure out the statistical features of the text to create more text.

- “Pretrained” means OpenAI created a large and powerful language model, which they fine-tuned for specific tasks like machine translation later on. This is kind of like transfer learning with Imagenet, except it’s for NLP. This retraining approach became quite popular in 2018 and is very likely to be a trend that continues throughout 2019.

- “Transformer” means OpenAI used the transformer architecture, as opposed to an RNN, LSTM, GRU or any other 3/4 letter acronym you have in mind. I’m not going to discuss the transformer architecture in detail since there’s already another great article on the FloydHub blog that explains how it works.

- “2” means this isn’t the first time they’re trying this whole GPT thing out.




## BERT
BERT is based on the Trans and GPT, with the introduction of Bi-dir LM, with two tasks: MLM and Next-sentence prediction
BERT is said to be the imagenet moment
http://jalammar.github.io/illustrated-bert/


The main contribution of BERT is the masked language model pre-training.

### AR vs AE
https://medium.com/dair-ai/xlnet-outperforms-bert-on-several-nlp-tasks-9ec867bb563b

Two pretraining objectives that have been successful for pretraining neural networks used in transfer learning NLP are autoregressive (AR) language modeling and autoencoding (AE).

Autoregressive language modeling is not able to model deep bidirectional context which has recently been found to be effective in several downstream NLP tasks such as sentiment analysis and question answering.

On the other hand, autoencoding based pretraining aims to reconstruct original data from corrupted data. A popular example of such modeling is used in BERT, an effective state-of-the-art technique used to address several NLP tasks.

One advantage of models like BERT is that bidirectional contexts can be used in the reconstruction process, something that AR language modeling lacks. However, BERT partially masks the input (i.e. tokens) during pretraining which results in a pre-training-finetune discrepancy. In addition, BERT assumes independence on predicted tokens, something which AR models allow for via the product rule which is used to factorize the joint probability of predicted tokens. This could potentially help with the pretrain-finetune discrepancy found in BERT.

### MLM
https://mlexplained.com/2019/01/07/paper-dissected-bert-pre-training-of-deep-bidirectional-transformers-for-language-understanding-explained/
Language modeling – although it sounds formidable – is essentially just predicting words in a blank. More formally, given a context, a language model predicts the probability of a word occurring in that context. For instance, given the following context

“The _____ sat on the mat”

where _____ is the word we are trying to predict, a language model might tell us that the word “cat” would fill the blank 50% of the time, “dog” would fill the blank 20% of the time, etc.

Language models have generally been trained from “left to right“. They are given a sequence of words, then have to predict the next word.

For instance, if the network is given the sequence

“Which Sesame Street”

the network is trained to predict what word comes next. This approach is effective when we actually want to generate sentences. We can predict the next word, append that to the sequence, then predict the next word, etc..

However this method is not the only way of modeling language. There is no need to train language models from left to right when we are not interested in generating sentences. 
__This is one of the key traits of BERT: Instead of predicting the next word after a sequence of words, BERT randomly masks words in the sentence and predicts them.__

___This is crucial since this forces the model to use information from the entire sentence simulatenously – regardless of the position – to make a good predictions.___

#### MLM Training
__[MASK] will not be in the real data!__ This is what XLNet/Permutation LM tries to solve. MLM has a hacky solution, replace with random words instead of MASK, then ask the model to reconstruct the original word (in place of the random one). In this way the model doesn't depend on the token [MASK] all the time.

https://mlexplained.com/2019/01/07/paper-dissected-bert-pre-training-of-deep-bidirectional-transformers-for-language-understanding-explained/

__But__ This must be done with caution though since swapping words at random is a very strong form of noise that can potentially confuse the model and degrade results.


"Though masked language modeling seems like a relatively simply task, there are a couple of subtleties to doing it right.

The most naive way of training a model on masked language modeling is to randomly replace a set percentage of words with a special [MASK] token and to require the model to predict the masked token. Indeed, in the majority of cases, this is what BERT is trained to do. For each example, 15% of the tokens are selected uniformly at random to be masked.

The problem with this approach is that the model only tries to predict when the [MASK] token is present in the input. This means the model can “slack-off” when the input token is not the [MASK] token, meaning the hidden state for the input token might not be as rich as it could be. What we really want the model to do is to try to predict the correct tokens regardless of what token is present in the input.

To solve this problem, the authors sometimes replace words in the sentence with random words instead of the [MASK] token. This must be done with caution though since swapping words at random is a very strong form of noise that can potentially confuse the model and degrade results. This is why BERT only swaps 10% of the 15% tokens selected for masking (in total 1.5% of all tokens) and leaves 10% of the tokens intact (it does not mask or swap them). The remaining 80% are actually replaced with the [MASK] token."

__Why not BiLSTM?__
If you are familiar with the NLP literature, you might know about bidirectional LSTM based language models and wonder why they are insufficient. Bidirectional LSTM based language models train a standard left-to-right language model and also train a right-to-left (reverse) language model that predicts previous words from subsequent words. Actually, this is what methods like ELMo and ULMFiT did. In ELMo, there is a single LSTM for the forward language model and backward language model each. The crucial difference is this: neither LSTM takes both the previous and subsequent tokens into account at the same time.

![AE/MLM vs AR LM](https://i2.wp.com/mlexplained.com/wp-content/uploads/2019/01/Screen-Shot-2019-01-03-at-4.40.22-PM.png?resize=1024%2C207&ssl=1)

https://mlexplained.com/2019/06/30/paper-dissected-xlnet-generalized-autoregressive-pretraining-for-language-understanding-explained/
Traditional language models are trained in a left-to-right fashion to predict the next word given a sequence of words. This has the limitation of not requiring the model to model bidirectional context. What does “bidirectional context” mean? For some words, their meaning might only become apparent when you look at both the left and right context simultaneously. The simultaneous part is important: models like ELMo train two separate models that each take the left and right context into account but do not train a model that uses both at the same time.

BERT solves this problem by introducing a new task in the form of masked language modeling. The idea is simple: instead of predicting the next token in a sequence, BERT replaces random words in the input sentence with the special [MASK] token and attempts to predict what the original token was. In addition to this, BERT used the powerful Transformer architecture (which I will explain next) to incorporate information from the entire input sentence.

#### Next sentence prediction

In addition to masked language modeling, BERT also uses a next sentence prediction task to pretrain the model for tasks that require an understanding of the relationship between two sentences (e.g. question answering and natural language inference).

When taking two sentences as input, BERT separates the sentences with a special [SEP] token. During training, BERT is fed two sentences and 50% of the time the second sentence comes after the first one and 50% of the time it is a randomly sampled sentence. BERT is then required to predict whether the second sentence is random or not.

![NextSentence](https://i1.wp.com/mlexplained.com/wp-content/uploads/2019/01/Screen-Shot-2019-01-05-at-5.21.26-PM.png?w=634&ssl=1)


### BERT vs ULMFiT

ULMFiT is based on LSTM, BERT on Trans
ULMFiT uses AR LM, BERT uses AE LM (MLM or Next sent prediction)

### BERT vs. ELMo
ELMo is based on LSTM, BERT on Trans
ELMo produces word embeddings (although contextual, i.e. it uses the context of the sentence, but only to produce word embeddings and for that it uses LM, BERT produces sentence embeddings

### BERT vs GPT
https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf

BERT is based on GPT, but using AE LM: MLM and Next sentence

GPT = ULMFiT (AR LM) + Transformer

BERT = GPT + Bidir LM (MLM, Next sent)


### BERT vs GPT2

https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf

GPT2 = Large GPT 

https://blog.floydhub.com/gpt2/

GPT-2 stands for “Generative Pretrained Transformer 2”:

- “Generative” means the model was trained to predict (or “generate”) the next token in a sequence of tokens in an unsupervised way. In other words, the model was thrown a whole lot of raw text data and asked to figure out the statistical features of the text to create more text.

- “Pretrained” means OpenAI created a large and powerful language model, which they fine-tuned for specific tasks like machine translation later on. This is kind of like transfer learning with Imagenet, except it’s for NLP. This retraining approach became quite popular in 2018 and is very likely to be a trend that continues throughout 2019.

- “Transformer” means OpenAI used the transformer architecture, as opposed to an RNN, LSTM, GRU or any other 3/4 letter acronym you have in mind. I’m not going to discuss the transformer architecture in detail since there’s already another great article on the FloydHub blog that explains how it works.

- “2” means this isn’t the first time they’re trying this whole GPT thing out.




### Permutations LM explained

https://mlexplained.com/2019/06/30/paper-dissected-xlnet-generalized-autoregressive-pretraining-for-language-understanding-explained/

![PLM](https://i0.wp.com/mlexplained.com/wp-content/uploads/2019/06/ezgif.com-gif-maker-2.gif?zoom=1.25&fit=960%2C360&ssl=1)

# What’s Wrong with BERT?
BERT was already a revolutionary method with strong performance across multiple tasks, but it wasn’t without its flaws. XLNet pointed out two major problems with BERT.

1. The [MASK] token used in training does not appear during fine-tuning

BERT is trained to predict tokens replaced with the special [MASK] token. The problem is that the [MASK] token – which is at the center of training BERT – never appears when fine-tuning BERT on downstream tasks.

This can cause a whole host of issues such as:

What does BERT do for tokens that are not replaced with [MASK]?
In most cases, BERT can simply copy non-masked tokens to the output. So would it really learn to produce meaningful representations for non-masked tokens?
Of course, BERT still needs to accumulate information from all words in a sequence to denoise [MASK] tokens. But what happens if there are no [MASK] tokens in the input sentence?
There are no clear answers to the above problems, but it’s clear that the [MASK] token is a source of train-test skew that can cause problems during fine-tuning. The authors of BERT were aware of this issue and tried to circumvent these problems by replacing some tokens with random real tokens during training instead of replacing them with the [MASK] token. However, this only constituted 10% of the noise. When only 15% of the tokens are noised to begin with, this only amounts to 1.5% of all the tokens, so is a lackluster solution.

2. BERT generates predictions independently

Another problem stems from the fact that BERT predicts masked tokens in parallel. Let’s illustrate with an example: Suppose we have the following sentence.

I went to [MASK] [MASK] and saw the [MASK] [MASK] [MASK].

One possible way to fill this out is

I went to New York and saw the Empire State building.

Another way is

I went to San Francisco and saw the Golden Gate bridge.

However, the sentence

I went to San Francisco and saw the Empire State building

is not valid. Despite this, BERT predicts all masked positions in parallel, meaning that during training, it does not learn to handle dependencies between predicting simultaneously masked tokens. In other words, it does not learn dependencies between its own predictions. Since BERT is not actually used to unmask tokens, this is not directly a problem. The reason this can be a problem is that this reduces the number of dependencies BERT learns at once, making the learning signal weaker than it could be.

Note that neither of these problems is present in traditional language models. Language models have no [MASK] token and generate all words in a specified order so it learns dependencies between all the words in a sentence.

### Permutations LM explained

https://mlexplained.com/2019/06/30/paper-dissected-xlnet-generalized-autoregressive-pretraining-for-language-understanding-explained/

![PLM](https://i0.wp.com/mlexplained.com/wp-content/uploads/2019/06/ezgif.com-gif-maker-2.gif?zoom=1.25&fit=960%2C360&ssl=1)

__The Best of Both Worlds: Permutation Language Modeling__
Of course, despite its flaws, BERT has one major advantage over traditional language models: it captures bidirectional context. This bidirectionality was a crucial factor in BERT’s success, so going back to traditional language modeling is simply not an option. The question then becomes: _can we train a model to incorporate bidirectional context while avoiding the [MASK] token and parallel independent predictions?_

The answer is yes: XLNet does this by introducing a variant of language modeling called “permutation language modeling”. Permutation language models are trained to predict one token given preceding context like traditional language model, but instead of predicting the tokens in sequential order, it predicts tokens in some random order. To illustrate, let’s take the following sentence as an example:

I like cats more than dogs.

A traditional language model (AR LM) would predict the tokens in the order

“I”, “like”, “cats”, “more”, “than”, “dogs”

where each token uses all previous tokens as context.

![AR LM](ttps://i2.wp.com/mlexplained.com/wp-content/uploads/2019/06/ezgif.com-gif-maker-1.gif?zoom=1.25&resize=447%2C170)

In permutation language modeling, the order of prediction is not necessarily left to right and is sampled randomly instead. For instance, it could be:

“cats”, “than”, “I”, “more”, “dogs”, “like”

where “than” would be conditioned on seeing “cats“, “I” would be conditioned on seeing “cats, than” and so on. The following animation demonstrates this.

![An example of how a permutation language model would predict tokens for a certain permutation. Shaded words are provided as input to the model while unshaded words are masked out.](https://i0.wp.com/mlexplained.com/wp-content/uploads/2019/06/ezgif.com-gif-maker-2.gif?zoom=1.25&resize=421%2C158)


![MLM vs PLM](https://i1.wp.com/mlexplained.com/wp-content/uploads/2019/06/Screen-Shot-2019-06-22-at-5.38.12-PM.png?resize=1024%2C567&ssl=1)

The conceptual difference between BERT and XLNet. Transparent words are masked out so the model cannot rely on them. XLNet learns to predict the words in an arbitrary order but in an autoregressive, sequential manner (not necessarily left-to-right). BERT predicts all masked words simultaneously.

For example, when we feed the order above, we feed the correct positions:

(“3, cats”), (“5, than”), (“1, I”), (“4, more”), (“6, dogs”), (“2, like”)

So when we feed (3, cats), and we want to predict (5, than), this is equivalent to MLM with masks everywhere else (XX means the word to predict, MASK means not existing word:

input: [MASK], [MASK], [cats], [MASK], [XX], [MASK] --> predict: [than] --> p(than | cats)

input: [XX], [MASK], [cats], [MASK], [than], [MASK] --> predict: [I] --> p(I | cats, than)

....

So the prediction is in AR manner, but each step is using AE manner--> AE + AR.
In all steps, the positional encoding is fed to keep the order.

"

__Does it mean we are feeding wrong sequence to the model?__
_Use positional embedding: feed the word with its true order in the __sequence__ but with permuted order in __time__ _

As a word of caution, in permutation language modeling, we are not changing the actual order of words in the input sentence. We are just changing the order in which we predict them. If you’re used to thinking of language modeling in a sequential manner, this may be hard to grasp: how can we change the order in which we predict tokens while not changing the order in which we feed them to the model? Just remember that Transformers use masking to choose which inputs to feed into the model and use positional embeddings to provide positional information. This means that we can feed input tokens in an arbitrary order simply by adjusting the mask to cover the tokens we want to hide from the model. As long as we keep the positional embeddings consistent, the model will see the tokens “in the right order”.


## XLNet


__XLNet = AR + AE = tile: XLNet: Generalized Autoregressive Pretraining for Language Understanding__

https://medium.com/dair-ai/xlnet-outperforms-bert-on-several-nlp-tasks-9ec867bb563b

The proposed model (XLNet) borrows ideas from the two types of language pretraining objectives (AR and AE) while avoiding their limitations.

https://towardsdatascience.com/what-is-xlnet-and-why-it-outperforms-bert-8d8fce710335

https://medium.com/dair-ai/xlnet-outperforms-bert-on-several-nlp-tasks-9ec867bb563b



Same as BERT is based on the Transformer, XLNet is based on XL-Transformer.

The main contribution of XLNet is the replacement of the masked LM with permutation LM.In MLM, a new token, [MASK], is introduced, which not naturally found in normal text.

Moreover, MLM is based on AR LM concept (predict the next tokens based on past predictions). Such AR approach can only learn forward or backward dependencies, but not both.

In permutation LM, all possible dependencies are modeled, without the need to any mask:

Taking the same example as in the paper, suppose we have a sequence of tokens [x1, x2, x3, x4], and we want to model the dependency of x3 given all others, fwd and bwd, then we need to model;

Fwd:

p(x3|x1, x2)

p(x3|x1)

p(x3|x2)

Bwd:

p(x3|x4)


In addition to the idea of Xtra Long context in XL-Transformer, we need to model also x3 given any past context:

p(x3|mem_context)

Those probs are the result of x3 being in the 1st, 2nd, 3rd, 4th positions, with any other token in the other positions.

(x3, ?, ?, ?)

(?, x3, ?, ?)

(?,?, x3, ?)

(?, ?, ?, x3)

? Means x1, x2 or x4

For example predictong x3 from the sequence (x2, x1, x3, x4) means modeling:

p(x3| x1, x2, x4)

Which is mixing:

Fwd: p(x3| x1, x2)
Bwd: p(x3|x4)

And hence, capturing both fwd and bwd dependencies.

![MLM_vs_PLM_2](https://i1.wp.com/mlexplained.com/wp-content/uploads/2019/06/Screen-Shot-2019-06-22-at-5.38.12-PM.png?resize=1024%2C567&ssl=1)
The conceptual difference between BERT and XLNet. Transparent words are masked out so the model cannot rely on them. XLNet learns to predict the words in an arbitrary order but in an autoregressive, sequential manner (not necessarily left-to-right). BERT predicts all masked words simultaneously.

For each permutation order, say [x1, x4, x2, x3], the tokens are fed in order, in addition to their original pos embedding:
[(1,x1), (4,x4), (2,x2). (3,x3)]

The prediction goes in steps--> auto regressive from left to right according to the permuted order:
1- input: [0, START] --> predicts: (1,x1) --> models: p(x1 | [START]) --> MLM equivalent: [START], [MASK] --> [START], [x1]
2- input: (1,x1) --> predicts: (4,x4) --> models: p(x4|[START], x1) --> MLM equivalent: [START], [x1], [MASK] --> [START], [x1], [x4]
3- input: (1,x1), (4,x4) --> predicts: (2, x2) -->models: p(x2 | [START], x1, x4) --> MLM equivalent: [START], [x1], [MASK], [x4] --> [START], [x1], [x3], [x4]
4- input: (1,x1), (4,x4), (2,x2) --> predicts: (3, x3) --> models: p(x3 | [START], x1, x4, x2), note here, the correct order is kept thant to pos emb, so it's actually modeling p(x3 | [START], x1, x2, x4) --> MLM equivalent: [START], [x1], [MASK], [x3],[x4] --> [START], [x1], [x2], [x3], [x4]

The above steps are auto regressive (AR). Within each step, AE objective (reconstruction) is used to model different dependencies for both directions (bi-directional).


__Revisit issues with MLM of BERT__

1. The [MASK] token used in training does not appear during fine-tuning
Now, we feed only tokens that appear in text

2. BERT generates predictions independently
The steps prediction solves the 2nd issue in BERT: simultaneous prediction

Bidirectionality is kept.

__What else differs from BERT in XLNet: Transformer XL + Rel Pos Emb__

Aside from using permutation language modeling, XLNet improves upon BERT by using the Transformer XL as its base architecture. The Transformer XL showed state-of-the-art performance in language modeling, so was a natural choice for XLNet.

XLNet uses the two key ideas from Transformer XL: relative positional embeddings and the recurrence mechanism. The hidden states from the previous segment are cached and frozen while conducting the permutation language modeling for the current segment. Since all the words from the previous segment are used as input, there is no need to know the permutation order of the previous segment.

The authors found that using the Transformer XL improved performance over BERT, even in the absence of permutation language modeling. This shows that better language models can lead to better representations, and thus better performance across a multitude of tasks, motivating the necessity of research into language modeling.

__Handling Position: Two-Stream Self-Attention__
For language models using Transformers like BERT, when predicting a token at position i , the entire embedding for that word is masked out including the positional embedding. This means that the model is cut off from knowledge regarding the position of the token it is predicting.

Suppose we are predicting the word “like” in the sentence

“I like cats more than dogs“

where the previous words in the permutation were “more” and “dogs“. The content stream would encode information for the words “more” and “dogs“. The query stream would encode the positional information of “like” and the information from the content stream which would then be used to predict the word “like“.

__Handling optimization difficulties__
Permutation language modeling is more challenging compared to traditional language modeling which apparently causes the model to converge slowly. To address this problem, the authors chose to predict the last n  tokens in the permutation instead of predicting the entire sentence from scratch.

![Skip_PLM](https://i1.wp.com/mlexplained.com/wp-content/uploads/2019/06/Screen-Shot-2019-06-22-at-6.10.05-PM.png?resize=768%2C796&ssl=1)

For the example above, say the permutation is [x1, x4, x2, x3], then we have 5 steps. Instead we do the only last 2 for example

3- input: (1,x1), (4,x4) --> predicts: (2, x2) -->models: p(x2 | [START], x1, x4) --> MLM equivalent: [START], [x1], [MASK], [x4] --> [START], [x1], [x3], [x4]

4- input: (1,x1), (4,x4), (2,x2) --> predicts: (3, x3) --> models: p(x3 | [START], x1, x4, x2), note here, the correct order is kept thant to pos emb, so it's actually modeling p(x3 | [START], x1, x2, x4) --> MLM equivalent: [START], [x1], [MASK], [x3],[x4] --> [START], [x1], [x2], [x3], [x4]

__Modeling Multiple Segments__

Many downstream tasks that we would want to use XLNet to take multiple segments of text as input. For instance, in the case of question answering, we take a question and answer as input. 

To enable the model to distinguish between words in different segments,  BERT learns a segment embedding. In contrast, XLNet learns an embedding that represents whether two words are from the same segment. This embedding is used during attention computation between any two words. This idea is very similar to the relative positional encoding idea introduced in the Transformer XL. Similar to BERT, XLNet also feeds special [CLS] and [SEP] tokens to delimit the input sequences. The advantage of this scheme is that XLNet can now be extended to tasks that take arbitrary numbers of sequences as input.

# Adapters

Google

Parameter-Efficient Transfer Learning for NLP

Add auto encoder like modules called adapters. Their parameters are the onlh fine tuned weights. 
The rest remains constant across modeks snd tasks.
The adapter mosuoe role is to encode thd task specific input into a common soace betweem tasks
then spits out a decoded vector of the same input size such that thecwholecore trained model remains the same in terms of compatible dimensions.

Adapter init weights as identity (just copy), so that no change to the original model.

"Adapter-based tuning relates to multi-task and continual learning. Multi-task learning also results in compact models. However, multi-task learning requires simultaneous access to all tasks, which adapter-based tuning does not require. Continual learning systems aim to learn from an endless stream of tasks. This paradigm is challenging because networks forget previous tasks after re-training (McCloskey &Cohen, 1989; French, 1999). Adapters differ in that the tasks do not interact and the shared parameters are frozen. This means that the model has perfect memory of previous tasks using a small number of task-specific parameters."



Efficient parametrization of multi-domain deep neural networks.

"Wepresent a strategy for tuning a large text model on several downstream tasks. Our strategy has three key properties: (i) it attains good performance, (ii) it permits training on tasks sequentially, that is, it does not require simultaneous access to all datasets, and (iii) it adds only a small number of additional parameters per task. These properties are especially useful in the context of cloud services, where many models need to be trained on a series of downstream tasks, so a high degree of sharing is desirable. To achieve these properties, we propose a new bottleneck adapter module. Tuning with adapter modules involves adding a small number of new parameters to a model, which are trained on the downstream task (Rebuffi et al., 2017). When performing vanilla fine-tuning of deep networks, a modification is made to the top layer of the network. This is required because the label spaces and losses for the upstream and downstream tasks differ. Adapter modules perform more general architectural modifications to re-purpose a pretrained network for a downstream task. In particular, the adapter tuning strategy involves injecting new layers into the original network. The weights of the original network are untouched, whilst the new adapter layers are initialized at random. In standard fine-tuning, the new top-layer and the original weights are co-trained. In contrast, in adaptertuning, the parameters of the original network are frozen and therefore may be shared by many tasks. Adapter modules have two main features: a small number of parameters, and a near-identity initialization. The adapter modules need to be small compared to the layers of the original network. This means that the total model size grows relatively slowly when more tasks are added. A near-identity initialization is required for stable training of the adapted model; we investigate this empirically in Section 3.6. By initializing the adapters to a near-identity function, original network is unaffected when training starts. During training, the adapters may then be activated to change the distribution of activations throughout the network. The adapter modules may also be ignored if not required; in Section 3.6 we observe that some adapters have more influence on the network than others. We also observe that if the initialization deviates too far from the identity function, the model may fail to train."

"To limit the number of parameters, we propose a bottleneck architecture. The adapters first project the original d-dimensional features into a smaller dimension, m, apply a nonlinearity, then project back to d dimensions. The total number of parameters added per layer, including biases, is 2md +d+m. By setting m d, we limit the number of parameters added per task; in practice, we use around 0.5 − 8% of the parameters of the original model. The bottleneck dimension, m, provides a simple means to tradeoff performance with parameter efficiency. The adapter module itself has a skip-connection internally. With the skip-connection, if the parameters of the projection layers are initialized to near-zero, the module is initialized to an approximate identity function."



## What's next?
The idea of transfer from pretrained LM, and fine tune for other tasks seems to be taking over since its introduction (or formalization) in ULMFiT.
From ULMFiT to GPT to BERT to XLNet, two factors are changind:

1. The encoder model.
2. The LM training method.

Number seems to be good candidate for extensions.
For number 1, hierarichal models like HATT, still not exploited for longer context modeling.



