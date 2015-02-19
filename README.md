# Word Predictor
Predicts your next word. Can be used for predictive typing as seen in the SwiftKey keyboard and iOS QuickType.

## Usage
Create an instance of ```WordPredictor``` and start training it by providing it sample text. After training, the next term can be predicted by calling the ```predict()``` method with the preceding phrase. The ```predict()``` method returns a Patricia-trie of the words, making it possible to perform fast prefix lookup. The following shows how to train the predictor with the Gutenberg corpus provided by NLTK and predict three terms based on the user input:

    from word_predictor.word_predictor import WordPredictor
	import nltk

    wp = WordPredictor()
    for corpus in nltk.corpus.gutenberg.fileids():
        wp.learn_from_text(nltk.corpus.gutenberg.raw(corpus))
    print "Ready"
    while True:
        phrase = raw_input()
        print wp.predict(phrase).terms()[0:3]

## How it works
The predictor learns a n-th order Markov chain based on the training data. It stores a phrase-term sparse matrix of term sequence frequencies, with rows representing the ```n``` previous terms and columns representing the following term. The dimension of the matrix is ![V^OxV](https://raw.github.com/mkaloer/wordPredictor/master/doc/eq_matrix_size.png), where ![V](https://raw.github.com/mkaloer/wordPredictor/master/doc/eq_v.png) is the size of the vocabulary and ![O](https://raw.github.com/mkaloer/wordPredictor/master/doc/eq_o.png) the order of the markov chain. When a word is to be predicted, the probability:

![P(Xn+1|Xn,Xn-1,...,Xn-O+1)](https://raw.github.com/mkaloer/wordPredictor/master/doc/eq_predict.png)

is calculated by estimating the joint probabilities from the frequencies in the training data.

### Hashing
Each term in the vocabulary is associated with an index starting from 0. Every time a new term is found, its id is the previous id incremented by one. To convert a sequence of terms into a phrase-term matrix index, the following hash function is used:
Matrix index for phrase ![sn sn-1 ... s0](https://raw.github.com/mkaloer/wordPredictor/master/doc/eq_phrase.png), where ![si](https://raw.github.com/mkaloer/wordPredictor/master/doc/eq_phrase_i.png) is the term at index ```ì```, is calculated as:

![sum(i=0..n) (index(s_i) + V^i)](https://raw.github.com/mkaloer/wordPredictor/master/doc/eq_hash.png)

Where ![V](https://raw.github.com/mkaloer/wordPredictor/master/doc/eq_v.png) is the length of the vocabulary.

## Known Issues
### Bad Hash Function
The size of the range of the hash function is ```V^O```, which, even with a relatively small vocabulary size and Markov chain order, causes some indices to be unrepresentable by a 32-bit integer. This induces memory errors in the SciPy library. A solution could be to make the assumption that the probabilities are independent, and then calculate the joint probabilities instead of measuring them, reducing the matrix dimensions to ```V*O```.

## Future Work
### Context-based suggestions
The ability to specify different contexts (such as email, sms, specific contacts, etc.) and retrieve predictions based on that context.
