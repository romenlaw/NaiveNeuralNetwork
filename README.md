# NaiveNeuralNetwork
Naiive implementation of neural network as a study of Andrej Kaparthy's tutorials, among others.

MiniGrad files:
* nnn.py - NLP related classes used by the notebook
* visualiser.py - class to draw computational graph
* tests.ipynb - notebook going through the MiniGrad, incl. computational graphs
* Classifier_demo.ipynb - using the NLP to classify samples

MakeMore files:
* names.txt - list of names used as training data
* makemore.ipynb - notebook going through the makemore project, incl. analysing and plotting NN internals
* Bigram_W.pt - saved weights values already optimised and converged
* makemore_backprop.ipynb - makemore part4 backprop ninja
* makemore_wavenet.ipynb - makemore part5 using causal conv layers from Wevenet paper

NanoGPT files:
* gpt_dev.ipynb - development of nanoGPT following the lecture
* gpt_shakespeare.ipynb - generate shakespeare styled outout using tinysp as vocab of tokeniser

Tokeniser files:
* tokenisation.ipynb - notebook to study tokenisation
* tinysp_tokens.pt - list of tokens of tiny Shakespeare using the above tokeniser. Use torch.load to load.
* tinysp_vocab.json - tokens vocabulary created using tiny shakespeare as input data, vocab size around 10240
* tinysp_vocab_min.json - minimised vocab from 10237 to 8652 tokens by removing unsed ones
* tinysp_merges.json - byte pair merging results using tiny Shakespeare input data

Visualising Loss in 3D
* visualise-loss.ipynb - following [What's Inside a NN](https://towardsdatascience.com/whats-inside-a-neural-network-799daf235463)
