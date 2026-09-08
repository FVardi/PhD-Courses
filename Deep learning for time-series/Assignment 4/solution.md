# Part A

<i>

Read the method and experimental sections of the T-Loss, TS2Vec and TF-C papers. Explain the three learning
ideas briefly in your own words and compare:

- what constitutes related views or positive examples;
- what information the objective encourages the encoder to preserve;
- how a fixed-length representation is obtained; and
- how the representation is used for classification.

Trace the following code elements in the supplied checkouts. For each one, identify (i) the closest paper location - an equation, algorithm, figure or section, as applicable; (ii) its inputs and outputs; (iii) its immediate caller or call site; and (iv) its role in training or representation extraction. If PyTorch invokes a forward method indirectly through module(...) or nn.Sequential, state that call site. An implementation helper will not always have its own paper equation; in that case, state this and connect it to the appropriate method step. The verified checkout commits are
T-Loss 4aff592, TS2Vec b0088e1 and TF-C 9667582.
</i>

## T-loss
- Positive samples: a positive example $x^{pos}$ is a sub-series of the reference series $x^{ref}$.
- Objective: triple loss with one anchor, one positive, and $K\geq1$ negative samples $x_{k}^{neg}$. The objective is to have the distance between $x^{pos}$ and $x^{ref}$ be smaller than the sum of the distances between $x^{ref}$ and $x_k^{neg}$.
- Fixed length: the convolutional network ends with a representation $Y\in\mathbb{R}^{160 \times L}$ which is max pooled into $z\in\mathbb{R}^{160}$ and thereafter transformed through $f(x)=Wz+b$ where $W\in\mathbb{R}^{320\times160}$. The final dot product are therefore always between vectors of length 320. If the time series chosen in the beginning are too short for the full CNN they are left zero padded.
- Classification: an SVM with a radial basis function is trained on the features with their labels now avalable. It is evaluated in a train/test split.

## TS2Vec

## TF C