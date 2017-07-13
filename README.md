# Dense-ExternalMemory - MAIN Branch
Neural Network with Dense structure and external memory. Goal is to study the use of memory for NN facing complex tasks

Based on 

Graves, Alex, et al. "Hybrid computing using a neural network with dynamic external memory." Nature 538.7626 (2016): 471-476.

# Work per branch
- auto: Efficient layer. much closer to the one described in the article. The application problem
is to infer an automaton from accepted and refused words. Uses `libalf` external library.
If you need the layer, use the one present in this one.

- base: a simple fully working straightforward network. Works on the test if number 
appeared in the previous sequence. Its a simpler version. With nan loss problems sometime.
Can be used as reference but for an efficient layer, you may want to look at the `auto` branch !

- shifou: Just for fun, can a Neural network beat you at Rock-Paper-Cisors ?

# Dependencies
`python 3.6`, `numpy`, `tensorflow`, `keras`. ANd some other specifics for each branch

# Usage
See in each branch README
