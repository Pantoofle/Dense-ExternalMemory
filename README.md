# Automaton inference code

# Dense-ExternalMemory - DNC - Memory size extension
Neural Network with DNC structure and external memory. Goal is to study the use of memory for NN facing complex tasks

# Dependencies
`python 3.6`, `numpy`, `tensorflow >= 1.2.1`, `keras >= 1.0.5`, `h5py`, `plotly `

# Installation
## With local pip

- Install the dependancies: `pip install --upgrade tensorflow keras h5py`
- Build the program (it justs make directories where models will be stored): `make build`
- Run the program with `make run` OR `make`

## With virtualenv

- Install virtualenv: `pip install virtualenv`
- Make new environnement: `virtualenv environnement`
- Run the environnement: `source environnement/bin/activate`
- Run the same things than for a classic local pip install

To uninstall everything, it's easy: 

`make clean` and `rm -rf environnement`

# Usage
```
make 
```
And then the program will ask if you want to load/save the models/results
