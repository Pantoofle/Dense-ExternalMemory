# Automaton inference code

# Dense-ExternalMemory - DNC - Memory size extension
Neural Network with DNC structure and external memory. Goal is to study the use of memory for NN facing complex tasks

# Dependencies
`python 3.6`, `numpy`, `tensorflow >= 1.2.1`, `keras >= 1.0.5`, `h5py`, `plotly`, `matplotlib`, `graphviz`

You will also need the `libalf` C++ library to infere the automaton

# Libalf installation

It seems you can't install `libalf` in a `virtualenv`

Download the last sources in the [official libalf website](http://libalf.informatik.rwth-aachen.de/index.php?page=download)

You don't need the files `examples`

Extract each source with `tar xvjf <archive>`

Go to each created dir and run `make install`. You may need root privileges

BEWARE. You must install the different libs following the order displayed in the [download page](http://libalf.informatik.rwth-aachen.de/index.php?page=download)
Else, the install will be incomplete
 
 To uninstall `libalf`, go to each dir and run `make uninstall`

# Main installation
## With local pip

- Install the dependancies: `make pip_deps` or manually run the `pip install` for each package
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

Or run `multiple_run.sh` if you want the program to run multiple times

Edit it to set the number of runs
