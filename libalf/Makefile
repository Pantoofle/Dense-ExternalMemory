NAME=online

CPP=g++
C_FLAGS=-lalf

all: run

online: online.cpp
	$(CPP) $(NAME).cpp $(C_FLAGS) -o $(NAME)

run: online
	LD_LIBRARY_PATH=/usr/local/lib ./$(NAME)

clean:
	rm -f $(NAME)
