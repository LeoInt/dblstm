objects = main.o net.o layer.o AffineTransform.o kernels.o

all: $(objects)
	g++  -lgsl -lgslcblas -lstdc++ $(objects) -o app

%.o: %.cpp
	g++ -x cpp -I. -lgsl -lgslcblas -lstdc++ -dc $< -o $@

clean:
	rm -f *.o app