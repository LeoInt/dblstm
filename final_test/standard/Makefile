objects = main.o net.o layer.o AffineTransform.o kernels.o

all: $(objects)
	nvcc -arch=sm_32 -lcurand -lcublas  $(objects) -o app

%.o: %.cu
	nvcc -x cu -arch=sm_32 -I. -lcurand -lcublas -dc $< -o $@

clean:
	rm -f *.o app
