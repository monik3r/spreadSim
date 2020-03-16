CC=nvcc
CFLAGS+="--gpu-architecture=sm_61 -Xptxas -O3"

all:
	$(CC) $(CXXFLAGS) -o a.out main.cu

clean:
	rm a.out