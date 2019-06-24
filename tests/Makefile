CUDA=/usr/local/cuda
LOFSINC=/home/khalbert/CODEBASE/LOFS/cm1tools-3.0

CFLAGS = -I$(LOFSINC)
LINKOPTS = -L$(LOFSINC) -L$(CUDA)/lib64 -lh5zzfp -lzfp -lhdf5_hl -lhdf5 -lnetcdf -lnetcdf_c++4 -lcm -lcudart -lm -fopenmp
CC = mpiCC 

all: solid_body 


solid_body: solid_body_vortex.cpp ../integrate.o ../datastructs.o
	$(CC) -O3 -std=c++11 -o solid_body solid_body_vortex.cpp ../integrate.o $(LOFSINC)/libcm.a $(CFLAGS) $(LINKOPTS) 

clean:
	rm -rf *.o 
