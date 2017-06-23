# To disable openmp parallelization, set USE_OPENMP to 0
USE_OPENMP := 1

# To enable debugging, set USE_DEBUG to 1
USE_DEBUG := 0

ifeq ($(USE_OPENMP), 1)
	FLAG += -fopenmp
endif

ifeq ($(USE_DEBUG), 1)
	FLAG += -g
endif

main.out: data.cpp tree.cpp bagging.cpp main.cpp
	g++ $^ -o $@ $(FLAG)

clean:
	rm -f *.o *.out