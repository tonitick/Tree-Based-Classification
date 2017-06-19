SRCS := data.cpp tree.cpp bagging.cpp main.cpp

main.out: data.cpp tree.cpp bagging.cpp main.cpp
	g++ $^ -o $@ -fopenmp -g

clean:
	rm -f *.o *.out