SRCS := data.cpp tree.cpp main.cpp

main.out: data.cpp tree.cpp main.cpp
	g++ $^ -o $@ -fopenmp

clean:
	rm -f *.o *.out