main: main.c vec.*
	gcc -std=c99 -g -O0 -lm -o main main.c vec.c

main-inline: main.c vec.*
	gcc -DINLINE -std=c99 -g -O0 -lm -o main-inline main.c vec.c

main-inline-cpp: main.c vec.*
	g++ -DINLINE -g -O0 -lm -o main-inline-cpp main.c vec.c

main.s: main.c vec.*
	gcc -S -std=c99 -g -O0 -lm main.c vec.c

main-inline.s: main.c vec.*
	gcc -DINLINE -S -std=c99 -g -O0 -lm main.c vec.c

histogram: histogram.c
	gcc -std=c99 -g -O3 -o histogram histogram.c -ldwarf
