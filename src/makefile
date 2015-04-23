ifeq ($(shell uname -s), Darwin)
	CC = clang
else
	CC = gcc
	CFLAGS += -pthread -lm
endif

CFLAGS += -Ofast -std=c99

all: cwe.c
	${CC} cwe.c ${CFLAGS} -o cwe
