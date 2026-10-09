CC = C:/mingw64/bin/gcc.exe
CFLAGS = -O3 -fopenmp -Wall -Wextra -Iinclude
LDFLAGS = -lm

SRCS = src/dataset.c src/sequential.c src/parallel_omp.c src/verify.c src/main.c
OBJS = $(SRCS:.c=.o)
TARGET = recommender.exe

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)

clean:
	del /Q $(TARGET) *.o 2>nul || rm -f $(TARGET) *.o
