
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void convolutionCPU(unsigned int *image, unsigned int *mask, unsigned int *output, int M, int N) {
    int offset = N / 2;
    for (int r = 0; r < M; r++) {
        for (int c = 0; c < M; c++) {
            unsigned int res = 0;
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    int row = r - offset + i;
                    int col = c - offset + j;
                    if (row >= 0 && row < M && col >= 0 && col < M)
                        res += image[row * M + col] * mask[i * N + j];
                }
            }
            output[r * M + c] = res;
        }
    }
}

int main(int argc, char **argv) {
    int M = atoi(argv[1]), N = atoi(argv[2]);
    unsigned int *img = malloc(M*M*sizeof(int)), *msk = malloc(N*N*sizeof(int)), *out = malloc(M*M*sizeof(int));
    clock_t start = clock();
    convolutionCPU(img, msk, out, M, N);
    printf("%f", (double)(clock() - start) / CLOCKS_PER_SEC);
    free(img); free(msk); free(out);
    return 0;
}
