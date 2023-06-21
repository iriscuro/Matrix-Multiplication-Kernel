#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

const int TILE_WIDTH = 16;  // Tamaño del bloque en las operaciones de matriz

// Generar matriz aleatoria
void generateRandomMatrix(float* matrix, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matrix[i * size + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
}


// FIGURE 4.16: A tiled Matrix Multiplication Kernel using shared memory
__global__ void MatrixMulKernel_Figure4_16(float* M, float* N, float* P, int Width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    float Pvalue = 0;

    for (int ph = 0; ph < Width / TILE_WIDTH; ++ph) {
        Mds[ty][tx] = M[Row * Width + ph * TILE_WIDTH + tx];
        Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * Width + Col];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }

    P[Row * Width + Col] = Pvalue;
}

// FIGURE 4.3: A simple matrix multiplication kernel using one thread to compute one P element
__global__ void MatrixMulKernel_Figure4_3(float* M, float* N, float* P, int Width) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if (Row < Width && Col < Width) {
        float Pvalue = 0;

        for (int k = 0; k < Width; ++k) {
            Pvalue += M[Row * Width + k] * N[k * Width + Col];
        }

        P[Row * Width + Col] = Pvalue;
    }
}

int main() {
    // Configurar semilla aleatoria
    srand(time(NULL));

    // Tamaños de las matrices para probar
    int matrixSizes[] = {64, 128, 256, 512, 1024};
    int numSizes = sizeof(matrixSizes) / sizeof(matrixSizes[0]);

    for (int i = 0; i < numSizes; ++i) {
        int matrixSize = matrixSizes[i];

        // Alojar memoria en el host (CPU)
        float* hostMatrixA = new float[matrixSize * matrixSize];
        float* hostMatrixB = new float[matrixSize * matrixSize];
        float* hostResult = new float[matrixSize * matrixSize];

        // Generar matrices aleatorias
        generateRandomMatrix(hostMatrixA, matrixSize);
        generateRandomMatrix(hostMatrixB, matrixSize);

        // Alojar memoria en el dispositivo (GPU)
        float* deviceMatrixA;
        float* deviceMatrixB;
        float* deviceResult;
        cudaMalloc((void**)&deviceMatrixA, matrixSize * matrixSize * sizeof(float));
        cudaMalloc((void**)&deviceMatrixB, matrixSize * matrixSize * sizeof(float));
        cudaMalloc((void**)&deviceResult, matrixSize * matrixSize * sizeof(float));

        // Copiar matrices del host al dispositivo
        cudaMemcpy(deviceMatrixA, hostMatrixA, matrixSize * matrixSize * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceMatrixB, hostMatrixB, matrixSize * matrixSize * sizeof(float), cudaMemcpyHostToDevice);

        // Configurar dimensiones de bloque y de cuadrícula
        dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
        dim3 numBlocks(matrixSize / TILE_WIDTH, matrixSize / TILE_WIDTH);

        // Ejecutar y medir tiempo 
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        

        // Ejecutar y medir tiempo para FIGURE 4.16
        cudaEventRecord(start);
        MatrixMulKernel_Figure4_16<<<numBlocks, threadsPerBlock>>>(deviceMatrixA, deviceMatrixB, deviceResult, matrixSize);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float timeFigure4_16;
        cudaEventElapsedTime(&timeFigure4_16, start, stop);

        // Ejecutar y medir tiempo para FIGURE 4.3
        cudaEventRecord(start);
        MatrixMulKernel_Figure4_3<<<numBlocks, threadsPerBlock>>>(deviceMatrixA, deviceMatrixB, deviceResult, matrixSize);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float timeFigure4_3;
        cudaEventElapsedTime(&timeFigure4_3, start, stop);

        // Copiar resultado del dispositivo al host
        cudaMemcpy(hostResult, deviceResult, matrixSize * matrixSize * sizeof(float), cudaMemcpyDeviceToHost);

        // Liberar memoria en el dispositivo
        cudaFree(deviceMatrixA);
        cudaFree(deviceMatrixB);
        cudaFree(deviceResult);

        // Calcular el tiempo de ejecución en segundos
        float timeFigure4_16_sec = timeFigure4_16 / 1000.0;
        float timeFigure4_3_sec = timeFigure4_3 / 1000.0;

        // Imprimir resultados
        std::cout << "Matrix Size: " << matrixSize << "x" << matrixSize << std::endl;
        std::cout << "Time (Figure 4.16): " << timeFigure4_16_sec << " seconds" << std::endl;
        std::cout << "Time (Figure 4.3): " << timeFigure4_3_sec << " seconds" << std::endl;
        std::cout << std::endl;

        // Liberar memoria en el host
        delete[] hostMatrixA;
        delete[] hostMatrixB;
        delete[] hostResult;
    }

    return 0;
}
