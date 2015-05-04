
/* 
Based off work by Nelson, et al.
Brigham Young University (2010)

Adapted by Kevin Yuh (2015)
*/


#include <stdio.h>
#include <cuda.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cufft.h>

#define PI 3.14159265358979

texture<float, 2, cudaReadModeElementType> texreference;

/* Check errors on CUDA runtime functions */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}



/* Check errors on cuFFT functions */
void gpuFFTchk(int errval){
    if (errval != CUFFT_SUCCESS){
        printf("Failed FFT call, error code %d\n", errval);
    }
}



/* Check errors on CUDA kernel calls */
void checkCUDAKernelError()
{
    cudaError_t err = cudaGetLastError();
    if  (cudaSuccess != err){
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
    } else {
        fprintf(stderr, "No kernel error detected\n");
    }

}


/* Kernel for ramp filtering */
__global__
void
cudaScaleKernel(cufftComplex *sinogram_dev, int nAngles, int sinogram_width) {

    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < sinogram_width * nAngles) {
        // Scale highest frequencies by more
        sinogram_dev[index].x *= (1 - fabs((index % sinogram_width) * 2.0 
            / (sinogram_width - 1.0) - 1));
        sinogram_dev[index].y *= (1 - fabs((index % sinogram_width) * 2.0 
            / (sinogram_width - 1.0) - 1));

        index += blockDim.x * gridDim.x;
    }
}

/* Kernel for copying complex results to floats */
__global__
void
cudaMoveKernel(cufftComplex *sinogram_dev, float *sinogram_dev_float,
    int nAngles, int sinogram_width) {

    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < sinogram_width * nAngles) {
        sinogram_dev_float[index] = sinogram_dev[index].x;
        index += blockDim.x * gridDim.x;
    }
}

/* Kernel to perform backprojection */
__global__
void
cudaBackProjectKernel(float *sinogram_dev_float, int width, int height,
    int sinogram_width, float *dev_output, int nAngles) {

    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    float m, q, d, t, xi, yi, xo, yo;
    while (index < width * height) {
        xo = index % width - width / 2.0;
        yo = height / 2.0 - (index + 0.0) / width;
        dev_output[index] = 0;
        // Handle edge cases (otherwise will divide by zero)
        for (int i = 0; i < nAngles; i++) {
            t = (PI * i) / nAngles;
            if (i == 0)
                d = xo;
            else if (2 * i == nAngles)
                d = yo;
            else {  
	    m = 0 - cos(t) / sin(t);
	    q = -1.0 / m;
	    xi = (yo - m * xo) / (q - m);
	    yi = q * xi;
	    d = sqrt(xi * xi + yi * yi);
	    if ((q > 0 && xi < 0) || (q < 0 && xi > 0))
		d = 0 - d;
            }
            dev_output[index] += tex2D(texreference, i,
                (int) (d + sinogram_width / 2.0));
            //dev_output[index] += sinogram_dev_float[i * sinogram_width + 
            //    (int) (d + sinogram_width / 2.0)];
        }
        index += blockDim.x * gridDim.x;
    }
}
  



int main(int argc, char** argv){

    if (argc != 7){
        fprintf(stderr, "Incorrect number of arguments.\n\n");
        fprintf(stderr, "\nArguments: \n \
        < Sinogram filename > \n \
        < Width or height of original image, whichever is larger > \n \
        < Number of angles in sinogram >\n \
        < threads per block >\n \
        < number of blocks >\n \
        < output filename >\n");
        exit(EXIT_FAILURE);
    }


    /********** Parameters **********/

    int width = atoi(argv[2]);
    int height = width;
    int sinogram_width = (int)ceilf( height * sqrt(2) );

    int nAngles = atoi(argv[3]);


    int threadsPerBlock = atoi(argv[4]);
    int nBlocks = atoi(argv[5]);


    /********** Data storage *********/


    // GPU DATA STORAGE
    cufftComplex *dev_sinogram_cmplx;
    float *dev_sinogram_float; 
    float* output_dev;  // Image storage


    cufftComplex *sinogram_host;

    size_t size_result = width*height*sizeof(float);
    float *output_host = (float *)malloc(size_result);




    /*********** Set up IO, Read in data ************/

    sinogram_host = (cufftComplex *)malloc(  sinogram_width*nAngles*sizeof(cufftComplex) );

    FILE *dataFile = fopen(argv[1],"r");
    if (dataFile == NULL){
        fprintf(stderr, "Sinogram file missing\n");
        exit(EXIT_FAILURE);
    }

    FILE *outputFile = fopen(argv[6], "w");
    if (outputFile == NULL){
        fprintf(stderr, "Output file cannot be written\n");
        exit(EXIT_FAILURE);
    }

    int j, i;

    for(i = 0; i < nAngles * sinogram_width; i++){
        fscanf(dataFile,"%f",&sinogram_host[i].x);
        sinogram_host[i].y = 0;
    }

    fclose(dataFile);


    /*********** Assignment starts here *********/

    /* Allocate memory for all GPU storage above, copy input sinogram
    over to dev_sinogram_cmplx. */
    cudaMalloc(&dev_sinogram_cmplx, 
        nAngles * sinogram_width * sizeof(cufftComplex));
    cudaMalloc(&dev_sinogram_float, nAngles * sinogram_width * sizeof(float));
    cudaMalloc(&output_dev, width * height * sizeof(float));  // Image storage

    cudaMemcpy(dev_sinogram_cmplx, sinogram_host, 
        nAngles * sinogram_width * sizeof(cufftComplex), cudaMemcpyHostToDevice);
   

    /* The high-pass filter:
        - Use cuFFT for the forward FFT
        - Create your own kernel for the frequency scaling.
        - Use cuFFT for the inverse FFT
        - extract real components to floats
        - Free the original sinogram (dev_sinogram_cmplx)

        Note: If you want to deal with real-to-complex and complex-to-real
        transforms in cuFFT, you'll have to slightly change our code above.
    */
    cufftHandle plan;
    // Use a batched FFT to transform each sinogram at once
    cufftPlan1d(&plan, sinogram_width, CUFFT_C2C, nAngles);

    cufftExecC2C(plan, dev_sinogram_cmplx, dev_sinogram_cmplx, CUFFT_FORWARD);
    cudaScaleKernel<<<nBlocks, threadsPerBlock>>>
        (dev_sinogram_cmplx, nAngles, sinogram_width);
    
    checkCUDAKernelError();
    
    cufftExecC2C(plan, dev_sinogram_cmplx, dev_sinogram_cmplx, CUFFT_INVERSE);
    cufftDestroy(plan);
    
    cudaMoveKernel<<<nBlocks, threadsPerBlock>>>
    	(dev_sinogram_cmplx, dev_sinogram_float, nAngles, sinogram_width);
    checkCUDAKernelError();

    cudaArray* carray;
    cudaChannelFormatDesc channel;
    channel = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

    cudaMallocArray(&carray, &channel, sinogram_width, nAngles);
    cudaMemcpyToArray(carray, 0, 0, dev_sinogram_float, 
        nAngles * sinogram_width * sizeof(float), cudaMemcpyDeviceToDevice);

    texreference.filterMode = cudaFilterModeLinear;
    texreference.addressMode[0] = cudaAddressModeClamp;
    texreference.addressMode[1] = cudaAddressModeClamp;
    
    cudaBindTextureToArray(texreference, carray);
    

    /* Backprojection.
        - Create your own kernel to accelerate backprojection.
        - Copy the reconstructed image back to output_host.
        - Free all remaining memory on the GPU.
    */
    cudaBackProjectKernel<<<nBlocks, threadsPerBlock>>>
        (dev_sinogram_float, width, height, sinogram_width, 
            output_dev, nAngles);
    checkCUDAKernelError();
           
    cudaMemcpy(output_host, output_dev, width * height * sizeof(float),
        cudaMemcpyDeviceToHost);    
    
    cudaFree(dev_sinogram_cmplx);
    cudaFree(dev_sinogram_float);
    cudaFree(output_dev);    

    /* Export image data. */

    for(j = 0; j < width; j++){
        for(i = 0; i < height; i++){
            fprintf(outputFile, "%e ",output_host[j*width + i]);
        }
        fprintf(outputFile, "\n");
    }


    /* Cleanup: Free host memory, close files. */

    free(sinogram_host);
    free(output_host);

    fclose(outputFile);

    return 0;
}



