// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code


#include "HIPStream.h"

[[noreturn]] inline void error(char const* file, int line, char const* expr, hipError_t e) {
  std::fprintf(stderr, "Error at %s:%d: %s (%d)\n  %s\n", file, line, hipGetErrorString(e), e, expr);
  exit(e);
}

// The do while is there to make sure you remember to put a semi-colon after calling HIP_CHECK
#define HIP_CHECK(EXPR) do { auto __e = (EXPR); if (__e != hipSuccess) error(__FILE__, __LINE__, #EXPR, __e); } while(false)

// It is best practice to include __device__ and constexpr even though in BabelStream it only needs to be __host__ const
__host__ __device__ constexpr size_t ceil_div(size_t a, size_t b) { return (a + b - 1)/b; }

hipStream_t stream;

template <class T>
HIPStream<T>::HIPStream(const intptr_t array_size, const int device_index)
  : array_size(array_size)
{
  // Set device
  int count;
  HIP_CHECK(hipGetDeviceCount(&count));
  if (device_index >= count)
    throw std::runtime_error("Invalid device index");
  HIP_CHECK(hipSetDevice(device_index));

  HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

  // Print out device information
  std::cout << "Using HIP device " << getDeviceName(device_index) << std::endl;
  std::cout << "Driver: " << getDeviceDriver(device_index) << std::endl;
#if defined(MANAGED)
  std::cout << "Memory: MANAGED" << std::endl;
#elif defined(PAGEFAULT)
  std::cout << "Memory: PAGEFAULT" << std::endl;
#else
  std::cout << "Memory: DEFAULT" << std::endl;
#endif

  // Query device for sensible dot kernel block count
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, device_index));
  dot_num_blocks = props.multiProcessorCount * 4;

  // Size of partial sums for dot kernels
  size_t sums_bytes = sizeof(T) * dot_num_blocks;
  size_t array_bytes = sizeof(T) * array_size;
  size_t total_bytes = array_bytes * size_t(3) + sums_bytes;
  std::cout << "Reduction kernel config: " << dot_num_blocks << " groups of (fixed) size " << TBSIZE << std::endl;

  // Check buffers fit on the device
  if (props.totalGlobalMem < total_bytes)
    throw std::runtime_error("Device does not have enough memory for all 3 buffers");

  // Create device buffers
#if defined(MANAGED)
  HIP_CHECK(hipMallocManaged(&d_a, array_bytes));
  HIP_CHECK(hipMallocManaged(&d_b, array_bytes));
  HIP_CHECK(hipMallocManaged(&d_c, array_bytes));
  HIP_CHECK(hipHostMalloc(&sums, sums_bytes, hipHostMallocDefault));
#elif defined(PAGEFAULT)
  d_a = (T*)malloc(array_bytes);
  d_b = (T*)malloc(array_bytes);
  d_c = (T*)malloc(array_bytes);
  sums = (T*)malloc(sums_bytes);
#else
  HIP_CHECK(hipMalloc(&d_a, array_bytes));
  HIP_CHECK(hipMalloc(&d_b, array_bytes));
  HIP_CHECK(hipMalloc(&d_c, array_bytes));
  HIP_CHECK(hipHostMalloc(&sums, sums_bytes, hipHostMallocDefault));
#endif
}


template <class T>
HIPStream<T>::~HIPStream()
{
  HIP_CHECK(hipStreamDestroy(stream));

#if defined(PAGEFAULT)
  free(d_a);
  free(d_b);
  free(d_c);
  free(sums);
#else
  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_c));
  HIP_CHECK(hipHostFree(sums));
#endif
}


template <typename T>
__global__ void init_kernel(T * a, T * b, T * c, T initA, T initB, T initC, size_t array_size)
{
  for (size_t i = (size_t)threadIdx.x + (size_t)blockDim.x * blockIdx.x; i < array_size; i += (size_t)gridDim.x * blockDim.x) {
    a[i] = initA;
    b[i] = initB;
    c[i] = initC;
  }
}

template <class T>
void HIPStream<T>::init_arrays(T initA, T initB, T initC)
{
  size_t blocks = ceil_div(array_size, TBSIZE);
  init_kernel<<<blocks, TBSIZE, 0, stream>>>(d_a, d_b, d_c, initA, initB, initC, array_size);
  HIP_CHECK(hipPeekAtLastError());
  HIP_CHECK(hipStreamSynchronize(stream));
}

template <class T>
void HIPStream<T>::read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c)
{

  // Copy device memory to host
#if defined(PAGEFAULT) || defined(MANAGED)
  HIP_CHECK(hipStreamSynchronize(stream));
  for (intptr_t i = 0; i < array_size; ++i)
  {
    a[i] = d_a[i];
    b[i] = d_b[i];
    c[i] = d_c[i];
  }
#else
  HIP_CHECK(hipMemcpy(a.data(), d_a, a.size()*sizeof(T), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(b.data(), d_b, b.size()*sizeof(T), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(c.data(), d_c, c.size()*sizeof(T), hipMemcpyDeviceToHost));
#endif
}

template <typename T>
__global__ void copy_kernel(const T * a, T * c, size_t array_size)
{
  for (size_t i = (size_t)threadIdx.x + (size_t)blockDim.x * blockIdx.x; i < array_size; i += (size_t)gridDim.x * blockDim.x) {
    c[i] = a[i];
  }
}

template <class T>
void HIPStream<T>::copy()
{
  size_t blocks = ceil_div(array_size, TBSIZE);
  copy_kernel<<<blocks, TBSIZE, 0, stream>>>(d_a, d_c, array_size);
  HIP_CHECK(hipPeekAtLastError());
  HIP_CHECK(hipStreamSynchronize(stream));
}

template <typename T>
__global__ void mul_kernel(T * b, const T * c, size_t array_size)
{
  const T scalar = startScalar;
  for (size_t i = (size_t)threadIdx.x + (size_t)blockDim.x * blockIdx.x; i < array_size; i += (size_t)gridDim.x * blockDim.x) {
    b[i] = scalar * c[i];
  }
}

template <class T>
void HIPStream<T>::mul()
{
  size_t blocks = ceil_div(array_size, TBSIZE);
  mul_kernel<<<blocks, TBSIZE, 0, stream>>>(d_b, d_c, array_size);
  HIP_CHECK(hipPeekAtLastError());
  HIP_CHECK(hipStreamSynchronize(stream));
}

template <typename T>
__global__ void add_kernel(const T * a, const T * b, T * c, size_t array_size)
{
  for (size_t i = (size_t)threadIdx.x + (size_t)blockDim.x * blockIdx.x; i < array_size; i += (size_t)gridDim.x * blockDim.x) {
    c[i] = a[i] + b[i];
  }
}

template <class T>
void HIPStream<T>::add()
{
  size_t blocks = ceil_div(array_size, TBSIZE);
  add_kernel<<<blocks, TBSIZE, 0, stream>>>(d_a, d_b, d_c, array_size);
  HIP_CHECK(hipPeekAtLastError());
  HIP_CHECK(hipStreamSynchronize(stream));
}

template <typename T>
__global__ void triad_kernel(T * a, const T * b, const T * c, size_t array_size)
{
  const T scalar = startScalar;
  for (size_t i = (size_t)threadIdx.x + (size_t)blockDim.x * blockIdx.x; i < array_size; i += (size_t)gridDim.x * blockDim.x) {
    a[i] = b[i] + scalar * c[i];
  }
}

template <class T>
void HIPStream<T>::triad()
{
  size_t blocks = ceil_div(array_size, TBSIZE);
  triad_kernel<<<blocks, TBSIZE, 0, stream>>>(d_a, d_b, d_c, array_size);
  HIP_CHECK(hipPeekAtLastError());
  HIP_CHECK(hipStreamSynchronize(stream));
}

template <typename T>
__global__ void nstream_kernel(T * a, const T * b, const T * c, size_t array_size)
{
  const T scalar = startScalar;
  for (size_t i = (size_t)threadIdx.x + (size_t)blockDim.x * blockIdx.x; i < array_size; i += (size_t)gridDim.x * blockDim.x) {
    a[i] += b[i] + scalar * c[i];
  }
}

template <class T>
void HIPStream<T>::nstream()
{
  size_t blocks = ceil_div(array_size, TBSIZE);
  nstream_kernel<<<blocks, TBSIZE, 0, stream>>>(d_a, d_b, d_c, array_size);
  HIP_CHECK(hipPeekAtLastError());
  HIP_CHECK(hipStreamSynchronize(stream));
}

template <class T>
__global__ void dot_kernel(const T * a, const T * b, T* sums, size_t array_size)
{
  __shared__ T smem[TBSIZE];
  T tmp = T(0.);
  const size_t tidx = threadIdx.x;
  for (size_t i = tidx + (size_t)blockDim.x * blockIdx.x; i < array_size; i += (size_t)gridDim.x * blockDim.x) {
    tmp += a[i] * b[i];
  }
  smem[tidx] = tmp;

  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    __syncthreads();
    if (tidx < offset) smem[tidx] += smem[tidx+offset];
  }

  // First thread writes to host memory directly from the device
  if (tidx == 0) sums[blockIdx.x] = smem[tidx];
}

template <class T>
T HIPStream<T>::dot()
{
  dot_kernel<<<dot_num_blocks, TBSIZE, 0, stream>>>(d_a, d_b, sums, array_size);
  HIP_CHECK(hipPeekAtLastError());
  HIP_CHECK(hipStreamSynchronize(stream));

  T sum = 0.0;
  for (intptr_t i = 0; i < dot_num_blocks; ++i) sum += sums[i];

  return sum;
}

void listDevices(void)
{
  // Get number of devices
  int count;
  HIP_CHECK(hipGetDeviceCount(&count));

  // Print device names
  if (count == 0)
  {
    std::cerr << "No devices found." << std::endl;
  }
  else
  {
    std::cout << std::endl;
    std::cout << "Devices:" << std::endl;
    for (int i = 0; i < count; i++)
    {
      std::cout << i << ": " << getDeviceName(i) << std::endl;
    }
    std::cout << std::endl;
  }
}


std::string getDeviceName(const int device)
{
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, device));
  return std::string(props.name);
}


std::string getDeviceDriver(const int device)
{
  HIP_CHECK(hipSetDevice(device));
  int driver;
  HIP_CHECK(hipDriverGetVersion(&driver));
  return std::to_string(driver);
}

template class HIPStream<float>;
template class HIPStream<double>;
