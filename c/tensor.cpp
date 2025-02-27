#include "vector.cpp"
/*
Tensor Library Requirements

A Tensor library extends vectors to higher dimensions (matrices and beyond).

1. Data Structure & Initialization
	•	1D, 2D, ND Tensor support
	•	Initialization:
	•	Zero Tensor, Identity matrix, Random Tensor
	•	From nested lists/arrays
	•	Copying & reshaping
	•	Efficient storage format (Row-major vs Column-major vs Sparse)

2. Core Operations
	•	Element-wise operations (+, -, *, /)
	•	Matrix multiplication (A @ B)
	•	Dot product, Cross product (3D only)
	•	Tensor contraction & Einstein summation (einsum)
	•	Kronecker product
	•	Outer product
	•	Tensor decomposition (LU, QR, SVD, etc.)
	•	Transpose, Inverse, Determinant
	•	Norms (Frobenius, Max, L1, L2, etc.)

3. Advanced Features
	•	Eigenvalues & Eigenvectors
	•	Cholesky decomposition
	•	Fast Fourier Transform (FFT)
	•	Sparse Tensor support

4. Performance & Optimization
	•	SIMD for faster operations
	•	Multi-threading & GPU acceleration (CUDA, OpenCL)
	•	Lazy evaluation for large Tensor computations

5. Utility Functions

	•	Pretty-print Tensors

	•	Efficient serialization (JSON, NumPy, etc.)
	•	Interfacing with other libraries (NumPy, PyTorch, etc.)

template <typename T> class Tensor{

    public:
        T* data;
        vector<int> dims;
        int capacity;
    
        Tensor(const vector<int>& dims) : dims(dims){
            for(int n_d:dims){
                capacity *= n_d;
            }
            data = new T[capacity];
        }

        ~Tensor(){
            delete[] data;
        }

        unsigned getdimensions(){
            return num_dims;
        }
        
       
};
*/


template <typename T>
class ndArray{
    private:
        T* storage;
        vector<unsigned> dims;
        vector<unsigned> strides;
        unsigned capacity;

        void calculate_strides() {
            int size = dims.size();
            strides = new vector.zeroes(size);
            std::cout << strides << std::endl;
            strides[size - 1] = 1; 

            for (int i = n-2; i >= 0; --i) {
                strides[i] = strides[i + 1] * dims[i + 1];
            }
        }
        void init_storage(){
            capacity = strides.acumulate(0, strides.size(), 1); 
            storage = new T[capacity];
        }

    public:
        
        ndArray(const vector<T>& dims) : dims_(dims){
            calculate_strides;
            init_storage

        }

};