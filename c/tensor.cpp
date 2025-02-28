/*
Vector Library Requirements

A vector library deals with 1D arrays and should support fundamental operations efficiently.

1. Data Structure & Initialization
	•	Support for fixed-size and dynamic-size vectors.
	•	Efficient memory management (e.g., stack vs heap allocation).
	•	Ability to initialize:
	•	Zero vector (zero(n))
	•	Unit vector (unit(n, i), where i is the index of 1)
	•	Random vector
	•	From lists/arrays
	•	Copy constructor

2. Core Operations
	•	Arithmetic:
	•	Addition (+), Subtraction (-), Scalar Multiplication (*), Scalar Division (/)
	•	Hadamard (element-wise) product
	•	Vector Norms & Metrics:
	•	Euclidean norm (L2 norm)
	•	Manhattan norm (L1 norm)
	•	Max norm (L∞ norm)
	•	Dot product (A ⋅ B)
	•	Angle between two vectors
	•	Distance metrics (Euclidean, Cosine similarity, etc.)
	•	Projections & Orthogonality:
	•	Projection of A onto B
	•	Orthogonality check (A ⋅ B == 0)
	•	Linear Independence Check
	•	Sorting & Statistics:
	•	Min, Max, Sum, Mean, Variance
	•	Sorting (ascending/descending)
	•	Operations on Basis Vectors
	•	Support for Complex Numbers (Optional)

3. Utility Functions
	•	Pretty printing (toString(), repr())
	•	Type checking & dimension validation
	•	Serialization (JSON, Binary, CSV)
	•	Efficient I/O operations

4. Performance Considerations
	•	Optimized memory management
	•	SIMD acceleration (optional)
	•	Parallel computing support
*/
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
#include <iostream>
#include <random>

template <typename T> 
class vector {
    private:
        T* storage;
        unsigned capacity;
        unsigned current;

        void check_capacity() {
            if (current == capacity) {
                upgrade_capacity();
            }
        }

        void check_capacity(int size) {
            while (capacity < size) {
                upgrade_capacity();
            }
        }

        void upgrade_capacity() {
            T* temp = new T[capacity * 2]; 
            for (int i = 0; i < capacity; i++) {
                temp[i] = storage[i];
            }
            delete[] storage;
            capacity *= 2;
            storage = temp;
        }

    public:
        vector() {
            storage = new T[1];
            capacity = 1;
            current = 0;
        }

        vector(int size, T element) {
            check_capacity(size);
            storage = new T[size];
            for (int i = 0; i < size; i++) {
                storage[i] = element;
            }
            current = size;
        }

        vector(T* list, int size) {
            storage = new T[size]; 
            capacity = size;
            current = size;
            check_capacity(size);
            for (int i = 0; i < size; i++) {
                force_push(list[i]);
            }
        }

        ~vector() { 
            delete[] storage;   
        }

        void force_push(T data) {
            check_capacity();
            storage[current] = data;
            current++;
        }

        void push(T data) {
            check_capacity();
            storage[current] = data;
            current++;
        }

        void push(T data, int index) {
            if (index == capacity) {
                push(data);
            } else {
                storage[index] = data;
            }
        }

        void front_push(T data) {
            check_capacity();
            for (int i = current; i > 0; i--) {
                storage[i] = storage[i - 1];
            }
            storage[0] = data;
            current++;
        }

        T get(int index) {
            if (index < current) {
                return storage[index];
            }
            return -1;
        }

        void pop() { current--; }
        int size() { return current; }
        int getcapacity() { return capacity; }

        int accumulate(int start, int end, int starter) {
            if (end < current) {
                for (int i = start; i < end; i++) {
                    starter *= storage[i];
                }
                return starter;
            }
            return -1;
        }

        void print() {
            for (int i = 0; i < current; i++) {
                std::cout << storage[i] << " ";
            }
            std::cout << std::endl;
        }
};

template <typename T>
class ndArray {
    private:
        T* storage;
        vector<unsigned> dims;
        vector<unsigned> strides;
        unsigned capacity;

        void calculate_strides() {
            int size = dims.size();
            strides = vector<unsigned>(size, 0);
            strides.push(1, size - 1); // Last stride is 1

            for (int i = size - 2; i >= 0; --i) {
                strides.push(strides.get(i + 1) * dims.get(i + 1), i);
            }
        }

        void init_storage() {
            capacity = 1;
            for (int i = 0; i < strides.size(); i++) {
                capacity *= strides.get(i);
            }
            storage = new T[capacity];
        }

        void populate(int min, int max) {
            std::default_random_engine generator;
            std::uniform_int_distribution<T> distribution(min, max);
            
            for (unsigned i = 0; i < capacity; i++) {
                storage[i] = distribution(generator);
            }
        }

    public:
        ndArray(const vector<unsigned>& dims) : dims(dims) {
            calculate_strides();
            init_storage();
        }

        ndArray(int min, int max, const vector<unsigned>& dims) : dims(dims) {
            calculate_strides();
            init_storage();
            populate(min, max);
        }

        ~ndArray() {
            delete[] storage;
        }

        void print() {
            for (unsigned i = 0; i < capacity; i++) {
                std::cout << "[" << storage[i] << "] ";
            }
        }
};

int main() {
    unsigned arr[] = {3, 5, 2}; 
    vector<unsigned> dimensions = vector<unsigned>(arr, 3);  

    int min_value = 1;
    int max_value = 10;
    ndArray<int> array(min_value, max_value, dimensions);

    std::cout << "Generated ndArray:\n";
    array.print();

    return 0;
}