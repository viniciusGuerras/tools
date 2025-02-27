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

#include <iostream>

template <typename T> class vector{
    private:
        int* arr;
        unsigned capacity;
        unsigned current;

        void check_capacity(){
            if(current == capacity){
                upgrade_capacity();
            }
        }

        void check_capacity(int size){
            while(capacity < size){
                upgrade_capacity();
            }
        }

        void upgrade_capacity(){
            T* temp = new T[capacity * 2]; 
            for(int i = 0; i < capacity; i++){
                temp[i] = arr[i];
            }
            delete[] arr;
            capacity *= 2;
            arr = temp;
        }

    public:

        //initializes - destructs
        vector(){
            arr = new T[1];
            capacity = 1;
            current = 0;
        }

        vector(T* list, int size){
            arr = new T[size]; 
            capacity = size;
            current = size;
        
            check_capacity(size);
            for(int i =0;i<size;i++){
                force_push(list[i]);
            }
        }

        void zeores(int size){
            arr = new T[size]();
            capacity = size;
            current = size;
        }

        ~vector(){ 
            delete[] arr;   
        }

        //pushes
        void force_push(T data){
            arr[current]=data;
            current++;
        }

        void push(T data){
            check_capacity();
            arr[current]=data;
            current++;
        }

        void push(T data, int index){
            if(index == capacity){
                push(data);
            }
            else{
                arr[index] = data;
            }
        }

        void front_push(T data){
            check_capacity();

            for(int i=current;i>0;i--){
                arr[i] = arr[i-1];
            }
            arr[0]=data;
            current++;
        }

        T get(int index){
            if(index < current){
                return arr[index];
            }
            return -1;
        }

        void pop(){ current--; }
        int size(){ return current; }
        int getcapacity(){ return capacity; }

        int acummulate(int start, int end, int starter){
            if(end < current){
                for(int i=start;i<end;++){
                    starter *= arr[i];
                }
                return starter;
            }
            return -1;
        }

        void print(){
            for(int i=0;i<current;i++){
                std::cout<<arr[i]<<" ";
            }
            std::cout<<std::endl;
        }
};
