# Accelerate-LLM-Inference

This is the second assignment of my operating system course in the university. This assignment provides a complete pure C implementation of its inference in seq.c as the baseline model which is based on the open-source project llama2.c by Andrej Karpathy, an open-source variation of GPT, llama2 released by Meta. Similar to other neural networks, GPT and its variations utilize matrix-vector-multiplication, or called fully- connected/linear layer in DL, to apply the parameter learned, which takes >70% of the whole calculation. Thus, to accelerate the GPT and get faster response, it’s critical to have faster matrix- vector-multiplication, and multi-threading are usually considered powerful.

### Mutex Lock + Conditional Variable
To optimize the matrix-vector-multiplication algorithm of GPT by multi-threading, I've used pthread.h with mutex_lock + conditional variable to implement a multi-threading version of matrix-vector-multiplication. 

### Parallel Checking on Matrix-Vector-Multiplication
To parallelize the outer iteration by allocating rows to threads, in the case of a Matrix with 𝑑 rows and 𝑛 threads working on the computation, if 𝑑 is divisible by 𝑛, the k-th thread (𝑘 = 0, 1, ... , 𝑛 − 1) will handle the rows from [𝑘 × 𝑑/𝑛] to [(𝑘 + 1) × 𝑑/𝑛 − 1]. If 𝑑 is not divisible by 𝑛, we can assign first 𝑛 − 1 threads (𝑘 = 0, 1, ... , 𝑛 − 2) with [d/n]. rows, while the last thread handles remaining rows. In order to reduce overhead, you are required to create one set of threads and reuse them for all mat_vec_mul() function calls, instead of creating threads for each mat_vec_mul()function call. 

### Synchronization Workflow (My Tasks)
1. Create n thread in create_mat_vec_mul(intthr_count) at the beginning of program and put them sleep immediately after initialization 
2. Expose API to do Matrix-Vector-Multiplication by assigninhg new parameters (out, vec, mat, col, row) to threads, then wake up threads to do calculation, and let main thread wait until all threads finished task
3. Clear all resources related with multi-threading at the end of program in destroy_mat_vec_mul(). All threads are waken up to collect the system usage (of themselves) and terminates, the program will wait until all threads to exit and collect system usage of threads. After the system usage of main thread is collected, both usage of each thread and main thread will be diaplayed.
4. Do Matrix-Vector-Multiplication in void*thr_func(void*arg). The thread function is put to sleep immediately after initialization that can be woke up by main thread to work on assigned tasks. After finishing the task, the thread function will inform main thread, collet the system usage (of itself) and terminate. 


## Result & Finding
This multi-threading version has significantly accelerated the inference of Large Language Model and the result is provided in the report.txt for analysis.

