/*
Please download the model and tokenizer to the same folder:
$ wget -O model.bin https://huggingface.co/huangs0/llama2.c/resolve/main/model.bin
$ wget -O tokenizer.bin https://huggingface.co/huangs0/llama2.c/resolve/main/tokenizer.bin

In compile, remember to add `-pthred` to link library:
$ gcc -o accelerate_llama2 accelerate_llama2.c utilities.c -O2 -pthread -lm

Then Run with:
$ ./accelerate_llama2 <seed> <thr_count>
*/

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "utilities.h"

/**
 * ----------------------------------------------------------------------------
 * TASK - Optimize Matrix-Vector Multiplication by Multi-Threading
 * 
 * Matrix-Vector Multiplication, used in Attention and Feed-Forward Network
 * is the most time-consuming part of GPT. Luckily, most of computation is 
 * independent of each row, so we can use Multi-Threading for acceleration.
 * 
 * Please use <pthread.h> and your favorite synchronization method,
 * semaphore / mutex lock + conditional variable
 * 
 * A sequential version is provided in seq.c, please modify it to parallel version.
*/

// YOUR CODE STARTS HERE
// Addtional Header File Here
#include <pthread.h>
#include <semaphore.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <sys/resource.h>

// Global Variables
struct rusage main_usage;        // get usage for main thread
struct rusage thread_usage;      // get usage for child thread
int global_thr_count;
pthread_t *thread;
struct mat_vec {float* out; float* vec; float* mat; int col; int row;}; 
struct mat_vec* thread_args;
pthread_mutex_t lock;
pthread_cond_t canRun, canWrite;
int mul = 0;
int * thread_active;

void *thr_func(void *arg);

int init_mat_vec_mul(int thr_count) { //main thread start
    global_thr_count = thr_count;
    thread_args = malloc(sizeof(thread_args[0])*global_thr_count);
    thread = malloc(sizeof(pthread_t)*global_thr_count);
    thread_active = malloc(sizeof(int) *global_thr_count);
    pthread_mutex_init(&lock, NULL);
    pthread_cond_init(&canRun, NULL);
    pthread_cond_init(&canWrite, NULL);

    for (int i = 0; i < thr_count; i++){
        // Let threads identify themselves, i.e., each thread knows it is the i-th threads
        int * thread_id = malloc(sizeof(int));
        *thread_id = i;
        // Create n threads
        pthread_create(thread+i, NULL, &thr_func, thread_id);  
    }
    return 0;
}

void mat_vec_mul(float* out, float* vec, float* mat, int col, int row) {
    // Assign new parameters (out, vec, mat, col, row) to threads
    for (int i = 0; i < global_thr_count; i++){
        (thread_args + i) -> out = out;
        (thread_args + i) -> vec = vec;
        (thread_args + i) -> mat = mat;
        (thread_args + i) -> col = col;
        (thread_args + i) -> row = row;
    }
    // Wake up threads to do calculation
    pthread_mutex_lock(&lock);
    for (int i = 0; i < global_thr_count; i++){
        thread_active[i] = 1;
    }
    pthread_mutex_unlock(&lock);

    // Main thread wait until all threads finished task, and then return
    mul = 1;
    pthread_cond_broadcast(&canRun);

    for (int i = 0; i < global_thr_count; i++){
        pthread_mutex_lock(&lock);
        while (thread_active[i] != 0) {
            pthread_cond_wait(&canWrite, &lock);
        }
        pthread_mutex_unlock(&lock);
    }
}

int close_mat_vec_mul() {

    // Wake up threads to collect the system usage (of themselves) and terminates
    for (int i = 0; i < global_thr_count; i++){
        mul = -1;
        thread_active[i] = 1;
        pthread_cond_broadcast(&canRun);
        // Wait until all threads to exit and collect system usage of threads
        pthread_join(*(thread+i), NULL);
    }

    // Collect system usage of main thread, and display both usage of each thread and main thread
    getrusage(RUSAGE_SELF, &main_usage);
    float utime = main_usage.ru_utime.tv_sec + (float) main_usage.ru_utime.tv_usec / 1000000;
    float stime = main_usage.ru_stime.tv_sec + (float) main_usage.ru_stime.tv_usec / 1000000;
    printf("main thread - user: %.4f s, system: %.4f s\n", utime, stime);
    // Clear all resources related with multi-threading, and return
    free(thread);
    free(thread_args);
    free(thread_active);
    pthread_mutex_destroy(&lock);
    pthread_cond_destroy(&canRun);
    pthread_cond_destroy(&canWrite);
    return 0;
}

void *thr_func(void *arg) { // child thread
    int index = *(int*) arg;
    while(1){
        //  Fall asleep immediately after initialization
        pthread_mutex_lock(&lock);
        while (thread_active[index] != 1) {
            //  Can be woke up by main thread 
            pthread_cond_wait(&canRun, &lock);
        }
        pthread_mutex_unlock(&lock);
        
        // to work on assigned tasks
        if (mul > 0){
            struct mat_vec args = thread_args[index];
            for (int i = args.row * index/global_thr_count; i < args.row*(index+1)/global_thr_count; i++) {
                float val = 0.0f;
                for (int j = 0; j <args.col; j++) {
                    val += args.mat[i * args.col + j] * args.vec[j];
                }
                args.out[i] = val;
            }

            pthread_mutex_lock(&lock);
            thread_active[index] = 0;
            pthread_cond_signal(&canWrite);
            pthread_mutex_unlock(&lock);

        } else if (mul < 0){

            //  Being able to collet the system usage (of itself) and terminate
            getrusage(RUSAGE_THREAD, &thread_usage);
            float utime = thread_usage.ru_utime.tv_sec + (float) thread_usage.ru_utime.tv_usec / 1000000;
            float stime = thread_usage.ru_stime.tv_sec + (float) thread_usage.ru_stime.tv_usec / 1000000;
            printf("child thread: %d - user: %.4f s, system: %.4f s\n", index, utime, stime);
            //  After finishing the task, inform main thread
            pthread_exit(NULL);
            mul = 0;
            return 0;
            
        }
    }
}
// YOUR CODE ENDS HERE


int transformer(int token, int pos, LLMConfig* p, LLMRuntime* s, LLMWeight* w) {
    
    // a few convenience variables
    int dim = p->dim, hidden_dim =  p->hidden_dim, head_size = p->dim / p->n_heads;

    // copy the token embedding into x
    memcpy(s->x, &(w->token_embedding_table[token * dim]), dim*sizeof(float));

    // forward all the layers
    for(int l = 0; l < p->n_layers; l++) {

        // Attention
        {
            // attention normalization
            normalize(s->xb, s->x, w->rms_att_weight + l*dim, dim);

            // q, k, v = w_q @ x, w_k @ x, w_v @ x, respectively
            mat_vec_mul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
            mat_vec_mul(s->k, s->xb, w->wk + l*dim*dim, dim, dim);
            mat_vec_mul(s->v, s->xb, w->wv + l*dim*dim, dim, dim);

            // apply positional embedding
            position_embedding(s->q, s->k, w, pos, p->dim, p->n_heads);

            // save intermediate result for later reference
            key_value_cache(l, pos, p, s);
            
            // attention calculation
            attention(l, pos, p, s, w);

            // wo @ x to get final result
            mat_vec_mul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

            // residual connection back into x
            accum(s->x, s->xb2, dim);
        }
    
        // Feed-Forward Network: w2 @ (silu(w1 @ x) * (w3 @ x)), * is element-wise multiply
        {
            // FFN Normalization
            normalize(s->xb, s->x, w->rms_ffn_weight + l*dim, dim);

            // w1 @ x
            mat_vec_mul(s->h1, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
            // silu(w1 @ x)
            silu(s->h1, hidden_dim);
            // w3 @ x
            mat_vec_mul(s->h2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);
            // silu(w1 @ x) * (w3 @ x)
            element_wise_mul(s->h1, s->h2, hidden_dim);
            // w2 @ (silu(w1 @ x) * (w3 @ x))
            mat_vec_mul(s->xb, s->h1, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

            // residual connection
            accum(s->x, s->xb, dim);
        }
    }
    
    // final normalization
    normalize(s->x, s->x, w->rms_final_weight, dim);
    // classifier into logits
    mat_vec_mul(s->logits, s->x, w->token_embedding_table, p->dim, p->vocab_size);
    // apply the temperature to the logits
    for (int q=0; q<p->vocab_size; q++) { s->logits[q] /= 0.9f; }
    // apply softmax to the logits to get the probabilities for next token
    softmax(s->logits, p->vocab_size);
    // now sample from this distribution to get the next token
    return sample(s->logits, p->vocab_size);
}

int main(int argc, char* argv[]) {

    unsigned int seed;
    int thr_count;

    if (argc == 3) {
        seed = atoi(argv[1]);
        thr_count = atoi(argv[2]);
    } else {
        printf("Usage: ./compiled <seed> <thr_count>\n");
        return 1;
    }

    // Initialize
    srand(seed);
    init_mat_vec_mul(thr_count);

    // load model
    LLMConfig config;
    LLMWeight weights;
    if (load_LLM_Config_Weight(&config, &weights) == 1) { return 1; }

    // load tokenizer
    char** vocab = malloc(config.vocab_size * sizeof(char*));
    if (load_tokenizer(vocab, config.vocab_size) == 1) { return 1; }

    // create and init the application LLMRuntime
    LLMRuntime state;
    malloc_LLMRuntime(&state, &config);
    
    // the current position we are in
    long start = time_in_ms();

    int next, token = 1, pos = 0; // token = 1 -> <START>
    while (pos < config.seq_len) {

        // forward the transformer to get logits for the next token
        next = transformer(token, pos, &config, &state, &weights);

        printf("%s", vocab[next]);
        fflush(stdout); // force print

        token = next;
        pos++;
    }

    long end = time_in_ms();
    printf("\n\nlength: %d, time: %f s, achieved tok/s: %f\n", config.seq_len, (double)(end-start)/1000, config.seq_len / (double)(end-start)*1000);

    // cleanup
    close_mat_vec_mul();
    free_LLMRuntime(&state);
    free_LLMWeight(&weights);
    for (int i = 0; i < config.vocab_size; i++) { free(vocab[i]); }
    free(vocab);
    return 0;
}