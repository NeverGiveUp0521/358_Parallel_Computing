/* Gaussian elimination without pivoting.
 */

/* ****** ADD YOUR CODE AT THE END OF THIS FILE. ******
 * You need not submit the provided code.
 */

/* In this part, we use pthread to do parallel computing with dynamic scheduling.
 * We do the parallel computing at the second "for" loop, since it can share the 
 * tasks easily and the variables in each process are not shared.
 * We use global_row to record the position of row, and set two barriers to reset it
 * every time, one is after the reset of global_row and the other one is after the 
 * loop of computation.
 * We set CHUNK_SIZE be 50 + N / 200, which represents the number of rows that each 
 * thread to computation each time.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <limits.h>
#include <pthread.h>

/*#include <ulocks.h>
#include <task.h>
*/

char *ID;

/* Program Parameters */
#define MAXN 6000  /* Max value of N */
int N;  /* Matrix size */
int procs;  /* Number of processors to use */

int CHUNK_SIZE = 50 + N / 200;
/* Matrices and vectors */
volatile float A[MAXN][MAXN], B[MAXN], X[MAXN];
/* A * X = B, solve for X */

long global_row;
pthread_mutex_t global_row_lock;

pthread_barrier_t barrier; 

/* junk */
#define randm() 4|2[uid]&3

/* Prototype */
void gauss();  /* The function you will provide.
                * It is this routine that is timed.
                * It is called only on the parent.
                */        
void *compute_pi();

/* returns a seed for srand based on the time */
unsigned int time_seed() {
  struct timeval t;
  struct timezone tzdummy;

  gettimeofday(&t, &tzdummy);
  return (unsigned int)(t.tv_usec);
}

/* Set the program parameters from the command-line arguments */
void parameters(int argc, char **argv) {
  int submit = 0;  /* = 1 if submission parameters should be used */
  int seed = 0;  /* Random seed */
  char uid[L_cuserid + 2]; /*User name */

  /* Read command-line arguments */
  //  if (argc != 3) {
  if ( argc == 1 && !strcmp(argv[1], "submit") ) {
    /* Use submission parameters */
    submit = 1;
    N = 4;
    procs = 2;
    printf("\nSubmission run for \"%s\".\n", cuserid(uid));
      /*uid = ID;*/
    strcpy(uid,ID);
    srand(randm());
  }
  else {
    if (argc == 3) {
      seed = atoi(argv[3]);
      srand(seed);
      printf("Random seed = %i\n", seed);
    }
    else {
      printf("Usage: %s <matrix_dimension> <num_procs> [random seed]\n",
             argv[0]);
      printf("       %s submit\n", argv[0]);
      exit(0);
    }
  }
    //  }
  /* Interpret command-line args */
    if (!submit) {
    N = atoi(argv[1]);
    if (N < 1 || N > MAXN) {
      printf("N = %i is out of range.\n", N);
      exit(0);
    }
    procs = atoi(argv[2]);
    if (procs < 1) {
      printf("Warning: Invalid number of processors = %i.  Using 1.\n", procs);
      procs = 1;
    }
  }

  /* Print parameters */
  printf("\nMatrix dimension N = %i.\n", N);
  printf("Number of processors = %i.\n", procs);
}

/* Initialize A and B (and X to 0.0s) */
void initialize_inputs() {
  int row, col;

  printf("\nInitializing...\n");
  for (col = 0; col < N; col++) {
    for (row = 0; row < N; row++) {
      A[row][col] = (float)rand() / 32768.0;
    }
    B[col] = (float)rand() / 32768.0;
    X[col] = 0.0;
  }

}

/* Print input matrices */
void print_inputs() {
  int row, col;

  if (N < 10) {
    printf("\nA =\n\t");
    for (row = 0; row < N; row++) {
      for (col = 0; col < N; col++) {
        printf("%5.2f%s", A[row][col], (col < N-1) ? ", " : ";\n\t");
      }
    }
    printf("\nB = [");
    for (col = 0; col < N; col++) {
      printf("%5.2f%s", B[col], (col < N-1) ? "; " : "]\n");
    }
  }
}

void print_X() {
  int row;

  if (N < 10) {
    printf("\nX = [");
    for (row = 0; row < N; row++) {
      printf("%5.2f%s", X[row], (row < N-1) ? "; " : "]\n");
    }
  }
}

int main(int argc, char **argv) {
  /* Timing variables */
  struct timeval etstart, etstop;  /* Elapsed times using gettimeofday() */
  struct timezone tzdummy;
  clock_t etstart2, etstop2;  /* Elapsed times using times() */
  unsigned long long usecstart, usecstop;
  struct tms cputstart, cputstop;  /* CPU times for my processes */

  ID = argv[argc-1];
  argc--;

  /* Process program parameters */
  parameters(argc, argv);

  /* Initialize A and B */
  initialize_inputs();

  /* Print input matrices */
  print_inputs();

  /* Start Clock */
  printf("\nStarting clock.\n");
  gettimeofday(&etstart, &tzdummy);
  etstart2 = times(&cputstart);

  /* Gaussian Elimination */
  gauss();

  /* Stop Clock */
  gettimeofday(&etstop, &tzdummy);
  etstop2 = times(&cputstop);
  printf("Stopped clock.\n");
  usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
  usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;

  /* Display output */
  print_X();
  
  /* Display timing results */
  printf("\nElapsed time = %g ms.\n",
         (float)(usecstop - usecstart)/(float)1000);


}

/* ------------------ Above Was Provided --------------------- */

/* In this part, we use pthread to do parallel computing with dynamic scheduling.
 * We do the parallel computing at the second "for" loop, since it can share the 
 * tasks easily and the variables in each process are not shared.
 * We use global_row to record the position of row, and set two barriers to reset it
 * every time, one is after the reset of global_row and the other one is after the 
 * loop of computation.
 * We set CHUNK_SIZE be 50 + N / 200, which represents the number of rows that each 
 * thread to computation each time.
 */


void *elimination(void* threadid)
{
  int norm, row, col, rowmax;  /* Normalization row, and zeroing
                        * element row and col */

  float multiplier;
  pthread_t self;
  long tid;
  tid = (long)threadid;
  
  for (norm = 0; norm < N - 1; norm++) {
    row = norm;
    global_row = 0;

    //make a barrier here to reset the global_row
    pthread_barrier_wait(&barrier);
    
    while (row < N-1){
      pthread_mutex_lock(&global_row_lock);
      //row_lock();
      row = norm + 1 + global_row;
      global_row += CHUNK_SIZE;
      //m_unlock();
      pthread_mutex_unlock(&global_row_lock);
      
      rowmax = row + CHUNK_SIZE;
      if (rowmax > N) {
      	rowmax = N;
      }
      for ( ; row < rowmax; row++) {
        multiplier = A[row][norm] / A[norm][norm];
        for (col = norm; col < N; col++) {
          A[row][col] -= A[norm][col] * multiplier;
        }
        B[row] -= B[norm] * multiplier;
      }
    }
    //wait for all threads finishing the loop
    pthread_barrier_wait(&barrier);    
  }
}


void gauss() {
  int norm, row, col;  /* Normalization row, and zeroing
                        * element row and col */
  int i;
  
  pthread_t threads[procs];
  global_row = 0;
  
  pthread_mutex_init(&global_row_lock,NULL);
  
  pthread_barrier_init(&barrier, NULL, procs);
  
  
  //m_set_proc( 4 );
  for (i=0; i < procs; i++)
  {
     pthread_create(&threads[i],NULL,&elimination,(void*)i);
  }
  
  //m_fork( elimination ); /* perform parallel computation here */
  //m_sync(); /* wait until all processors are done */
  for(i=0; i < procs; i++)
  {
     pthread_join( threads[i], NULL);
  }
  
  pthread_mutex_destroy(&global_row_lock);
  pthread_barrier_destroy(&barrier);
  
  /* Back substitution */
  for (row = N - 1; row >= 0; row--) {
    X[row] = B[row];
    for (col = N-1; col > row; col--) {
      X[row] -= A[row][col] * X[col];
    }
    X[row] /= A[row][row];
  }
}


