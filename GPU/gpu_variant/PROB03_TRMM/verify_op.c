#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define ERROR_THRESHOLD 1e-3

extern void COMPUTE_NAME_REF( int m0, int n0,
			      float *A_distributed,
			      float *B_distributed,
			      float *C_distributed );

extern void COMPUTE_NAME_TST( int m0, int n0,
			      float *A_distributed,
			      float *B_distributed,
			      float *C_distributed );


extern void DISTRIBUTED_ALLOCATE_NAME_REF( int m0, int n0,
					   float **A_distributed,
					   float **B_distributed,
					   float **C_distributed );

extern void DISTRIBUTED_ALLOCATE_NAME_TST( int m0, int n0,
					   float **A_distributed,
					   float **B_distributed,
					   float **C_distributed );



extern void DISTRIBUTE_DATA_NAME_REF( int m0, int n0,
				      float *A_sequential,
				      float *B_sequential,
				      float *A_distributed,
				      float *B_distributed );

extern void DISTRIBUTE_DATA_NAME_TST( int m0, int n0,
				      float *A_sequential,
				      float *B_sequential,
				      float *A_distributed,
				      float *B_distributed );




extern void COLLECT_DATA_NAME_REF( int m0, int n0,
				   float *C_distributed,
				   float *C_sequential );

extern void COLLECT_DATA_NAME_TST( int m0, int n0,
				   float *C_distributed,
				   float *C_sequential );




extern void DISTRIBUTED_FREE_NAME_REF( int m0, int n0,
				       float *A_distributed,
				       float *B_distributed,
				       float *C_distributed );

extern void DISTRIBUTED_FREE_NAME_TST( int m0, int n0,
				       float *A_distributed,
				       float *B_distributed,
				       float *C_distributed );




void fill_buffer_with_random( int num_elems, float *buff )
{
  //long long range = RAND_MAX;
  long long range = 1000;
  
  for(int i = 0; i < num_elems; ++i)
    {
      buff[i] = ((float)(rand()-((range)/2)))/((float)range);
    }
}

void fill_buffer_with_value( int num_elems, float val, float *buff )
{
  for(int i = 0; i < num_elems; ++i)
    buff[i] = val;
}


float max_pair_wise_diff(int m, int n, int rs, int cs, float *a, float *b)
{
  float max_diff = 0.0;

  for(int i = 0; i < m; ++i)
    for(int j = 0; j < n; ++j)
      {
	float sum  = fabs(a[i*rs+j*cs]+b[i*rs+j*cs]);
	float diff = fabs(a[i*rs+j*cs]-b[i*rs+j*cs]);


	float res = 0.0f;

	if(sum == 0.0f)
	  res = diff;
	else
	  res = 2*diff/sum;

	if( res > max_diff )
	  max_diff = res;

      }

  return max_diff;
}


int scale_p_on_pos_ret_v_on_neg(int p, int v)
{
  if (v < 1)
    return -1*v;
  else
    return v*p;
}

int main( int argc, char *argv[] )
{
  int rid;
  int num_ranks;
  int tag = 0;
  MPI_Status  status;
  int root_rid = 0;

  MPI_Init(&argc,&argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  // What we will output to
  FILE *result_file;
  
  // Problem parameters
  int min_size;
  int max_size;
  int step_size;

  int in_m0;
  int in_n0;

  // Get command line arguments
  if(argc == 1 )
    {
      min_size  = 16;
      max_size  = 256;
      step_size = 16;

      // defaults
      in_m0=1;
      in_n0=-3;

      // default to printing to stdout
      result_file = stdout;
    }
  else if(argc == 5 + 1 || argc == 6 + 1 )
    {
      min_size  = atoi(argv[1]);
      max_size  = atoi(argv[2]);
      step_size = atoi(argv[3]);

      in_m0=atoi(argv[4]);
      in_n0=atoi(argv[5]);

      // default to printing to stdout
      result_file = stdout;

      if(argc == 6 + 1)
	{
	  // we don't want every node opening the same file
	  // to write to.
	  if(rid == 0 )
	    {
	      result_file = fopen(argv[6],"w");
	    }
	  else
	    {
	      result_file = NULL;
	    }
	}
    }
  else
    {
      printf("usage: %s min max step m0 n0 [filename]\n",
	     argv[0]);
      exit(1);
    }

  // Print out the first line of the output in csv format
  if( rid == 0 )
    {
      /*root node */ 
      fprintf(result_file, "num_ranks,m0,n0,result\n");
    }
  else
    {/* all other nodes*/ }


  for( int p = min_size;
       p < max_size;
       p += step_size )
    {

      // input sizes
      int m0=scale_p_on_pos_ret_v_on_neg(p,in_m0);
      int n0=scale_p_on_pos_ret_v_on_neg(p,in_n0);

      // How big of a buffer do we need
      int A_sequential_sz =m0*m0;
      int C_sequential_sz =m0*n0;
      int B_sequential_sz =m0*n0;

      float *A_sequential_ref   = (float *)malloc(sizeof(float)*A_sequential_sz);
      float *C_sequential_ref  = (float *)malloc(sizeof(float)*C_sequential_sz);
      float *B_sequential_ref = (float *)malloc(sizeof(float)*B_sequential_sz);

      float *A_sequential_tst   = (float *)malloc(sizeof(float)*A_sequential_sz);
      float *C_sequential_tst  = (float *)malloc(sizeof(float)*C_sequential_sz);
      float *B_sequential_tst = (float *)malloc(sizeof(float)*B_sequential_sz);


      // We only want to allocate the buffers on every node, but
      // we don't want to fill them with random data on every node
      // just from the root node.

      if( rid == 0)
	{ /* root node */

	  // fill src_ref with random values
	  fill_buffer_with_random( A_sequential_sz, A_sequential_ref );
	  fill_buffer_with_random( B_sequential_sz, B_sequential_ref );
	  fill_buffer_with_value( C_sequential_sz, -1, C_sequential_ref );

     
	  // copy src_ref to src_tst
	  memcpy(A_sequential_tst,A_sequential_ref,A_sequential_sz*sizeof(float));
	  memcpy(B_sequential_tst,B_sequential_ref,B_sequential_sz*sizeof(float));
	  memcpy(C_sequential_tst,C_sequential_ref,C_sequential_sz*sizeof(float));
	}
      else
	{/* all other nodes. */}

      /*
	Run the reference
      */

      float *A_distributed_ref;
      float *B_distributed_ref;
      float *C_distributed_ref;

      // Allocate distributed buffers for the reference
      DISTRIBUTED_ALLOCATE_NAME_REF( m0, n0,
				     &A_distributed_ref,
				     &B_distributed_ref,
				     &C_distributed_ref );

      // Distribute the sequential buffers 
      DISTRIBUTE_DATA_NAME_REF( m0, n0,
				A_sequential_ref,
				B_sequential_ref,
				A_distributed_ref,
				B_distributed_ref );
     
      // Perform the computation
      COMPUTE_NAME_REF( m0, n0,
			A_distributed_ref,
			B_distributed_ref,
			C_distributed_ref );


      // Collect the distributed data and write it to a sequential buffer
      COLLECT_DATA_NAME_REF( m0, n0,
			     C_distributed_ref,
			     C_sequential_ref );     
     
      // Finally free the buffers
      DISTRIBUTED_FREE_NAME_REF( m0, n0,
				 A_distributed_ref,
				 B_distributed_ref,
				 C_distributed_ref );
     

      // run the test
      float *A_distributed_tst;
      float *B_distributed_tst;
      float *C_distributed_tst;

      // Allocate distributed buffers for the reference
      DISTRIBUTED_ALLOCATE_NAME_TST( m0, n0,
				     &A_distributed_tst,
				     &B_distributed_tst,
				     &C_distributed_tst );

      // Distribute the sequential buffers 
      DISTRIBUTE_DATA_NAME_TST( m0, n0,
				A_sequential_tst,
				B_sequential_tst,
				A_distributed_tst,
				B_distributed_tst );
     
      // Perform the computation
      COMPUTE_NAME_TST( m0, n0,
			A_distributed_tst,
			B_distributed_tst,
			C_distributed_tst );


      // Collect the distributed data and write it to a sequential buffer
      COLLECT_DATA_NAME_TST( m0, n0,
			     C_distributed_tst,
			     C_sequential_tst );     
     
      // Finally free the buffers
      DISTRIBUTED_FREE_NAME_TST( m0, n0,
				 A_distributed_tst,
				 B_distributed_tst,
				 C_distributed_tst );


      // We only need to verify the results sequentially
      if( rid == 0)
	{
	  /* root node */
	  
	  float res = max_pair_wise_diff(m0,n0,1,m0, C_sequential_ref, C_sequential_tst);
	  
	  fprintf(result_file, "%i,%i,%i,",
		  num_ranks,
		  m0,n0);
	  
	  // if our error is greater than some threshold
	  if( res > ERROR_THRESHOLD )
	    fprintf(result_file, "FAIL\n");
	  else
	    fprintf(result_file, "PASS\n");
	}
      else
	{/* all other nodes */}

      // Free the sequential buffers
      free(A_sequential_ref);
      free(C_sequential_ref);
      free(B_sequential_ref);
      free(A_sequential_tst);
      free(C_sequential_tst);
      free(B_sequential_tst);

    }


  // Only needs to be done by root node
  if(rid == 0)
    {
      /* root node */
      fclose(result_file);
    }
  else
    {/* all other nodes */}
     

  
 MPI_Finalize();
}
