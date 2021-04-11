#pragma once
#include <mpi.h>
#include <string>
#include <list>

class Matrix;

struct Job {
	int id;
    enum class operation {
        GEN_MATRIX,         // requires size of matrix/vector in args:  arg[0]=rows, arg[1]=cols
        PLUS,               // requires id of Proc that contain matrix: arg[0]=op1, arg[1]=op2
        MINUS,
        MULT,               // requires id of Proc that contain matrix: arg[0]=op1, arg[1]=op2
        PLUS_MAT_TRANSPOSE, // requires
        PLUS_TRANSPOSE_MAT,
        MULT_MAT_TRANSPOSE,
        MULT_TRANSPOSE_MAT,
        PLUS_WITH_SCALAR,
        MULT_WITH_SCALAR,
        WAIT,
        END
    } op;
	int arg[2];
    double scalar[2];
    //Job(int id, int *arg, operation op) : id(id),
};


MPI_Datatype get_Job_struct_mpi();

Matrix generate_C_matrix(int n);
Matrix generate_b_vector(int n);

Job get_job_from_list_and_erase(std::list<Job> &lst, Matrix *mat, std::list<Job>::iterator &to_erase);
bool send_generate_job(int tag, int row, int col);
void send_res_generate_job(int tag);

void send_matrix_to_proc(const Matrix &op1, int proc_rank);
Matrix recv_mat_from_main(int proc_rank, int mat_id);

void do_job(int proc_rank, Job job, Matrix *mat);

