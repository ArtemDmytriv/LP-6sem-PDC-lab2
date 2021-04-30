#pragma once
#include <mpi.h>
#include <string>
#include <list>
#include <map>

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

const std::map<int, std::string> matrix_map {{1, "A"}, {2, "bi"}, {3, "A1"}, {4, "b1"}, {5, "c1"}, {6, "A2"}, {7, "B2"}, {8, "Cij"},
											 {9, "2b1 + 3c1"}, {10, "B2 - C2"}, {11, "y1"}, {12, "y2"}, {13, "Y3"},
											 {14, "y1y1'"}, {15, "y2y2'"}, {16, "Y3^2"}, {17, "y1y2'"}, {18, "Y3y2"},
											 {19, "y1y1'Y3"}, {20, "Y3^2 + y1y2'"}, {21, "y1y2'Y3y2"}, {22, "y1y1'Y3y2y2'"},
											 {23, "y1y2'Y3y2 + y1"}, {24, "y1y1'Y3y2y2' + Y3^2 + y1y2'"}, {25, "Res"}};


MPI_Datatype get_Job_struct_mpi();

Matrix generate_C_matrix(int n);
Matrix generate_b_vector(int n);

Job get_job_from_list_and_erase(std::list<Job> &lst, Matrix *mat, std::list<Job>::iterator &to_erase);
bool send_generate_job(int tag, int row, int col);
void send_res_generate_job(int tag);

void send_matrix_to_proc(int proc_rank, const Matrix *mat, int id, int to_proc_rank);
Matrix recv_mat_from_proc(int proc_rank, int mat_id);

void do_job(int proc_rank, Job job, Matrix *mat);

void logoutput(std::ostream &os, Matrix *mat, int n, const std::map<int, std::string> &mat_map);

