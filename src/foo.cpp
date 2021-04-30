#include <mpi.h>

#include "foo.h"
#include "matrix.h"

MPI_Datatype get_Job_struct_mpi() {
    Job job_temp;
	MPI_Datatype job_type;

	int count = 4;
	int blocklen[] = { 1, 2, 2, 1};
	MPI_Aint disp[] = { offsetof(Job, id), offsetof(Job, arg), offsetof(Job, scalar), offsetof(Job, op) };
	MPI_Datatype types[] = {MPI_INT, MPI_INT, MPI_DOUBLE, MPI_INT};

	MPI_Type_create_struct(count, blocklen, disp, types, &job_type);
	MPI_Type_commit(&job_type);

    return job_type;
}

Job get_job_from_list_and_erase(std::list<Job> &lst, Matrix *mat, std::list<Job>::iterator &to_erase) {
	Job job {-1, Job::operation::WAIT };
	for (auto ptr = lst.begin(); ptr != lst.end(); ptr++) {
		if (ptr->op == Job::operation::GEN_MATRIX ||
		(mat[ptr->arg[0]].get_col() != 0 && mat[ptr->arg[1]].get_col() != 0)) {
			job = *ptr;
			to_erase = ptr;
			break;
		}
	}
	return job;
}

Matrix generate_C_matrix(int n) {
	Matrix C2(n, n);
	for (int i = 0; i < C2.get_row(); ++i) {
		for (int j = 0; j < C2.get_col(); ++j) {
			C2.at(i, j) = 1.0 / ((i+1) + (j+1) + 2); // count from 1
		}
	}
	return C2;
}

Matrix generate_b_vector(int n) {
	Matrix b(n, 1);
	for (int i = 0; i < b.get_row(); ++i) {
		b.at(i, 0) = 8.0/(i+1); // count from 1
	}
	return b;
}

void send_matrix_to_proc(int proc_rank, const Matrix* matrix, int id, int to_proc_rank) {
	std::pair<int, int> shape1 = matrix[id].get_shape();
	std::cout << "Proc: " << proc_rank << " -> " << to_proc_rank << " Send matrix [" << matrix_map.at(id) << "]"<<  std::endl;
	MPI_Send(&shape1.first, 2, MPI_INT, to_proc_rank, id, MPI_COMM_WORLD);
	MPI_Send(matrix[id].data(), matrix[id].get_col() * matrix[id].get_row(), MPI_DOUBLE, to_proc_rank, id, MPI_COMM_WORLD);
}

Matrix recv_mat_from_proc(int proc_rank, int mat_id) {
	Matrix mat;
	std::pair<int, int> shape;
	MPI_Status status;

	MPI_Recv(&shape.first, 2, MPI_INT, MPI_ANY_SOURCE, mat_id, MPI_COMM_WORLD, &status);
	mat.set_shape(shape);
	// recv matrix
	MPI_Recv(mat.data(), shape.first * shape.second, MPI_DOUBLE, MPI_ANY_SOURCE, mat_id, MPI_COMM_WORLD, &status);
	std::cout << "Proc: " << proc_rank << " <- " << status.MPI_SOURCE <<  " Recv end [" << matrix_map.at(mat_id) << "]" << std::endl;
	return mat;
}

void do_job(int proc_rank, Job job, Matrix *mat) {
	MPI_Status status;

	switch (job.op) {
	case Job::operation::GEN_MATRIX: {
		mat[job.id].set_shape({job.arg[0], job.arg[1]});
		mat[job.id].init_matrix();
		break;
	}
	case Job::operation::MINUS: {
		mat[job.id] = mat[job.arg[0]] - mat[job.arg[1]];
		break;
	}
	case Job::operation::MULT: {
		mat[job.id] = mat[job.arg[0]] * mat[job.arg[1]];
		break;
	}
	case Job::operation::MULT_MAT_TRANSPOSE: {
		mat[job.id] = M_x_Mt(mat[job.arg[0]], mat[job.arg[1]]);
		break;
	}
	case Job::operation::MULT_TRANSPOSE_MAT: {
		mat[job.id] = Mt_x_M(mat[job.arg[0]], mat[job.arg[1]]);
		break;
	}
	case Job::operation::PLUS: {
		mat[job.id] = mat[job.arg[0]] + mat[job.arg[1]];
		break;
	}
	case Job::operation::PLUS_MAT_TRANSPOSE: {
		mat[job.id] = M_plus_Mt(mat[job.arg[0]], mat[job.arg[1]]);
		break;
	}
	case Job::operation::PLUS_TRANSPOSE_MAT: {
		mat[job.id] = M_plus_Mt(mat[job.arg[1]], mat[job.arg[0]]);
		break;
	}
	case Job::operation::MULT_WITH_SCALAR: {
		mat[job.id] = (mat[job.arg[0]] * job.scalar[0]) * (mat[job.arg[1]] * job.scalar[1]);
		break;
	}
	case Job::operation::PLUS_WITH_SCALAR: {
		// std::cout << "Proc: " << proc_rank << " - plus matrices (with scalar) by id: " << job.arg[0] << "+" << job.arg[1] << std::endl;
		mat[job.id] = mat[job.arg[0]] * job.scalar[0] + mat[job.arg[1]] * job.scalar[1];
		break;
	}
	}
}


void logoutput(std::ostream &os, Matrix *mat, int n, const std::map<int, std::string> &mat_map) {
	for (int i = 1; i < n; ++i) {
		if (mat[i].get_row() != 0) {
			os << mat_map.at(i) << " =\n";
			os << mat[i];
			os << "--------------------------------------\n";
		}
	}
}