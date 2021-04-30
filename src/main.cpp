#include <mpi.h>

#include "foo.h"
#include "matrix.h"
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <map>
#include <algorithm>
#include <utility>
#include <chrono>

using namespace std::chrono;

const std::string separator_line{ "\n---------------------------------------\n" };
extern const int count_of_nodes;
extern const std::map<int, std::string> matrix_map;

int main(int argc, char* argv[]) {
	int n;
	int proc_num, proc_rank;
	MPI_Status status;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
	MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

	if (proc_num != 3) {
		std::cerr << "Need to start with 3 proc\n";
		exit(-1);
	}

	bool isRandGen = false;
	bool isDebugLog = false;
	if (proc_rank == 0) {
		std::cout << "n = ";
		std::cin >> n;
		if (argc == 1) {
			std::cout << "Generate random matrices '(y)es (n)o:  ";
			std::string ans;
			std::cin >> ans;
			std::for_each(ans.begin(), ans.end(), tolower);
			isRandGen = (ans == "yes" || ans == "y") ? true : false;
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&isRandGen, 1, MPI_INT, 0, MPI_COMM_WORLD);

	Matrix matrix[count_of_nodes];

	matrix[_A].set_shape({n, n});
	matrix[_bi].set_shape({n, 1});
	matrix[_A1].set_shape({n, n});
	matrix[_b1].set_shape({n, 1});
	matrix[_c1].set_shape({n, 1});
	matrix[_A2].set_shape({n, n});
	matrix[_B2].set_shape({n, n});
	matrix[_Cij].set_shape({n, n});

	if (proc_rank == 0) {
		if (!isRandGen) {
			std::cin >> matrix[_A]
				>> matrix[_A1]
				>> matrix[_b1]
				>> matrix[_c1]
				>> matrix[_A2]
				>> matrix[_B2];
			std::cout << "Manually set was saccessfull" << std::endl;
		}
	}

	if (!isRandGen) {
		MPI_Bcast(matrix[_A].data(), matrix[_A].get_col() * matrix[_A].get_row(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(matrix[_A1].data(), matrix[_A1].get_col() * matrix[_A1].get_row(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(matrix[_b1].data(), matrix[_b1].get_col() * matrix[_b1].get_row(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(matrix[_c1].data(), matrix[_c1].get_col() * matrix[_c1].get_row(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(matrix[_A2].data(), matrix[_A2].get_col() * matrix[_A2].get_row(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(matrix[_B2].data(), matrix[_B2].get_col() * matrix[_B2].get_row(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}
	// Start timer;
	auto start = std::chrono::high_resolution_clock::now();

	if (proc_rank == 0) { // main proc
		// All matrix
		MPI_Barrier(MPI_COMM_WORLD);

		if (isRandGen) {
			do_job(proc_rank, Job{_A,
					Job::operation::GEN_MATRIX, {n, n} }, matrix);
		}
		matrix[_bi] = generate_b_vector(n);

		do_job(proc_rank, Job{_y1,
				Job::operation::MULT, {_A, _bi} }, matrix);

		send_matrix_to_proc(proc_rank, matrix, _y1, 1);
		send_matrix_to_proc(proc_rank, matrix, _y1, 2);

		do_job(proc_rank, Job{_y1y1T,
				Job::operation::MULT_MAT_TRANSPOSE, {_y1, _y1} }, matrix);

		// recv Y3
		matrix[_Y3] = recv_mat_from_proc(proc_rank, _Y3);

		do_job(proc_rank, Job{_y1y1TY3,
				Job::operation::MULT, {_y1y1T, _Y3} }, matrix);

		// recv y2y2'
		matrix[_y2y2T] = recv_mat_from_proc(proc_rank, _y2y2T);

		do_job(proc_rank, Job{_y1y1TY3y2y2T,
				Job::operation::MULT, {_y1y1TY3, _y2y2T} }, matrix);

		// recv Y3^2 + y1y2'
		matrix[_Y3p2_plus_y1y2T] = recv_mat_from_proc(proc_rank, _Y3p2_plus_y1y2T);

		do_job(proc_rank, Job{_y1y1TY3y2y2T_plus_Y3p2_plus_y1y2T,
				Job::operation::PLUS, {_y1y1TY3y2y2T, _Y3p2_plus_y1y2T} }, matrix);

		// recv y1y2'Y3y2 + y1
		matrix[_y1y2TY3y2_plus_y1] = recv_mat_from_proc(proc_rank, _y1y2TY3y2_plus_y1);

		do_job(proc_rank, Job{_RES,
				Job::operation::MULT, {_y1y1TY3y2y2T_plus_Y3p2_plus_y1y2T, _y1y2TY3y2_plus_y1} }, matrix);

		MPI_Barrier(MPI_COMM_WORLD);
	}
	else if (proc_rank == 1) { // workers
		MPI_Barrier(MPI_COMM_WORLD);
		if (isRandGen) {
			do_job(proc_rank, Job{_A1,
					Job::operation::GEN_MATRIX, {n, n} }, matrix);
			do_job(proc_rank, Job{_b1,
					Job::operation::GEN_MATRIX, {n, 1} }, matrix);
			do_job(proc_rank, Job{_c1,
					Job::operation::GEN_MATRIX, {n, 1} }, matrix);
		}

		do_job(proc_rank, Job{_2b1_plus_3c1,
					Job::operation::PLUS_WITH_SCALAR, {_b1, _c1}, {2.0, 3.0} }, matrix);

		do_job(proc_rank, Job{_y2,
					Job::operation::MULT, {_A1, _2b1_plus_3c1} }, matrix);
		matrix[_y1] = recv_mat_from_proc(proc_rank, _y1);
		send_matrix_to_proc(proc_rank, matrix, _y2, 2);

		do_job(proc_rank, Job{_y1y2T,
					Job::operation::MULT_MAT_TRANSPOSE, {_y1, _y2} }, matrix);
		send_matrix_to_proc(proc_rank, matrix, _y1y2T, 2);

		do_job(proc_rank, Job{_y2y2T,
					Job::operation::MULT_MAT_TRANSPOSE, {_y2, _y2} }, matrix);
		send_matrix_to_proc(proc_rank, matrix, _y2y2T, 0);

		// recv Y3^2
		matrix[_Y3p2] = recv_mat_from_proc(proc_rank, _Y3p2);
		do_job(proc_rank, Job{_Y3p2_plus_y1y2T,
					Job::operation::PLUS, {_Y3p2, _y1y2T} }, matrix);
		send_matrix_to_proc(proc_rank, matrix, _Y3p2_plus_y1y2T, 0);

		MPI_Barrier(MPI_COMM_WORLD);
	}
	else if (proc_rank == 2) {
		MPI_Barrier(MPI_COMM_WORLD);
		if (isRandGen) {
			do_job(proc_rank, Job{_A2,
					Job::operation::GEN_MATRIX, {n, n} }, matrix);
			do_job(proc_rank, Job{_B2,
					Job::operation::GEN_MATRIX, {n, n} }, matrix);
		}
		matrix[_Cij] = generate_C_matrix(n);

		do_job(proc_rank, Job{_B2_minu_C2,
					Job::operation::MINUS, {_B2, _Cij} }, matrix);
		matrix[_y1] = recv_mat_from_proc(proc_rank, _y1);

		do_job(proc_rank, Job{_Y3,
					Job::operation::MULT, {_A2, _B2_minu_C2} }, matrix);
		matrix[_y2] = recv_mat_from_proc(proc_rank, _y2);

		send_matrix_to_proc(proc_rank, matrix, _Y3, 0);

		do_job(proc_rank, Job{_Y3p2,
					Job::operation::MULT, {_Y3, _Y3} }, matrix);

		matrix[_y1y2T] = recv_mat_from_proc(proc_rank, _y1y2T);
		send_matrix_to_proc(proc_rank, matrix, _Y3p2, 1);

		do_job(proc_rank, Job{_Y3y2,
					Job::operation::MULT, {_Y3, _y2} }, matrix);

		do_job(proc_rank, Job{_y1y2TY3y2,
					Job::operation::MULT, {_y1y2T, _Y3y2} }, matrix);


		do_job(proc_rank, Job{_y1y2TY3y2_plus_y1,
					Job::operation::PLUS, {_y1y2TY3y2, _y1} }, matrix);
		send_matrix_to_proc(proc_rank, matrix, _y1y2TY3y2_plus_y1, 0);
		MPI_Barrier(MPI_COMM_WORLD);
	}

	auto stop = std::chrono::high_resolution_clock::now();

	if (isRandGen) {
		MPI_Bcast(matrix[_A1].data(), matrix[_A1].get_col() * matrix[_A1].get_row(), MPI_DOUBLE, 1, MPI_COMM_WORLD);
		MPI_Bcast(matrix[_b1].data(), matrix[_b1].get_col() * matrix[_b1].get_row(), MPI_DOUBLE, 1, MPI_COMM_WORLD);
		MPI_Bcast(matrix[_c1].data(), matrix[_c1].get_col() * matrix[_c1].get_row(), MPI_DOUBLE, 1, MPI_COMM_WORLD);
		MPI_Bcast(matrix[_A2].data(), matrix[_A2].get_col() * matrix[_A2].get_row(), MPI_DOUBLE, 2, MPI_COMM_WORLD);
		MPI_Bcast(matrix[_B2].data(), matrix[_B2].get_col() * matrix[_B2].get_row(), MPI_DOUBLE, 2, MPI_COMM_WORLD);
		MPI_Bcast(matrix[_Cij].data(), matrix[_Cij].get_col() * matrix[_Cij].get_row(), MPI_DOUBLE, 2, MPI_COMM_WORLD);
	}

	if (proc_rank == 0) {
		// Timer stop
		auto duration = duration_cast<microseconds>(stop - start);

    	std::cout << "Time taken by program: " << duration.count() << " microseconds" << std::endl;
		std::cout << "Output input matrices for result checker" << std::endl;
		std::ofstream fout("matrices.txt", std::ios::trunc);
		fout << n << '\n'
			<< matrix[_A] // A
 			<< matrix[_A1] // A1
			<< matrix[_b1] // b1
			<< matrix[_c1] // c1
			<< matrix[_A2] // A2
			<< matrix[_B2]; // B2
		fout.close();
		std::cout << "Result=\n" << matrix[_RES];
		// log output, subresult

	}

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();

	if (proc_rank == 0) {
		start = std::chrono::high_resolution_clock::now();
		std::cout << "\nStart one thread" << std::endl;
		Matrix y1 = matrix[_A] * matrix[_bi];
		Matrix y2 = matrix[_A1] * (2.0 * matrix[_b1] + 3.0 * matrix[_c1]);
		Matrix Y3 = matrix[_A2] * (matrix[_B2] - matrix[_Cij]);

		Matrix one_thread_mat = (y1*y1.get_transpose()*Y3*y2*y2.get_transpose() + Y3*Y3 + y1*y2.get_transpose())*(y1*y2.get_transpose()*Y3*y2 + y1);

		stop = std::chrono::high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(stop - start);
		std::cout << "Check=\n" << one_thread_mat;
		std::cout << "Time taken by one thread program: "  << duration.count() << " microseconds" << std::endl;
	}
	return 0;
}
