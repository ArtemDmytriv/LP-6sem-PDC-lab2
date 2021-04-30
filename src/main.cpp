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
const int count_of_nodes = 26;
extern const std::map<int, std::string> matrix_map;

int main(int argc, char* argv[]) {
	int n;
	int proc_num, proc_rank;
	MPI_Status status;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
	MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);


	std::map <std::string, int> m;
	for (auto p = matrix_map.cbegin(); p != matrix_map.cend(); ++p) {
		m[p->second] = p->first;
	}

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

	matrix[m["A"]].set_shape({n, n});
	matrix[m["bi"]].set_shape({n, 1});
	matrix[m["A1"]].set_shape({n, n});
	matrix[m["b1"]].set_shape({n, 1});
	matrix[m["c1"]].set_shape({n, 1});
	matrix[m["A2"]].set_shape({n, n});
	matrix[m["B2"]].set_shape({n, n});
	matrix[m["Cij"]].set_shape({n, n});

	if (proc_rank == 0) {
		if (!isRandGen) {
			std::cin >> matrix[m["A"]]
				>> matrix[m["A1"]]
				>> matrix[m["b1"]]
				>> matrix[m["c1"]]
				>> matrix[m["A2"]]
				>> matrix[m["B2"]];
			std::cout << "Manually set was saccessfull" << std::endl;
		}
	}

	if (!isRandGen) {
		MPI_Bcast(matrix[m["A"]].data(), matrix[m["A"]].get_col() * matrix[m["A"]].get_row(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(matrix[m["A1"]].data(), matrix[m["A1"]].get_col() * matrix[m["A1"]].get_row(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(matrix[m["b1"]].data(), matrix[m["b1"]].get_col() * matrix[m["b1"]].get_row(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(matrix[m["c1"]].data(), matrix[m["c1"]].get_col() * matrix[m["c1"]].get_row(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(matrix[m["A2"]].data(), matrix[m["A2"]].get_col() * matrix[m["A2"]].get_row(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(matrix[m["B2"]].data(), matrix[m["B2"]].get_col() * matrix[m["B2"]].get_row(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}
	// Start timer;
	auto start = std::chrono::high_resolution_clock::now();

	if (proc_rank == 0) { // main proc
		// All matrix
		MPI_Barrier(MPI_COMM_WORLD);

		if (isRandGen) {
			do_job(proc_rank, Job{m["A"],
					Job::operation::GEN_MATRIX, {n, n} }, matrix);
		}
		matrix[m["bi"]] = generate_b_vector(n);

		do_job(proc_rank, Job{m["y1"],
				Job::operation::MULT, {m["A"], m["bi"]} }, matrix);
		send_matrix_to_proc(proc_rank, matrix, m["y1"], 1);
		send_matrix_to_proc(proc_rank, matrix, m["y1"], 2);

		do_job(proc_rank, Job{m["y1y1'"],
				Job::operation::MULT_MAT_TRANSPOSE, {m["y1"], m["y1"]} }, matrix);

		// recv Y3
		matrix[m["Y3"]] = recv_mat_from_proc(proc_rank, m["Y3"]);

		do_job(proc_rank, Job{m["y1y1'Y3"],
				Job::operation::MULT, {m["y1y1'"], m["Y3"]} }, matrix);

		// recv y2y2'
		matrix[m["y2y2'"]] = recv_mat_from_proc(proc_rank, m["y2y2'"]);

		do_job(proc_rank, Job{m["y1y1'Y3y2y2'"],
				Job::operation::MULT, {m["y1y1'Y3"], m["y2y2'"]} }, matrix);

		// recv Y3^2 + y1y2'
		matrix[m["Y3^2 + y1y2'"]] = recv_mat_from_proc(proc_rank, m["Y3^2 + y1y2'"]);

		do_job(proc_rank, Job{m["y1y1'Y3y2y2' + Y3^2 + y1y2'"],
				Job::operation::PLUS, {m["y1y1'Y3y2y2'"], m["Y3^2 + y1y2'"]} }, matrix);

		// recv y1y2'Y3y2 + y1
		matrix[m["y1y2'Y3y2 + y1"]] = recv_mat_from_proc(proc_rank, m["y1y2'Y3y2 + y1"]);

		do_job(proc_rank, Job{m["Res"],
				Job::operation::MULT, {m["y1y1'Y3y2y2' + Y3^2 + y1y2'"], m["y1y2'Y3y2 + y1"]} }, matrix);

		MPI_Barrier(MPI_COMM_WORLD);
	}
	else if (proc_rank == 1) { // workers
		MPI_Barrier(MPI_COMM_WORLD);
		if (isRandGen) {
			do_job(proc_rank, Job{m["A1"],
					Job::operation::GEN_MATRIX, {n, n} }, matrix);
			do_job(proc_rank, Job{m["b1"],
					Job::operation::GEN_MATRIX, {n, 1} }, matrix);
			do_job(proc_rank, Job{m["c1"],
					Job::operation::GEN_MATRIX, {n, 1} }, matrix);
		}

		do_job(proc_rank, Job{m["2b1 + 3c1"],
					Job::operation::PLUS_WITH_SCALAR, {m["b1"], m["c1"]}, {2.0, 3.0} }, matrix);

		do_job(proc_rank, Job{m["y2"],
					Job::operation::MULT, {m["A1"], m["2b1 + 3c1"]} }, matrix);
		matrix[m["y1"]] = recv_mat_from_proc(proc_rank, m["y1"]);
		send_matrix_to_proc(proc_rank, matrix, m["y2"], 2);

		do_job(proc_rank, Job{m["y1y2'"],
					Job::operation::MULT_MAT_TRANSPOSE, {m["y1"], m["y2"]} }, matrix);
		send_matrix_to_proc(proc_rank, matrix, m["y1y2'"], 2);

		do_job(proc_rank, Job{m["y2y2'"],
					Job::operation::MULT_MAT_TRANSPOSE, {m["y2"], m["y2"]} }, matrix);
		send_matrix_to_proc(proc_rank, matrix, m["y2y2'"], 0);

		// recv Y3^2
		matrix[m["Y3^2"]] = recv_mat_from_proc(proc_rank, m["Y3^2"]);
		do_job(proc_rank, Job{m["Y3^2 + y1y2'"],
					Job::operation::PLUS, {m["Y3^2"], m["y1y2'"]} }, matrix);
		send_matrix_to_proc(proc_rank, matrix, m["Y3^2 + y1y2'"], 0);

		MPI_Barrier(MPI_COMM_WORLD);
	}
	else if (proc_rank == 2) {
		MPI_Barrier(MPI_COMM_WORLD);
		if (isRandGen) {
			do_job(proc_rank, Job{m["A2"],
					Job::operation::GEN_MATRIX, {n, n} }, matrix);
			do_job(proc_rank, Job{m["B2"],
					Job::operation::GEN_MATRIX, {n, n} }, matrix);
		}
		matrix[m["Cij"]] = generate_C_matrix(n);

		do_job(proc_rank, Job{m["B2 - Cij"],
					Job::operation::MINUS, {m["B2"], m["Cij"]} }, matrix);
		matrix[m["y1"]] = recv_mat_from_proc(proc_rank, m["y1"]);

		do_job(proc_rank, Job{m["Y3"],
					Job::operation::MULT, {m["A2"], m["B2 - Cij"]} }, matrix);
		matrix[m["y2"]] = recv_mat_from_proc(proc_rank, m["y2"]);

		send_matrix_to_proc(proc_rank, matrix, m["Y3"], 0);

		do_job(proc_rank, Job{m["Y3^2"],
					Job::operation::MULT, {m["Y3"], m["Y3"]} }, matrix);

		matrix[m["y1y2'"]] = recv_mat_from_proc(proc_rank, m["y1y2'"]);
		send_matrix_to_proc(proc_rank, matrix, m["Y3^2"], 1);

		do_job(proc_rank, Job{m["Y3y2"],
					Job::operation::MULT, {m["Y3"], m["y2"]} }, matrix);

		do_job(proc_rank, Job{m["y1y2'Y3y2"],
					Job::operation::MULT, {m["y1y2'"], m["Y3y2"]} }, matrix);


		do_job(proc_rank, Job{m["y1y2'Y3y2 + y1"],
					Job::operation::PLUS, {m["y1y2'Y3y2"], m["y1"]} }, matrix);
		send_matrix_to_proc(proc_rank, matrix, m["y1y2'Y3y2 + y1"], 0);
		MPI_Barrier(MPI_COMM_WORLD);
	}

	auto stop = std::chrono::high_resolution_clock::now();

	if (isRandGen) {
		MPI_Bcast(matrix[m["A1"]].data(), matrix[m["A1"]].get_col() * matrix[m["A1"]].get_row(), MPI_DOUBLE, 1, MPI_COMM_WORLD);
		MPI_Bcast(matrix[m["b1"]].data(), matrix[m["b1"]].get_col() * matrix[m["b1"]].get_row(), MPI_DOUBLE, 1, MPI_COMM_WORLD);
		MPI_Bcast(matrix[m["c1"]].data(), matrix[m["c1"]].get_col() * matrix[m["c1"]].get_row(), MPI_DOUBLE, 1, MPI_COMM_WORLD);
		MPI_Bcast(matrix[m["A2"]].data(), matrix[m["A2"]].get_col() * matrix[m["A2"]].get_row(), MPI_DOUBLE, 2, MPI_COMM_WORLD);
		MPI_Bcast(matrix[m["B2"]].data(), matrix[m["B2"]].get_col() * matrix[m["B2"]].get_row(), MPI_DOUBLE, 2, MPI_COMM_WORLD);
		MPI_Bcast(matrix[m["Cij"]].data(), matrix[m["Cij"]].get_col() * matrix[m["Cij"]].get_row(), MPI_DOUBLE, 2, MPI_COMM_WORLD);
	}

	if (proc_rank == 0) {
		// Timer stop
		auto duration = duration_cast<microseconds>(stop - start);

    	std::cout << "Time taken by program: " << duration.count() << " microseconds" << std::endl;
		std::cout << "Output input matrices for result checker" << std::endl;
		std::ofstream fout("matrices.txt", std::ios::trunc);
		fout << n << '\n'
			<< matrix[1] // A
 			<< matrix[3] // A1
			<< matrix[4] // b1
			<< matrix[5] // c1
			<< matrix[6] // A2
			<< matrix[7]; // B2
		fout.close();
		std::cout << "Result=\n" << matrix[m["Res"]];
	}

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();

	if (proc_rank == 0) {
		start = std::chrono::high_resolution_clock::now();
		std::cout << "\nStart one thread" << std::endl;
		Matrix y1 = matrix[m["A"]] * matrix[m["bi"]];
		//std::cout << y1;
		Matrix y2 = matrix[3] * (2.0 * matrix[4] + 3.0 * matrix[5]);
		//std::cout << matrix[m["Cij"]];
		Matrix Y3 = matrix[m["A2"]] * (matrix[m["B2"]] - matrix[m["Cij"]]);
		//std::cout << Y3;

		Matrix one_thread_mat = (y1*y1.get_transpose()*Y3*y2*y2.get_transpose() + Y3*Y3 + y1*y2.get_transpose())*(y1*y2.get_transpose()*Y3*y2 + y1);

		stop = std::chrono::high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(stop - start);
		std::cout << "Check=\n" << one_thread_mat;
		std::cout << "Time taken by one thread program: "  << duration.count() << " microseconds" << std::endl;
	}
	return 0;
}
