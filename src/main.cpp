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
const std::map<int, std::string> matrix_map {{1, "A"}, {2, "bi"}, {3, "A1"}, {4, "b1"}, {5, "c1"}, {6, "A2"}, {7, "B2"}, {8, "Cij"},
											 {9, "2b1 + 3c1"}, {10, "B2 - C2"}, {11, "y1"}, {12, "y2"}, {13, "Y3"},
											 {14, "y1y1'"}, {15, "y2y2'"}, {16, "Y3^2"}, {17, "y1y2'"}, {18, "Y3y2"},
											 {19, "y1y1'Y3"}, {20, "Y3^2 + y1y2'"}, {21, "y1y2'Y3y2"}, {22, "22"},
											 {23, "23"}, {24, "24"}, {25, "24"}};

int main(int argc, char* argv[]) {
	int n;
	int proc_num, proc_rank;
	MPI_Status status;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
	MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);


	if (proc_num < 2) {
		std::cerr << "Not working with 1 proc\n";
		exit(-1);
	}

	// defind new struct for MPI
	MPI_Datatype job_type = get_Job_struct_mpi();

	bool isRandGen = false;
	bool isDebugLog = false;
	if (proc_rank == 0) {
		if (argc > 1) {
			if (argv[1] == "-g") {
				isDebugLog = true;
			}
		}
		std::cout << "sizeof(" << sizeof(Matrix) << ")\n";
		std::cout << "n = ";
		std::cin >> n;
		std::cout << "Generate random matrices '(y)es (n)o:  ";

		std::string ans;
		std::cin >> ans;
		std::for_each(ans.begin(), ans.end(), tolower);
		isRandGen = (ans == "yes" || ans == "y") ? true : false;
	}

	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

	Matrix matrix[count_of_nodes];

	// Generate work-flow list
	matrix[8] = generate_C_matrix(n);
	matrix[2] = generate_b_vector(n);

	std::list<Job> work_flow{
		// L1.5
		Job{9, Job::operation::PLUS_WITH_SCALAR, {4, 5}, {2.0, 3.0}},
		Job{10, Job::operation::MINUS, {7, 8}}, //
		// L2
		Job{11, Job::operation::MULT, {1, 2}},	// y1
		Job{12, Job::operation::MULT, {3, 9}},	// y2
		Job{13, Job::operation::MULT, {6, 10}}, // Y3
		// L3
		Job{14, Job::operation::MULT_MAT_TRANSPOSE, {11, 11}}, // y1y1'
		Job{15, Job::operation::MULT_MAT_TRANSPOSE, {12, 12}}, // y2y2'
		Job{16, Job::operation::MULT, {13, 13}}, // Y3^2
		Job{17, Job::operation::MULT_MAT_TRANSPOSE, {11, 12}}, //y1y2'
		Job{18, Job::operation::MULT, {13, 12}},
		// L4
		Job{19, Job::operation::MULT, {14, 13}},
		Job{20, Job::operation::PLUS, {16, 17}},
		Job{21, Job::operation::MULT, {17, 18}},
		//
		Job{22, Job::operation::MULT, {19, 15}},
		Job{23, Job::operation::PLUS, {21, 11}},
		Job{24, Job::operation::PLUS, {22, 20}},
		// Result
		Job{25, Job::operation::MULT, {24, 23}},
	};

	if (proc_rank == 0) {
		if (isRandGen) {
			std::cout << "Add ranodm jobs for generate matrices" << std::endl;
			work_flow.push_front(Job{7, Job::operation::GEN_MATRIX, {n, n}}); // B2
			work_flow.push_front(Job{6, Job::operation::GEN_MATRIX, {n, n}}); // A2
			work_flow.push_front(Job{5, Job::operation::GEN_MATRIX, {n, 1}}); // c1
			work_flow.push_front(Job{4, Job::operation::GEN_MATRIX, {n, 1}});	// b1
			work_flow.push_front(Job{3, Job::operation::GEN_MATRIX, {n, n}});	// A1
			work_flow.push_front(Job{1, Job::operation::GEN_MATRIX, {n, n}}); // A
		}
		else {
			matrix[1].set_shape({n, n});
			matrix[3].set_shape({n, n});
			matrix[4].set_shape({n, 1});
			matrix[5].set_shape({n, 1});
			matrix[6].set_shape({n, n});
			matrix[7].set_shape({n, n});

			std::cin >> matrix[1]
				>> matrix[3]
				>> matrix[4]
				>> matrix[5]
				>> matrix[6]
				>> matrix[7];
			std::cout << "Manually set was saccessfull" << std::endl;
		}
	}

	int id_result = work_flow.back().id;

	// Start timer;
	auto start = std::chrono::high_resolution_clock::now();

	if (proc_rank == 0) { // main proc
		// All matrix

		// std::cout << separator_line << "C2 = \n"
		// 	<< matrix[8];
		// std::cout << separator_line << "bi = \n"
		// 	<< matrix[2];
		MPI_Barrier(MPI_COMM_WORLD);

		Job end_job{ 999, Job::operation::END };

		while (!work_flow.empty()) {
			int rank, opt;
			// Send jobs
			std::cout << "\tMaster= Send set of jobs" << std::endl;
			for (rank = 1; rank < proc_num && !work_flow.empty(); ++rank) {
				auto to_erase = work_flow.end();
				Job send_job = get_job_from_list_and_erase(work_flow, matrix, to_erase);

				if (send_job.op == Job::operation::WAIT) {
					std::cout << "No jobs, wait for res matrices" << std::endl;
					// sleep(3);
					break;
				}
				// Send job to proc_rank
				MPI_Send(&send_job, 1, job_type, rank, send_job.id, MPI_COMM_WORLD);

				if (send_job.op != Job::operation::GEN_MATRIX) {
					MPI_Recv(&opt, 1, MPI_INT, rank, send_job.id, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					if (opt & 1)
						send_matrix_to_proc(matrix[send_job.arg[0]], rank);
					if (opt & 2)
						send_matrix_to_proc(matrix[send_job.arg[1]], rank);
				}
				work_flow.erase(to_erase);
			}
			std::cout << "\tMaster= Recv results" << std::endl;
			// Recv results
			for (int i = 1; i < rank; ++i) {
				std::pair<int, int> shape;
				MPI_Recv(&shape, 2, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

				Matrix temp(shape.first, shape.second);
				MPI_Recv(temp.data(), shape.first * shape.second, MPI_DOUBLE, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

				std::cout << "\tRecived result matrix [" << status.MPI_TAG << "] by " << status.MPI_SOURCE << std::endl;
				matrix[status.MPI_TAG] = temp;
			}
		}
		std::cout << "End working, finish all ranks" << std::endl;
		for (int i = 1; i < proc_num; ++i) {
			MPI_Send(&end_job, 1, job_type, i, 0, MPI_COMM_WORLD);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
	else { // workers

		bool is_loop = true;
		std::pair<int, int> shape1, shape2;
		MPI_Barrier(MPI_COMM_WORLD);
		while (is_loop) {
			Job job;

			MPI_Recv(&job, 1, job_type, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			std::cout << "$$ Recv Job Proc: " << proc_rank << " recv id=" << job.id << " Job_enum = " << (int)job.op
				<< " args {" << job.arg[0] << ", " << job.arg[1] << "}" << std::endl;
			if (job.op == Job::operation::END) {
				is_loop = false;
				continue;
			}

			int send = 0;
			// check for needs to recv matrices from master
			if (job.op != Job::operation::GEN_MATRIX) {
				send = (matrix[job.arg[0]].get_col() == 0) | ((matrix[job.arg[1]].get_col() == 0) << 1);
				// std::cout << "Proc: " << proc_rank << " send= " << send << std::endl;
				MPI_Send(&send, 1, MPI_INT, 0, job.id, MPI_COMM_WORLD);

				// recv data to local space;
				if (send & 1) {
					matrix[job.arg[0]] = recv_mat_from_main(proc_rank, job.arg[0]);
				}
				if (send & 2) {
					matrix[job.arg[1]] = recv_mat_from_main(proc_rank, job.arg[1]);
				}
			}

			do_job(proc_rank, job, matrix);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}

	if (proc_rank == 0) {
		// Timer stop
		auto stop = std::chrono::high_resolution_clock::now();
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
		if (isDebugLog) logoutput(std::cout, matrix, count_of_nodes, matrix_map);

		std::cout << "Result=\n" << matrix[id_result];
	}

	MPI_Type_free(&job_type);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
	return 0;
}
