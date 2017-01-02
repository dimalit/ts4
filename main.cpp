/*
 * main.cpp
 *
 *  Created on: Aug 20, 2014
 *      Author: dimalit
 */

#include <model_e4.pb.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>
using namespace pb;

#include "solver.h"

#include <petscts.h>
#include <mpi.h>
#include <cstring>
#include <unistd.h>

E4Config pconfig;
EXPetscSolverConfig sconfig;
E4State state;

int rank;
int size;
clock_t t1;		// for time counting in step_func
int max_steps; double max_time;
bool use_step = false;

void vec_to_state(Vec v, E4State*);
void state_to_vec(const E4State* state, Vec v);
bool step_func(Vec u, Vec rhs, int steps, double time);

// TMP
//#include <fcntl.h>

void broadcast_message(google::protobuf::Message& msg){
	char* buf; int buf_size;

	if(rank == 0)
		buf_size = msg.ByteSize();
	MPI_Bcast(&buf_size, 1, MPI_INT, 0, PETSC_COMM_WORLD);

	buf = new char[buf_size];

	if(rank == 0)
		msg.SerializeToArray(buf, buf_size);

	MPI_Bcast(buf, buf_size, MPI_BYTE, 0, PETSC_COMM_WORLD);

	if(rank != 0)
		msg.ParseFromArray(buf, buf_size);

	delete[] buf;
}

void parse_with_prefix(google::protobuf::Message& msg, int fd){
	int size;
	int ok = read(fd, &size, sizeof(size));
	assert(ok == sizeof(size));

	//TODO:without buffer cannot read later bytes
	char *buf = (char*)malloc(size);
	int read_size = 0;
	while(read_size != size){
		ok = read(fd, buf+read_size, size-read_size);
		read_size+=ok;
		assert(ok > 0 || read_size==size);
	}
	msg.ParseFromArray(buf, size);
	free(buf);
}

int main(int argc, char** argv){

//	int go = 0;
//	while(go==0){
//		sleep(1);
//	}

//	close(0);
//	open("../ode-env/all.tmp", O_RDONLY);

	if(argc > 1 && strcmp(argv[1], "use_step")==0){
		use_step = true;
		argc--;
		argv++;
	}

	PetscErrorCode ierr;
	ierr = PetscInitialize(&argc, &argv, (char*)0, (char*)0);CHKERRQ(ierr);

	ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);
	ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRQ(ierr);

	if(rank==0){
		E4Model all;
		//all.ParseFromFileDescriptor(0);
		parse_with_prefix(all, 0);

		sconfig.CopyFrom(all.sconfig());
		pconfig.CopyFrom(all.pconfig());
		state.CopyFrom(all.state());
	}

	broadcast_message(sconfig);
	broadcast_message(pconfig);

	// set global parameters
	init_step = sconfig.init_step();
	atolerance = sconfig.atol();
	rtolerance = sconfig.rtol();
	N = pconfig.n();
	delta_0 = pconfig.delta_0();
	alpha = pconfig.alpha();

	Vec u;
	VecCreate(PETSC_COMM_WORLD, &u);
	VecSetType(u, VECMPI);

	int addition = rank==0 ? 2 : 0;
	VecSetSizes(u, addition+pconfig.n()*4/size, PETSC_DECIDE);

	state_to_vec(&state, u);

	if(rank == 0){
		int ok;
		ok = read(0, &max_steps, sizeof(max_steps));
			assert(ok==sizeof(max_steps));
		ok = read(0, &max_time, sizeof(max_time));
			assert(ok==sizeof(max_time));
	}

	MPI_Bcast(&max_steps, 1, MPI_INT, 0, PETSC_COMM_WORLD);
	MPI_Bcast(&max_time, 1, MPI_DOUBLE, 0, PETSC_COMM_WORLD);

	t1 = clock();

	solve(u, max_steps, max_time, step_func);

	VecDestroy(&u);

	ierr = PetscFinalize();CHKERRQ(ierr);
	return 0;
}

bool step_func(Vec res, Vec res_rhs, int passed_steps, double passed_time){
	clock_t t2 = clock();
	double dtime = (double)(t2-t1)/CLOCKS_PER_SEC;

//	VecView(res_rhs, PETSC_VIEWER_STDERR_(PETSC_COMM_WORLD));

	// return if not using steps
	if(!use_step && passed_steps < max_steps && passed_time < max_time)
		return true;
	// wait if using steps
	else if(use_step){
		int ok;
		char c;
		if(rank==0){
			ok = read(0, &c, sizeof(c));
				assert(ok==sizeof(c));
		}
		MPI_Bcast(&c, 1, MPI_BYTE, 0, PETSC_COMM_WORLD);
		if(c=='f')
			return false;			// finish!
		assert(c=='s');
	}// if use_steps


	E4Solution sol;
	if(rank==0){
		for(int i=0; i<N; i++){
			sol.mutable_state()->add_particles();
			sol.mutable_d_state()->add_particles();
		}
	}

	vec_to_state(res, sol.mutable_state());
	vec_to_state(res_rhs, sol.mutable_d_state());

	if(rank == 0){
		// 1 write time and steps
		write(1, &passed_steps, sizeof(passed_steps));
		write(1, &passed_time, sizeof(passed_time));

		// 2 write state
		int size = sol.ByteSize();
		write(1, &size, sizeof(size));
		sol.SerializeToFileDescriptor(1);
	}

	t1 = t2;
	return true;
}

void state_to_vec(const E4State* state, Vec v){
	int vecsize;
	VecGetSize(v, &vecsize);
	assert(vecsize == pconfig.n()*4+2);

	double *arr;
	VecGetArray(v, &arr);

	PetscInt* borders;
	VecGetOwnershipRanges(v, (const PetscInt**)&borders);

	if(rank == 0){

		// write E and phi
		arr[0] = state->e();
		arr[1] = state->phi();

		for(int r = size-1; r>=0; r--){		// go down - because last will be mine
			int lo = borders[r];
			int hi = borders[r+1];
			if(r==0)
				lo += 2;

			assert((lo-2)%4 == 0);
			assert((hi-2)%4 == 0);

			int first = (lo-2) / 4;
			int count = (hi - lo) / 4;

			for(int i=0; i<count; i++){
				E4State::Particles p = state->particles(first+i);
				arr[2 + i*4+0] = p.a();
				arr[2 + i*4+1] = p.psi();
				arr[2 + i*4+2] = p.z();
				arr[2 + i*4+3] = p.delta();
			}

			if(r!=0)
				MPI_Send(arr+2, count*4, MPI_DOUBLE, r, 0, PETSC_COMM_WORLD);

		}// for

	}// if rank == 0
	else{
		int count3 = borders[rank+1] - borders[rank];
		MPI_Status s;
		int ierr = MPI_Recv(arr, count3, MPI_DOUBLE, 0, 0, PETSC_COMM_WORLD, &s);
		assert(MPI_SUCCESS == ierr);
	}// if rank != 0

	VecRestoreArray(v, &arr);
}

void vec_to_state(Vec v, E4State* state){
	const double *arr;
	VecGetArrayRead(v, &arr);

	PetscInt* borders;
	VecGetOwnershipRanges(v, (const PetscInt**)&borders);

	if(rank == 0){

		PetscScalar* buf = (PetscScalar*)malloc(sizeof(PetscScalar)*(borders[1]-borders[0]));

		state->set_e(arr[0]);
		state->set_phi(arr[1]);

		for(int r = 0; r<size; r++){
			int lo = borders[r];
			int hi = borders[r+1];

			assert((lo-2)%4 == 0 || lo==0);
			assert((hi-2)%4 == 0);

			int first = (lo-2) / 4;
			int count = (hi - lo) / 4;

			MPI_Status s;
			if(r!=0){
				int ok = MPI_Recv(buf, count*4, MPI_DOUBLE, r, 0, PETSC_COMM_WORLD, &s);
				assert(MPI_SUCCESS == ok);
			}
			else	// copy only particles (arr+2)
				memcpy(buf, arr+2, sizeof(PetscScalar)*(hi-lo-2));

			for(int i=0; i<count; i++){
				E4State::Particles* p = state->mutable_particles(first+i);
				p->set_a(buf[i*4+0]);

				double psi = buf[i*4+1];
				if(psi > 2*M_PI)
					psi -= 2*M_PI;
				else if(psi < 0)
					psi += 2*M_PI;

				p->set_psi(psi);
				p->set_z(buf[i*4+2]);
				p->set_delta(buf[i*4+3]);
			}
		}// for

		free(buf);
	}// if rank == 0
	else{
		int count3 = borders[rank+1] - borders[rank];
		int ierr = MPI_Send((double*)arr, count3, MPI_DOUBLE, 0, 0, PETSC_COMM_WORLD);
		assert(MPI_SUCCESS == ierr);
	}// if rank != 0

	VecRestoreArrayRead(v, &arr);
}

