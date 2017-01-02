/*
 * solver.cpp
 *
 *  Created on: Feb 19, 2015
 *      Author: dimalit
 */

#include "solver.h"
#include <cassert>

// externally-visible:
double init_step;
double atolerance, rtolerance;
int N;
double delta_0;
double alpha;

const double PI = 4*atan(1.0);

PetscErrorCode RHSFunction(TS ts, PetscReal t,Vec in,Vec out,void*);
PetscErrorCode solve(Vec initial_state,
		   int max_steps, double max_time,
		   bool (*step_func)(Vec state, Vec rhs, int steps, double time));
PetscErrorCode step_monitor(TS ts,PetscInt steps,PetscReal time,Vec u,void *mctx);

PetscErrorCode solve(Vec initial_state,
		   int max_steps, double max_time,
		   bool (*step_func)(Vec state, Vec rhs, int steps, double time))
{
//	VecView(initial_state, PETSC_VIEWER_STDERR_WORLD);

	PetscErrorCode ierr;

	int lo, hi;
	VecGetOwnershipRange(initial_state, &lo, &hi);

	TS ts;
	ierr = TSCreate(PETSC_COMM_WORLD, &ts);CHKERRQ(ierr);
	ierr = TSSetProblemType(ts, TS_NONLINEAR);CHKERRQ(ierr);

	ierr = TSSetType(ts, TSRK);CHKERRQ(ierr);
	ierr = TSRKSetType(ts, TSRK4);CHKERRQ(ierr);
	// XXX: strange cast - should work without it too!
	ierr = TSSetRHSFunction(ts, NULL, (PetscErrorCode (*)(TS,PetscReal,Vec,Vec,void*))RHSFunction, 0);CHKERRQ(ierr);

	ierr = TSSetInitialTimeStep(ts, 0.0, init_step);CHKERRQ(ierr);
	ierr = TSSetTolerances(ts, atolerance, NULL, rtolerance, NULL);CHKERRQ(ierr);
//	fprintf(stderr, "steps=%d time=%lf ", max_steps, max_time);

	ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

	ierr = TSSetSolution(ts, initial_state);CHKERRQ(ierr);

	ierr = TSSetDuration(ts, max_steps, max_time);CHKERRQ(ierr);

	ierr = TSMonitorSet(ts, step_monitor, (void*)step_func, NULL);
	ierr = TSSolve(ts, initial_state);CHKERRQ(ierr);			// results are "returned" in step_monitor

	double tstep;
	TSGetTimeStep(ts, &tstep);

	TSDestroy(&ts);
	return 0;
}

PetscErrorCode step_monitor(TS ts,PetscInt steps,PetscReal time,Vec u,void *mctx){
	bool (*step_func)(Vec state, Vec rhs, int steps, double time) = (bool (*)(Vec state, Vec rhs, int steps, double time)) mctx;

	PetscErrorCode ierr;

//	VecView(u, PETSC_VIEWER_STDERR_WORLD);

	// get final RHS
	Vec rhs;
	TSRHSFunction func;
	ierr = TSGetRHSFunction(ts, &rhs, &func, NULL);CHKERRQ(ierr);
	func(ts, time, u, rhs, NULL);	// XXX: why I need to call func instead of getting rhs from TSGetRhsFunction??

//	VecView(u, PETSC_VIEWER_STDERR_WORLD);
//	VecView(rhs, PETSC_VIEWER_STDERR_WORLD);

	PetscInt true_steps;
	TSGetTimeStepNumber(ts, &true_steps);

	bool res = step_func(u, rhs, true_steps, time);
	if(!res){
		//TSSetConvergedReason(ts, TS_CONVERGED_USER);
		TSSetDuration(ts, steps, time);
	}

	return 0;
}

PetscErrorCode RHSFunction(TS ts, PetscReal t,Vec in,Vec out,void*){
//	fprintf(stderr, "%s\n", __FUNCTION__);
	PetscErrorCode ierr;

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	double E, phi;

	// bcast E and phi
	const double* data;
	ierr = VecGetArrayRead(in, &data);CHKERRQ(ierr);
	if(rank == 0){
		E = data[0];
		phi = data[1];
	}
	ierr = VecRestoreArrayRead(in, &data);CHKERRQ(ierr);
	MPI_Bcast(&E, 1, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
	MPI_Bcast(&phi, 1, MPI_DOUBLE, 0, PETSC_COMM_WORLD);

	// compute sums
	int lo, hi;
	VecGetOwnershipRange(in, &lo, &hi);
	if(lo < 2)	// exclude E, phi
		lo = 2;

	double sum_sin = 0;
	double sum_cos = 0;
	for(int i=lo; i<hi; i+=4){
		double apzd[4];
		int indices[] = {i, i+1, i+2, i+3};
		VecGetValues(in, 4, indices, apzd);
		sum_sin += apzd[0]*sin(apzd[1]-apzd[2] - phi);
		sum_cos += apzd[0]*cos(apzd[1]-apzd[2] - phi);
	}

	double tmp = 0;
	MPI_Reduce(&sum_sin, &tmp, 1, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
		sum_sin = tmp;
	tmp = 0;
	MPI_Reduce(&sum_cos, &tmp, 1, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
		sum_cos = tmp;

	// compute derivatives of E_e and phi_e
	if(rank == 0){
		// RESOLVED: here appeared 2.0 and book should have E=E/sqrtR (without 2)
		double dE = 1.0/2.0/N*sum_sin;
		double dphi = - 1.0/2.0/N*sum_cos / E;
//		double dE = theta_e*E_e + 2.0/m*sum_cos;
//		double dphi = (delta_e*E_e - 2.0/m*sum_sin) / E_e;
			VecSetValue(out, 0, dE, INSERT_VALUES);
			VecSetValue(out, 1, dphi, INSERT_VALUES);
	}
	VecAssemblyBegin(out);
	VecAssemblyEnd(out);

	// compute n, a, k
	VecGetOwnershipRange(out, &lo, &hi);
	if(lo < 2)
		lo = 2;

	for(int i=lo; i<hi; i+=4){
		double apzd[4];
		int indices[] = {i, i+1, i+2, i+3};
		VecGetValues(in, 4, indices, apzd);

		double da = - 0.5*E*sin(apzd[1]-apzd[2] - phi);
		double dp = apzd[3] - 0.5*E/apzd[0]*cos(apzd[1] - apzd[2] - phi) - alpha*apzd[0]*apzd[0];

		VecSetValue(out, i, da, INSERT_VALUES);
		VecSetValue(out, i+1, dp, INSERT_VALUES);
		VecSetValue(out, i+2, 0, INSERT_VALUES);
		VecSetValue(out, i+3, 0, INSERT_VALUES);
	}

	VecAssemblyBegin(out);
	VecAssemblyEnd(out);

//	VecView(in, PETSC_VIEWER_STDERR_WORLD);
//	VecView(out, PETSC_VIEWER_STDERR_WORLD);

	return 0;
}

