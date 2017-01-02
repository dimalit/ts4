/*
 * solver.h
 *
 *  Created on: Feb 19, 2015
 *      Author: dimalit
 */

#ifndef SOLVER_H_
#define SOLVER_H_

#include <petscts.h>
#include <mpi.h>

extern double init_step;
extern double atolerance, rtolerance;
extern int N;
extern double delta_0;
extern double alpha;

extern PetscErrorCode solve(Vec initial_state, int max_steps, double max_time,
		   bool (*step_func)(Vec state, Vec rhs, int steps, double time));

#endif /* SOLVER_H_ */
