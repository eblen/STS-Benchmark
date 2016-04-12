/****************************************************************************
*                                                                           *
*             OpenMP MicroBenchmark Suite - Version 3.1                     *
*                                                                           *
*                            produced by                                    *
*                                                                           *
*             Mark Bull, Fiona Reid and Nix Mc Donnell                      *
*                                                                           *
*                                at                                         *
*                                                                           *
*                Edinburgh Parallel Computing Centre                        *
*                                                                           *
*         email: markb@epcc.ed.ac.uk or fiona@epcc.ed.ac.uk                 *
*                                                                           *
*                                                                           *
*      This version copyright (c) The University of Edinburgh, 2015.        *
*                                                                           *
*                                                                           *
*  Licensed under the Apache License, Version 2.0 (the "License");          *
*  you may not use this file except in compliance with the License.         *
*  You may obtain a copy of the License at                                  *
*                                                                           *
*      http://www.apache.org/licenses/LICENSE-2.0                           *
*                                                                           *
*  Unless required by applicable law or agreed to in writing, software      *
*  distributed under the License is distributed on an "AS IS" BASIS,        *
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. *
*  See the License for the specific language governing permissions and      *
*  limitations under the License.                                           *
*                                                                           *
****************************************************************************/
#include <cassert>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <string>

#include "barrier.h"

#include "common.h"
#include "sts.h"
#include "syncbench.h"

int main(int argc, char **argv) {
    init(argc, argv);

    /* GENERATE REFERENCE TIME */
    reference("reference time 1", &refer);

    /* TEST PARALLEL REGION */
    benchmark("PARALLEL", &testpr, std::vector<STS_Task_Info>{{"TESTPR", RUN_MANY}});

    /* TEST FOR */
    benchmark("FOR", &testfor, std::vector<STS_Task_Info>{{"TESTFOR", RUN_ONCE},{"TESTFOR_0", LOOP_MANY}});

    /* TEST PARALLEL FOR */
    benchmark("PARALLEL FOR", &testpfor, std::vector<STS_Task_Info>{{"TESTPFOR_0", LOOP_MANY}});

    /* TEST BARRIER */
    benchmark("BARRIER", &testbar, std::vector<STS_Task_Info>{{"TESTBAR", RUN_ONCE}});

    /* GENERATE NEW REFERENCE TIME */
    reference("reference time 3", &referred);

    /* TEST REDUCTION (1 var)  */
    benchmark("REDUCTION", &testred, std::vector<STS_Task_Info>{{"TESTRED_0", LOOP_MANY}});

    finalise();
    return EXIT_SUCCESS;
}

void refer(const std::vector<std::string> &task_labels) {
    int j;
    for (j = 0; j < innerreps; j++) {
	delay(delaylength);
    }
}

void referred(const std::vector<std::string> &task_labels) {
    int j;
    int aaaa = 0;
    for (j = 0; j < innerreps; j++) {
	delay(delaylength);
	aaaa += 1;
    }
}

void testpr(const std::vector<std::string> &task_labels) {
    for (int j = 0; j < innerreps; j++) {
        run(task_labels[j], [=]() {
	        delay(delaylength);
	    });
    }
}

void testfor(const std::vector<std::string> &task_labels) {
    run("TESTFOR", [=]() {
	    for (int j = 0; j < innerreps; j++) {
	        parallel_for(task_labels[j+1], 0, nthreads, [=](size_t i) {
	            delay(delaylength);
	        });
	    }
    });
}

void testpfor(const std::vector<std::string> &task_labels) {
    for (int j = 0; j < innerreps; j++) {
        parallel_for(task_labels[j], 0, nthreads, [=](size_t i){
            delay(delaylength);
        });
    }
}

void testbar(const std::vector<std::string> &task_labels) {
    static Barrier b(nthreads);
    run("TESTBAR", [=]() {
        for (int j = 0; j < innerreps; j++) {
            delay(delaylength);
            b.enter();
        }
    });
}

void testred(const std::vector<std::string> &task_labels) {
    for (int j = 0; j < innerreps; j++) {
        TaskReduction<int> red = createTaskReduction(task_labels[j], 0);
        parallel_for(task_labels[j], 0, nthreads, [=](size_t i) {
            delay(delaylength);
            collect(1);
        }, &red);
        assert(red.getResult() == nthreads);
    }
}

