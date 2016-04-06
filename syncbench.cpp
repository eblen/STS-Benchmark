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

#include "sts.h"
#include "barrier.h"

#include "common.h"
#include "syncbench.h"

void init_sts_threads() {
    enum TestType {RUN_ONCE, RUN_MANY, LOOP_ONCE, LOOP_MANY};
    std::vector<TestType> ttypes = {RUN_MANY, RUN_ONCE, LOOP_MANY, LOOP_MANY, RUN_ONCE, LOOP_MANY};
    std::vector<std::string> bmarks = {"TESTPR", "TESTFOR", "TESTFOR_0", "TESTPFOR_0", "TESTBAR", "TESTRED_0"};

    setNumThreads(nthreads);
    clearAssignments();

    for (int t=0; t<nthreads; t++) {
        for (int b=0; b<bmarks.size(); b++) {
            if (ttypes[b] == RUN_ONCE) {
                assign(bmarks[b], t);
            }
            else if (ttypes[b] == LOOP_ONCE) {
                assign(bmarks[b], t, {{t,nthreads},{t+1,nthreads}});
            }
            else {
                for (int r=0; r<innerreps; r++) {
                    if (ttypes[b] == RUN_MANY) {
                        assign(bmarks[b] + std::to_string(r), t);
                    }
                    else { // LOOP_MANY
                        assign(bmarks[b] + std::to_string(r), t, {{t,nthreads},{t+1,nthreads}});
                    }
                }
            }
        }
    }
    nextStep();
}

int main(int argc, char **argv) {
    init(argc, argv);

    /* GENERATE REFERENCE TIME */
    reference("reference time 1", &refer);

    init_sts_threads();

    /* TEST PARALLEL REGION */
    benchmark("PARALLEL", &testpr);

    /* TEST FOR */
    benchmark("FOR", &testfor);

    /* TEST PARALLEL FOR */
    benchmark("PARALLEL FOR", &testpfor);

    /* TEST BARRIER */
    benchmark("BARRIER", &testbar);

    /* GENERATE NEW REFERENCE TIME */
    reference("reference time 3", &referred);

    /* TEST REDUCTION (1 var)  */
    benchmark("REDUCTION", &testred);

    finalise();
    return EXIT_SUCCESS;
}

void refer() {
    int j;
    for (j = 0; j < innerreps; j++) {
	delay(delaylength);
    }
}

void referred() {
    int j;
    int aaaa = 0;
    for (j = 0; j < innerreps; j++) {
	delay(delaylength);
	aaaa += 1;
    }
}

void testpr() {
    for (int j = 0; j < innerreps; j++) {
        std::string taskName = "TESTPR" + std::to_string(j);
        run(taskName, [=]() {
	        delay(delaylength);
	    });
    }
}

void testfor() {
    run("TESTFOR", [=]() {
	    for (int j = 0; j < innerreps; j++) {
                std::string taskName = "TESTFOR_0" + std::to_string(j);
	        parallel_for(taskName, 0, nthreads, [=](size_t i) {
printf("%d %d\n", i, j);
	            delay(delaylength);
	        });
	    }
    });
}

void testpfor() {
    for (int j = 0; j < innerreps; j++) {
        std::string taskName = "TESTPFOR_0" + std::to_string(j);
        parallel_for(taskName, 0, nthreads, [=](size_t i){
            delay(delaylength);
        });
    }
}

void testbar() {
    static Barrier b(nthreads);
    run("TESTBAR", [=]() {
        for (int j = 0; j < innerreps; j++) {
            delay(delaylength);
            b.enter();
        }
    });
}

void testred() {
    TaskReduction<int> red = createTaskReduction("TESTRED_0", 0);
    for (int j = 0; j < innerreps; j++) {
        std::string taskName = "TESTRED_0" + std::to_string(j);
        parallel_for(taskName, 0, nthreads, [=](size_t i) {
            delay(delaylength);
            collect(1);
        }, &red);
    }
    assert(red.getResult() == innerreps * nthreads);
}

