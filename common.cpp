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
*      modified by John Eblen (jeblen@acm.org) for benchmarking STS         *
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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <sched.h>

#include "common.h"
#include "sts/sts.h"

#define CONF95 1.96

int nthreads = -1;           // Number of OpenMP threads
int delaylength = -1;        // The number of iterations to delay for
int outerreps = -1;          // Outer repetitions
double delaytime = -1.0;     // Length of time to delay for in microseconds
double targettesttime = 0.0; // The length of time in microseconds that the test
                             // should run for.
unsigned long innerreps; // Inner repetitions
double *times;           // Array of doubles storing the benchmark times in microseconds
double referencetime;    // The average reference time in microseconds to perform
			 // outerreps runs
double referencesd;      // The standard deviation in the reference time in
			 // microseconds for outerreps runs.
double testtime;         // The average test time in microseconds for
			 // outerreps runs
double testsd;		 // The standard deviation in the test time in
			 // microseconds for outerreps runs.

void usage(char *argv[]) {
    printf("Usage: %s.x \n"
           "\t--num-threads <num-threads> (default 10)\n"
	   "\t--outer-repetitions <outer-repetitions> (default %d)\n"
	   "\t--test-time <target-test-time> (default %0.2f microseconds)\n"
	   "\t--delay-time <delay-time> (default %0.4f microseconds)\n"
	   "\t--delay-length <delay-length> "
	   "(default auto-generated based on processor speed)\n",
	   argv[0],
	   DEFAULT_OUTER_REPS, DEFAULT_TEST_TARGET_TIME, DEFAULT_DELAY_TIME);
}

void parse_args(int argc, char *argv[]) {
    // Parse the parameters
    int arg;
    for (arg = 1; arg < argc; arg++) {
	if (strcmp(argv[arg], "--delay-time") == 0.0) {
	    delaytime = atof(argv[++arg]);
	    if (delaytime == 0.0) {
		printf("Invalid float:--delay-time: %s\n", argv[arg]);
		usage(argv);
		exit(EXIT_FAILURE);
	    }
	
        } else if (strcmp(argv[arg], "--num-threads") == 0) {
            nthreads = atoi(argv[++arg]);
	    if (nthreads < 1) {
		printf("Invalid integer:--num-threads: %s\n", argv[arg]);
		usage(argv);
		exit(EXIT_FAILURE);
	    }
	
	} else if (strcmp(argv[arg], "--outer-repetitions") == 0) {
	    outerreps = atoi(argv[++arg]);
	    if (outerreps == 0) {
		printf("Invalid integer:--outer-repetitions: %s\n", argv[arg]);
		usage(argv);
		exit(EXIT_FAILURE);
	    }
		
	} else if (strcmp(argv[arg], "--test-time") == 0) {
	    targettesttime = atof(argv[++arg]);
	    if (targettesttime == 0) {
		printf("Invalid integer:--test-time: %s\n", argv[arg]);
		usage(argv);
		exit(EXIT_FAILURE);
	    }
		
	} else if (strcmp(argv[arg], "-h") == 0) {
	    usage(argv);
	    exit(EXIT_SUCCESS);
		
	} else {
	    printf("Invalid parameters: %s\n", argv[arg]);
	    usage(argv);
	    exit(EXIT_FAILURE);
	}
    }
}

int getdelaylengthfromtime(double delaytime) {
    int i, reps;
    double lapsedtime, starttime; // seconds

    reps = 1000;
    lapsedtime = 0.0;

    delaytime = delaytime/1.0E6; // convert from microseconds to seconds
    // Note: delaytime is local to this function and thus the conversion
    // does not propagate to the main code. 

    // Here we want to use the delaytime in microseconds to find the 
    // delaylength in iterations. We start with delaylength=0 and 
    // increase until we get a large enough delaytime, return delaylength 
    // in iterations. 

    delaylength = 0;
    delay(delaylength);

    while (lapsedtime < delaytime) {
	delaylength = delaylength * 1.1 + 1;
	starttime = getclock();
	for (i = 0; i < reps; i++) {
	    delay(delaylength);
	}
	lapsedtime = (getclock() - starttime) / (double) reps;
    }
    return delaylength;

}

void setAffinity(int coreId) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(coreId, &mask);
    if (sched_setaffinity(0, sizeof(mask), &mask) != 0)
    {
        perror("sched_setaffinity");
    }
}

void init_sts_threads() {
    STS::startup(nthreads);
    // Set thread affinities here if needed
    STS::getInstance("default")->parallel_for("set_affinities", 0, nthreads, [&](int i) {
        setAffinity(i);
    });
}
/* Assign STS threads for a specific test */
std::vector<std::string> assign_sts_threads(const std::vector<STS_Task_Info> &sti, unsigned long reps) {
    std::vector<std::string> task_labels;

    static STS *sts = nullptr;
    if (sts == nullptr) {
        sts = new STS("bmark");
    }
    else {
        sts->wait();
    }
    sts->clearAssignments();
    for (int t=0; t<nthreads; t++) {
        for (int b=0; b<sti.size(); b++) {
            if (sti[b].type == RUN_ONCE) {
                task_labels.push_back(sti[b].label);
                sts->assign(task_labels.back(), t);
            }
            else if (sti[b].type == LOOP_ONCE) {
                task_labels.push_back(sti[b].label);
                sts->assign(task_labels.back(), t, {{t,nthreads},{t+1,nthreads}});
            }
            else {
                for (int r=0; r<reps; r++) {
                    if (sti[b].type == RUN_MANY) {
                        task_labels.push_back(sti[b].label + std::to_string(r));
                        sts->assign(task_labels.back(), t);
                    }
                    else { // LOOP_MANY
                        task_labels.push_back(sti[b].label + std::to_string(r));
                        sts->assign(task_labels.back(), t, {{t,nthreads},{t+1,nthreads}});
                    }
                }
            }
        }
    }
    sts->nextStep();
    return task_labels;
}

unsigned long getinnerreps(void (*test)(const std::vector<std::string> &), const std::vector<STS_Task_Info> &sti = {}) {
    innerreps = 10L;  // some initial value
    double time = 0.0;
    std::vector<std::string> task_labels;
    while (time < targettesttime) {
        if (sti.size() > 0)  task_labels = assign_sts_threads(sti, innerreps);
	double start  = getclock();
	test(task_labels);
    if (sti.size() > 0) wait();
	time = (getclock() - start) * 1.0e6;
	innerreps *=2;
	// Test to stop code if compiler is optimising reference time expressions away
	if (innerreps > (targettesttime*1.0e15)) {
	    printf("Compiler has optimised reference loop away, STOP! \n");
	    printf("Try recompiling with lower optimisation level \n");
	    exit(1);
	}
    }
    return innerreps;
}

void printheader(char *name) {
    printf("\n");
    printf("--------------------------------------------------------\n");
    printf("Computing %s time using %lu reps\n", name, innerreps);
}

void stats(double *mtp, double *sdp) {

    double meantime, totaltime, sumsq, mintime, maxtime, sd, cutoff;

    int i, nr;

    mintime = 1.0e10;
    maxtime = 0.;
    totaltime = 0.;

    for (i = 1; i <= outerreps; i++) {
	mintime = (mintime < times[i]) ? mintime : times[i];
	maxtime = (maxtime > times[i]) ? maxtime : times[i];
	totaltime += times[i];
    }

    meantime = totaltime / outerreps;
    sumsq = 0;

    for (i = 1; i <= outerreps; i++) {
	sumsq += (times[i] - meantime) * (times[i] - meantime);
    }
    sd = sqrt(sumsq / (outerreps - 1));

    cutoff = 3.0 * sd;

    nr = 0;

    for (i = 1; i <= outerreps; i++) {
	if (fabs(times[i] - meantime) > cutoff)
	    nr++;
    }

    printf("\n");
    printf("Sample_size       Average     Min         Max          S.D.          Outliers\n");
    printf(" %d                %f   %f   %f    %f      %d\n",
	   outerreps, meantime, mintime, maxtime, sd, nr);
    printf("\n");

    *mtp = meantime;
    *sdp = sd;

}

void printfooter(char *name, double testtime, double testsd,
		 double referencetime, double refsd) {
    printf("%s time     = %f microseconds +/- %f\n",
	   name, testtime, CONF95*testsd);
    printf("%s overhead = %f microseconds +/- %f\n",
	   name, testtime-referencetime, CONF95*(testsd+referencesd));

}

void printreferencefooter(char *name, double referencetime, double referencesd) {
    printf("%s time     = %f microseconds +/- %f\n",
	   name, referencetime, CONF95 * referencesd);
}

void init(int argc, char **argv)
{
    parse_args(argc, argv);

    if (nthreads < 1) nthreads = 10;
    init_sts_threads();

    if (outerreps == -1) {
	outerreps = DEFAULT_OUTER_REPS;
    }
    if (targettesttime == 0.0) {
	targettesttime = DEFAULT_TEST_TARGET_TIME;
    }
    if (delaytime == -1.0) {
	delaytime = DEFAULT_DELAY_TIME; 
    }
    delaylength = getdelaylengthfromtime(delaytime); // Always need to compute delaylength in iterations 
    
    times = (double *)malloc((outerreps+1) * sizeof(double));

    printf("Running OpenMP benchmark version 3.0\n"
	   "\t%d thread(s)\n"
	   "\t%d outer repetitions\n"
	   "\t%0.2f test time (microseconds)\n"
	   "\t%d delay length (iterations) \n"
	   "\t%f delay time (microseconds)\n",
	   nthreads,
	   outerreps, targettesttime,
	   delaylength, delaytime);
}

void finalise(void) {
    free(times);
    STS::getInstance("bmark")->wait();
    STS::shutdown();
}

void initreference(char *name) {
    printheader(name);

}

/* Calculate the reference time. */
void reference(char *name, void (*refer)(const std::vector<std::string> &)) {
    int k;
    double start;
    std::vector<std::string> dummy_task_labels;
    // Calculate the required number of innerreps
    innerreps = getinnerreps(refer);

    initreference(name);

    for (k = 0; k <= outerreps; k++) {
	start = getclock();
	refer(dummy_task_labels);
	times[k] = (getclock() - start) * 1.0e6 / (double) innerreps;
    }

    finalisereference(name);

}

void finalisereference(char *name) {
    stats(&referencetime, &referencesd);
    printreferencefooter(name, referencetime, referencesd);

}

void intitest(char *name) {
    printheader(name);

}

void finalisetest(char *name) {
    stats(&testtime, &testsd);
    printfooter(name, testtime, testsd, referencetime, referencesd);

}

/* Function to run a microbenchmark test*/

void benchmark(char *name, void (*test)(const std::vector<std::string> &), const std::vector<STS_Task_Info> &sti)
{
    int k;
    double start;

    // Calculate the required number of innerreps
    innerreps = getinnerreps(test, sti);

    intitest(name);

    for (k=0; k<=outerreps; k++) {
    std::vector<std::string> task_labels = assign_sts_threads(sti, innerreps);
	start = getclock();
	test(task_labels);
    wait();
	times[k] = (getclock() - start) * 1.0e6 / (double) innerreps;
    }

    finalisetest(name);

}

// For the Cray compiler on HECToR we need to turn off optimisation 
// for the delay and array_delay functions. Other compilers should
// not be afffected. 
#pragma _CRI noopt
void delay(int delaylength) {

    int i;
    float a = 0.;

    for (i = 0; i < delaylength; i++)
	a += i;
    if (a < 0)
	printf("%f \n", a);

}

void array_delay(int delaylength, double a[1]) {

    int i;
    a[0] = 1.0;
    for (i = 0; i < delaylength; i++)
	a[0] += i;
    if (a[0] < 0)
	printf("%f \n", a[0]);

}
// Re-enable optimisation for remainder of source. 
#pragma _CRI opt

double getclock() {
    double time;
    // Returns a value in seconds of the time elapsed from some arbitrary,
    // but consistent point.
    double omp_get_wtime(void);
    time = omp_get_wtime();
    return time;
}

int returnfalse() {
    return 0;

}

