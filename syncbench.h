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
#include <vector>

#ifndef SYNCBENCH_H
#define SYNCBENCH_H

void refer(const std::vector<std::string> &);

void referatom(const std::vector<std::string> &);

void referred(const std::vector<std::string> &);

void testpr(const std::vector<std::string> &);

void testfor(const std::vector<std::string> &);

void testpfor(const std::vector<std::string> &);

void testbar(const std::vector<std::string> &);

void testsing(const std::vector<std::string> &);

void testcrit(const std::vector<std::string> &);

void testlock(const std::vector<std::string> &);

void testorder(const std::vector<std::string> &);

void testatom(const std::vector<std::string> &);

void testred(const std::vector<std::string> &);

#endif //SYNCBENCH_H
