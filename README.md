# CUDA-TASK-HANDOVER-TIME
Check how much time is needed for CUDA device pick, allocation its memory, computation and result return

If you are interested in results i will present mine, but running on your own is strongly recomended.
Average value ignores 2 first of 10 runs
Those are kind of syntetic tests, what i mean by this is some smaller blocks of code ar ran 10 times (like 10x memory allocation then 10x calculation) not like typical workflow, this may change in future update.

Results on Kepler GPU (Capabilitu 3.0), latest supported CUDA Toolkit is version 10.2 (tested)
Test case 1: 50000 float addition operation
  CUDA - device choosed ns
          min: 100
          max: 41753900
          avg: 137.500000
  CUDA - memory prepare in ns)
          min: 465700
          max: 27941400
          avg: 510325.000000
  CUDA - calculations in ns
          min: 32000
          max: 103100
          avg: 34750.000000
  CUDA - return in ns
          min: 48200
          max: 388400
          avg: 49487.500000
  CUDA - all avg: 594700.000000 ns
  CPU in ns
          min: 91600
          max: 98000
          avg: 91962.500000
          
Test case 2: 100 float addition operation
  CUDA - device choosed ns
          min: 3200
          max: 9000
          avg: 3250.000000
  CUDA - memory prepare in ns
          min: 438700
          max: 546000
          avg: 474925.000000
  CUDA - calculations in ns
          min: 19600
          max: 38200
          avg: 20650.000000
  CUDA - return in ns
          min: 16700
          max: 238900
          avg: 17350.000000
  CUDA - all avg: 516175.000000 ns
  CPU in ns
          min: 200
          max: 300
          avg: 237.500000

