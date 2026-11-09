'''
This package contains the benchmarking and profiling code, one module per benchmark type:
- common.py: shared base configuration, optimization journeys, builders, measurement utilities, training step, run-or-reload
- train.py: training throughput and peak memory benchmark
- generate.py: generation (decode throughput, time to first token, memory) benchmark
- profiling.py: profiler of a training step, a prompt prefill, or one decode step
- memory_fit.py: memory model fit mem(T) = c + a T + b T^2 to a saved benchmark

Every benchmark and profile writes to 'results/<kind>_<preset>_<device>/' (the kind starts with 'benchmark' or
'profile', which keeps these folders apart from training runs):
- config.json: the full configuration, including derived values
- env.json: code version, software, and hardware at the start
- results.csv: flat raw rows, enough to rebuild the report with '--from-csv' (the profiler also writes ops.csv)
- report.md: the tables, followed by the configuration and the environment
- run.log: everything logged during the run


'''
