#!/usr/bin/env bash

# Added a timing experiment
# based on reviewer revisions requested for
# OOPSLA 2019


# without instrumentation
mkdir no_instrumentation_results/
python3 -u collect_data.py \
  --as_original \
  --output_dir no_instrumentation_results/ \
  --log no_instrumentation_results/times_without_instrumentation.log \
  --max_concurrent_processes 5 \
    2>&1 | tee  -a no_instrumentation_results/without_instrumentation_log.txt


# with instrumentation
# make sure to run scripts that succeeded when running without instrumentation
mkdir instrumentation_results/
python3 -u collect_data.py \
  --output_dir instrumentation_results/ \
  --log instrumentation_results/times_with_instrumentation.log \
  --max_concurrent_processes 5 \
  --script_ids_log no_instrumentation_results/times_without_instrumentation.log \
  2>&1 | tee  -a instrumentation_results/with_instrumentation_log.txt

# Create plot
python timing_experiments_plot.py \
  -i instrumentation_results/ \
  -n no_instrumentation_results/ \
  -o /tmp/test/
