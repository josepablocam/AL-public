# Script to collect a sample of dynamic traces for initial experimentation
from argparse import ArgumentParser
import sys
import os
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../traces/')
)
sys.path.append('../')
from collections import defaultdict
import copy
import hashlib
import os
import time
import subprocess
import tempfile
import time

from collection import errors
from instrumentation import transforms
from traces import *

import numpy as np

DATA_ROOT = "../../../data/"


def read_log(log_file):
    with open(log_file, 'r') as f:
        id_and_status = lambda e: tuple(int(v) for v in e.split(":"))
        contents = [
            id_and_status(e) for e in f.read().split('\n') if len(e) > 0
        ]
        contents = dict(contents)
        return contents


class DataCollector(object):
    def __init__(
            self,
            timeout='10m',
            sleep=1,
            max_concurrent_scripts=3,
            log_file=None,
    ):
        """
    Collect data by spinning up multiple processes, each executing an instrumented Kaggle script
    :param timeout: Time after which to fail executing kaggle script with timeout error
    :param sleep: Number of minutes to sleep between process status polls
    :param max_concurrent_scripts: Maximum number of concurrently executing kaggle scripts
    :param log_file: file to log script ids that complete (succesfully or with error, status logged as well)
    """
        self.timeout = timeout
        self.sleep = sleep
        self.max_concurrent_scripts = max_concurrent_scripts
        self.log_file = log_file
        self.active_scripts = []

    def uses_sklearn(self, module_names):
        check = 'sklearn'
        for name in module_names:
            if name[:len(check)] == check:
                return True
        return False

    def collect_script_sources(self, project_ids):
        """ Collect source code for scripts in relevant project ids """
        # map from kaggle project id to set of scripts
        project_traces = defaultdict(lambda: [])
        # use static traces we had already collected for simplicity...
        traces_file_nm = os.path.abspath(
            os.path.join(DATA_ROOT, "meta-kaggle/clean-traces.pkl")
        )
        with open(traces_file_nm, 'rb') as f:
            all_traces = pickle.load(f)
            # filter to only consider scripts that use sklearn at least once
            sklearn_traces = [
                t for t in all_traces if self.uses_sklearn(t[-1])
            ]
            print("All traces: %d" % len(all_traces))
            print("Sklearn traces: %d" % len(sklearn_traces))
            for t in sklearn_traces:
                project_id = t[0].project_id
                if project_id in project_ids:
                    project_traces[project_id].append(t)
        return project_traces

    def log_script_status(self, script_id, status):
        if self.log_file is not None:
            if not os.path.exists(self.log_file):
                perms = 'w'
            else:
                perms = 'a'
            with open(self.log_file, perms) as f:
                f.write("%d:%d\n" % (script_id, status))

    def read_log(self):
        if self.log_file is not None and os.path.exists(self.log_file):
            return read_log(self.log_file)
        else:
            return {}

    def dispatch_scripts(
            self,
            target_n,
            traces,
            exec_dir,
            exclude,
            output_dir=None,
            as_original=False,
            sequential=False,
    ):
        """ Dispatch a new set of scripts for concurrent execution """
        while len(self.active_scripts) < min(
                target_n, self.max_concurrent_scripts) and len(traces) > 0:
            # add in N scripts whenever we are done
            t = traces[0]
            traces = traces[1:]
            src = t[1]
            src_id = t[0].script_id
            if src_id in exclude:
                if exclude[src_id] == 0:
                    # dummy to mark previously completed succesfully
                    self.active_scripts.append(None)
            else:
                src_file = tempfile.NamedTemporaryFile(mode='w')
                src_file.write(src)
                src_file.flush()

                if output_dir is None:
                    output_dir = exec_dir

                out_file_name = os.path.join(
                    output_dir, 'script_{}.pkl'.format(src_id)
                )
                timeout_cmd = ['timeout', self.timeout]
                python_cmd = ['python3']
                kaggle_cmd = [
                    './execute_kaggle.py', src_file.name, '--file', '--id',
                    str(src_id), '--dir', exec_dir, '--write', out_file_name
                ]
                if as_original:
                    kaggle_cmd += ["--as_original"]
                combined_cmd = timeout_cmd + python_cmd + kaggle_cmd
                proc = subprocess.Popen(combined_cmd)
                if sequential:
                    proc.communicate()
                self.active_scripts.append((proc, src_id, src_file))
        return traces

    def check_scripts(self):
        """ Check if any of the active scripts have completed (succesfully or unsuccesfully) """
        # check if we added any scripts that are just dummies as completed before
        orig_len = len(self.active_scripts)
        self.active_scripts = [e for e in self.active_scripts if e is not None]
        previously_completed = orig_len - len(self.active_scripts)
        # sleep before checking if anything is done executing if necessary
        if len(self.active_scripts) > 0:
            time.sleep(self.sleep * 60)
        # check script statuses after executing
        ct_new_finished = previously_completed
        still_executing = []
        for proc, src_id, src_file in self.active_scripts:
            # poll each script to see if they are done executing
            ret = proc.poll()
            if ret is None:
                still_executing.append((proc, src_id, src_file))
            else:
                src_file.close()
                if ret == errors.ERROR_TIMEOUT:
                    print(
                        "Script %s exceeded %s timeout" %
                        (src_id, self.timeout)
                    )
                if ret == 0:
                    print("Script %s executed succesfully" % src_id)
                    ct_new_finished += 1
                self.log_script_status(src_id, ret)
        self.active_scripts = still_executing
        return ct_new_finished

    def collect_data(
            self,
            scripts_per_project,
            project_ids,
            script_ids=None,
            as_original=False,
            output_dir=None,
            sequential=False,
    ):
        """ Collect instrumented data for a number of scripts in relevant project ids """
        project_traces = self.collect_script_sources(project_ids)
        if script_ids is not None:
            print("Filtering down to target script_ids")
            project_traces = {
                project_id:
                [e for e in project_traces if e[0].script_id in script_ids]
                for project_id, project_traces in project_traces.items()
            }

        # load in anything we've done or tried to do and failed before
        exclude = self.read_log()
        # run timeouts again, sometimes these complete
        exclude = {
            k: v
            for k, v in exclude.items() if v != errors.ERROR_TIMEOUT
        }
        for project_id, traces in project_traces.items():
            exec_dir = os.path.expanduser(
                os.path.join(
                    DATA_ROOT,
                    "executed",
                    "project_{}".format(project_id),
                    "scripts",
                )
            )
            exec_dir = os.path.abspath(exec_dir)
            print(
                "Executing Kaggle scripts from %s (%d possible)" %
                (exec_dir, len(traces))
            )
            n_collected = 0
            if script_ids is None:
                # hack to keep running and try to runn
                # all desired script ids
                # will stop when traces == 0
                # i.e. we have dispatched all the scripts we want
                n_collected = (-np.inf)

            while n_collected < scripts_per_project and len(traces) > 0:
                traces = self.dispatch_scripts(
                    scripts_per_project,
                    traces,
                    exec_dir,
                    exclude,
                    as_original=as_original,
                    output_dir=output_dir,
                    sequential=sequential,
                )
                ct_new_finished = self.check_scripts()
                n_collected += ct_new_finished

        # continue running if things are pending
        while len(self.active_scripts) > 0:
            self.check_scripts()
        print("Done executing Kaggle scripts")


def get_args():
    parser = ArgumentParser(description="Collect traces and timing info")
    parser.add_argument(
        "--timeout",
        type=str,
        help="Timeout per script",
        default="25m",
    )
    parser.add_argument(
        "--sleep",
        type=int,
        help="Minutes to sleep between checking",
        default=2,
    )
    parser.add_argument(
        "--max_concurrent_processes",
        type=int,
        help="Max processes to run at once",
        default=3,
    )
    parser.add_argument(
        "--log",
        type=str,
        help="File to log execution info",
        default="kaggle_script_ids.log",
    )
    parser.add_argument(
        "--as_original",
        action="store_true",
        help="Run scripts without instrumentation for timing purposes",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory (optional, default is execution dir)",
    )
    parser.add_argument(
        "--script_ids_log",
        type=str,
        help="Path to log where we want to take script ids from"
    )
    parser.add_argument(
        "--script_ids",
        type=int,
        nargs="+",
        help="List of script IDs to run",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help=
        "Execute scripts in sequence, no concurrent (for debugging mainly)",
    )
    return parser.parse_args()


def main():
    # parameters for running kaggle scripts
    args = get_args()
    timeout = args.timeout
    sleep = args.sleep
    max_concurrent_processes = args.max_concurrent_processes
    log_file = args.log
    as_original = args.as_original
    output_dir = args.output_dir
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    script_ids_log = args.script_ids_log
    if script_ids_log is not None:
        script_ids_log = read_log(script_ids_log)
        script_ids = [k for k, v in script_ids_log.items() if v == 0]
    else:
        script_ids = None
    if args.script_ids is not None:
        script_ids = list(args.script_ids)

    collector = DataCollector(
        timeout=timeout,
        sleep=sleep,
        log_file=log_file,
        max_concurrent_scripts=max_concurrent_processes,
    )

    # number of scripts to collect for each kaggle project id
    scripts_per_project = 100
    # kaggle project ids
    project_ids = set([25, 70, 61, 13, 66, 29, 24, 12, 40])
    collector.collect_data(
        scripts_per_project,
        project_ids,
        script_ids=script_ids,
        as_original=as_original,
        output_dir=output_dir,
        sequential=args.sequential,
    )


if __name__ == "__main__":
    main()
