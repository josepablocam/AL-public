# Wrapper to execute a dynamically instrumented Kaggle script in the
# correct directory and save down results as a pickled list

from __future__ import print_function
import ast
import argparse
import os
import sys
sys.path.append('../')
import traceback
import time

from collection import errors
from instrumentation.instrument import DynamicInstrumentation
from synthesis.meta_features import DirectSummarizer
from instrumentation import transforms


def run_instrumented(code, _instr, instr_nm):
    try:
        instrumented, _ = transforms.compile_instrumented(instr_nm, code)
    except SyntaxError:
        print("Encountered syntax error consider retrying with python 2")
        sys.exit(errors.ERROR_COMPILE)
    exec(instrumented, {instr_nm: _instr})
    return _instr


def run_original(code):
    exec(code)


def log_time(file_name, dur):
    base_file, _ = os.path.splitext(file_name)
    time_file = base_file + "_time.txt"
    with open(time_file, "w") as fout:
        fout.write(str(dur))


def run(
        code,
        src_id,
        exec_dir,
        write,
        partial,
        as_original=False,
):
    print("Executing script: %d" % src_id)
    _instr = DynamicInstrumentation(
        relevant_modules=['sklearn', 'xgboost'],
        arg_summarizer=DirectSummarizer()
    )
    instr_nm = '_instr'

    if write:
        try:
            file_name = write % src_id
        except TypeError:
            file_name = write
    file_name = os.path.abspath(file_name)

    try:
        if not os.path.exists(exec_dir):
            print("Path not found:%s" % exec_dir)
            sys.exit(errors.ERROR_EXEC)
        os.chdir(exec_dir)
        start_time = time.time()
        if as_original:
            run_original(code)
            end_time = time.time()
            log_time(file_name, end_time - start_time)
        else:
            _instr = run_instrumented(code, _instr, instr_nm)
            end_time = time.time()
            log_time(file_name, end_time - start_time)
            if len(_instr.acc) == 0:
                print("No data collected")
                sys.exit(errors.ERROR_NO_DATA_COLLECTED)
            if write:
                _instr.flush(file_name)
            else:
                print(_instr.acc)
    except Exception:
        print("Failed executing script id: %d" % src_id)
        traceback.print_exc()
        if len(_instr.acc) > 0 and partial:
            if write:
                file_name += '-partial'
                _instr.flush(file_name)
            else:
                print('Partially collected trace:')
                print(_instr.acc)
        sys.exit(errors.ERROR_EXEC)


def main():
    argparser = argparse.ArgumentParser(
        description='Executes and instruments a Kaggle script'
    )
    argparser.add_argument(
        'code',
        type=str,
        help='String source code or file containing source code'
    )
    argparser.add_argument(
        '--file', action='store_true', help='Treat <code> as a file'
    )
    argparser.add_argument(
        '--dir',
        type=str,
        default='.',
        help='Directory to cd to before execution (default = .)'
    )
    argparser.add_argument(
        '--id', type=int, default=-1, help='Script id (default=-1)'
    )
    argparser.add_argument(
        '--write',
        type=str,
        nargs='?',
        const='script_%d.pkl',
        help=
        'Write pickled trace to disk using file name format (default=script_%%d.pkl)'
    )
    argparser.add_argument(
        '--partial',
        action='store_true',
        help='Write out partial traces if fails to complete'
    )
    argparser.add_argument(
        "--as_original",
        action="store_true",
        help="Run original programs (without instrumentation etc)"
    )
    args = argparser.parse_args()

    code = args.code
    if args.file:
        with open(code, 'r') as f:
            code = f.read()
    src_id = args.id
    exec_dir = args.dir
    write = args.write
    partial = args.partial
    as_original = args.as_original
    run(code, src_id, exec_dir, write, partial, as_original=as_original)


if __name__ == "__main__":
    main()
