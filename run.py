#!/usr/bin/env python3

import argparse
import os
import subprocess
from glob import glob
import pandas as pd

__version__ = "1.0.0"

def run(command, env={}):
    merged_env = os.environ.copy()
    merged_env.update(env)
    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, shell=True,
                               env=merged_env)
    while True:
        line = process.stdout.readline()
        line = str(line, 'utf-8')[:-1]
        print(line)
        if line == '' and process.poll() is not None:
            break
    if process.returncode != 0:
        raise Exception("Non-zero return code: %d" % process.returncode)

parser = argparse.ArgumentParser(description='BIDS App processing script.')
parser.add_argument('bids_dir', help='The directory with the input dataset formatted according to the BIDS standard.')
parser.add_argument('output_dir', help='The directory where the output files should be stored.')
parser.add_argument('analysis_level', choices=['participant', 'group'], 
                    help='Level of analysis to perform (participant or group).')
parser.add_argument('--participant_label', nargs='+',
                    help='Space-separated list of participant labels to analyze (without "sub-"). If not provided, all participants are analyzed.')
parser.add_argument('--pythonpath', default='venv/bin', help='Path to Python environment.')
parser.add_argument('--gpuid', default='0,1', help='Comma-separated list of GPU IDs.')
parser.add_argument('--masks', default='0', help='0: return log file; 1: predict and save prototypes.')
parser.add_argument('--pred_method', default='percentage',
                    help='Method for prediction: "percentage", "mean", or "median".')
parser.add_argument('--n_areas', default='3', help='Number of areas to sample each axis into.')
parser.add_argument('--modeldir', default='./saved_models/resnet152/19112020/',
                    help='Directory where the model weights and prototypes are stored.')

args = parser.parse_args()

subjects_to_analyze = []
# Determine subjects to analyze
if args.participant_label:
    subjects_to_analyze = args.participant_label
else:
    subject_dirs = glob(os.path.join(args.bids_dir, "sub-*"))
    subjects_to_analyze = [os.path.basename(subject_dir).split("-")[-1] for subject_dir in subject_dirs]

# Process participants
if args.analysis_level == "participant":
    for subject_label in subjects_to_analyze:
        # Find all sessions for the participant
        session_dirs = glob(os.path.join(args.bids_dir, f"sub-{subject_label}", "ses-*"))
        if not session_dirs:  # Handle single-session datasets
            session_dirs = [os.path.join(args.bids_dir, f"sub-{subject_label}")]
        
        for session_dir in session_dirs:
            session = os.path.basename(session_dir).split("-")[-1] if "ses-" in session_dir else "single_session"
            anat_dir = os.path.join(session_dir, "anat")
            if not os.path.isdir(anat_dir):
                continue
            
            # Process T1w images in the anat directory
            # multi-run case
            #T1_files = glob(os.path.join(anat_dir, "*_run-*_T1w.nii*"))
            #if not T1_files:
            T1_files = glob(os.path.join(anat_dir, "*_T1w.nii*"))
            for T1_file in T1_files:
                filename = os.path.basename(T1_file)
                run_dir = [el for el in os.path.basename(T1_file).split('_') if 'run-' in el]
                if not run_dir:
                    run_dir = 'single_run'
                else:
                    run_dir = run_dir[0]
                output_subdir = os.path.join(args.output_dir, f"sub-{subject_label}", f"ses-{session}", run_dir)
                os.makedirs(output_subdir, exist_ok=True)
                
                cmd = f"./preprocess_and_predict.sh {T1_file} {filename} sub-{subject_label} {output_subdir} {args.pythonpath} {args.gpuid} {args.masks} {args.pred_method} {args.n_areas} {args.modeldir}"
                run(cmd)

# Group-level analysis
elif args.analysis_level == "group":
    df_all = pd.DataFrame({"subid": [], "session": [], "run": [], "proba": [], "t": [], "pred": []})
    for subject_label in subjects_to_analyze:
        session_dirs = glob(os.path.join(args.output_dir, f"sub-{subject_label}", "ses-*"))
        if not session_dirs:  # Handle single-session datasets
            session_dirs = [os.path.join(args.output_dir, f"sub-{subject_label}", "single_session")]
        for session_dir in session_dirs:
            if os.path.isdir(session_dir):
                run_dirs = glob(os.path.join(session_dir, "run-*"))
                if not run_dirs:
                    run_dirs = [os.path.join(session_dir, "single_run")]
                for run_dir in run_dirs:
                    results_file = os.path.join(run_dir, "tot_df.csv")
                    if os.path.exists(results_file):
                        df = pd.read_csv(results_file)
                        df['subid'] = f"sub-{subject_label}"
                        df['session'] = os.path.basename(session_dir)
                        df['run'] = os.path.basename(run_dir)
                        df_all = pd.concat([df_all, df], axis=0, ignore_index=True, sort=False)
    df_all.to_csv(os.path.join(args.output_dir, "group_results.csv"), index=False)
