
# Transform log files into csv files 

import os
import argparse
import pandas as pd


def get_proba_per_axis_repartition_of_predictions(table, axis, slices):
    axis_proba = (table[(table.axis == axis) & (table.pred > 0) & ([x in slices for x in table.slice])].shape[0]) / float(len(slices))
    axis_proba = round(axis_proba, 2)
    return axis_proba


def get_proba_per_axis_mean(table, axis, slices):
    axis_proba = table[(table.axis == axis) & ([x in slices for x in table.slice])]["pred"].mean()
    axis_proba = round(axis_proba, 2)
    return axis_proba


def get_proba_per_axis_median(table, axis, slices):
    axis_proba = table[(table.axis == axis) & ([x in slices for x in table.slice])]["pred"].median()
    axis_proba = round(axis_proba, 2)
    return axis_proba


def get_proba_per_axis(table, axis, slices, method):
    assert method in ["percentage", "mean", "median"], 'Unkown method. The argument method has only three modalities: percentage, mean or median.'
    if method == "mean":
        return get_proba_per_axis_mean(table, axis, slices)
    elif method == "median":
        return get_proba_per_axis_median(table, axis, slices)
    else:
        return get_proba_per_axis_repartition_of_predictions(table, axis, slices)


def get_results_locally(n, min_slices_min1, max_slices_plus1, axis_slices, table, axis, method):
    ### Tests
    assert n > 1, "n_areas must be lower than the number of slices of each axis; verify axis z."
    assert method in ["percentage", "mean", "median"], 'Unkown method. The argument method has only three modalities: percentage, mean or median.'
    ### Function
    results_locally = {"slice_start": [], "slice_end": [], "proba": []}
    i = min_slices_min1
    if method == "percentage":
        while (i+n) < (max_slices_plus1-1):
            slices = [x for x in axis_slices if ((x > i) & (x < (i+n+1)))]
            results_locally["proba"].append(get_proba_per_axis_repartition_of_predictions(table, axis, slices))
            results_locally["slice_start"].append(i+1)
            results_locally["slice_end"].append(i+n)
            i+=n
        slices = [x for x in axis_slices if ((x > i) & (x < max_slices_plus1))]
        results_locally["proba"].append(get_proba_per_axis_repartition_of_predictions(table, axis, slices))
        results_locally["slice_start"].append(i+1)
        results_locally["slice_end"].append(max_slices_plus1-1)
    elif method == "mean":
        while (i+n) < (max_slices_plus1-1):
            slices = [x for x in axis_slices if ((x > i) & (x < (i+n+1)))]
            results_locally["proba"].append(get_proba_per_axis_mean(table, axis, slices))
            results_locally["slice_start"].append(i+1)
            results_locally["slice_end"].append(i+n)
            i+=n
        slices = [x for x in axis_slices if ((x > i) & (x < max_slices_plus1))]
        results_locally["proba"].append(get_proba_per_axis_mean(table, axis, slices))
        results_locally["slice_start"].append(i+1)
        results_locally["slice_end"].append(max_slices_plus1-1)
    else:
        while (i+n) < (max_slices_plus1-1):
            slices = [x for x in axis_slices if ((x > i) & (x < (i+n+1)))]
            results_locally["proba"].append(get_proba_per_axis_median(table, axis, slices))
            results_locally["slice_start"].append(i+1)
            results_locally["slice_end"].append(i+n)
            i+=n
        slices = [x for x in axis_slices if ((x > i) & (x < max_slices_plus1))]
        results_locally["proba"].append(get_proba_per_axis_median(table, axis, slices))
        results_locally["slice_start"].append(i+1)
        results_locally["slice_end"].append(max_slices_plus1-1)
    return pd.DataFrame(results_locally)


def get_predictions_from_log(directory, subid, logfile, method="percentage", n_areas=3):
    '''
    directory: input directory
    logfile: name of the log file
    method: method of calculation of the probabilities
    - percentage: percentage of slices predicted 1
    - mean: mean of predictions on each slice
    - median: median of predictions on each slice
    n_areas: integer to sample each axis into n areas, and computes the probability and prediction of class in each area
    '''
    #directory = os.path.join(indir, subid)
    filepath = os.path.join(directory, logfile)
    ### Tests
    # is directory existing
    assert os.path.exists(directory), 'The directory ' + directory + ' does not exist.'
    assert os.path.isdir(directory), directory + ' is not a directory.'
    # is logfile existing
    assert os.path.exists(filepath), 'The file ' + filepath + ' does not exist.'
    assert os.path.isfile(filepath), filepath + ' is not a file.'
    # is method in percentage, mean, median
    assert method in ["percentage", "mean", "median"], 'Unkown method. The argument method has only three modalities: percentage, mean or median.'
    # is n_areas an integer
    assert isinstance(n_areas, int), 'Please verify n_areas is an integer number.'
    ### Function
    # Get predictions at a global scale
    table = {"pred": [], "slice": [], "axis": []}
    with open(filepath, "r") as f: 
        for l in f.readlines():
            # get table
            if "0 (" in l:
                ls = ((l.split("0 (")[-1]).split(")")[0]).split(",")
                table["pred"].append(int(ls[0]))
                table["slice"].append(int(ls[1]))
                alphanumeric = [character for character in ls[2] if character.isalnum()]
                alphanumeric = "".join(alphanumeric)
                table["axis"].append(alphanumeric)
            # get time
            if "Processing time:" in l:
                t = round(float(l.split(": ")[1]), 2)
            # # get slices predicted 1:
            # if "Slices predicted 1:" in l:
            #     global_slices = l.split(":")[1]
            # # get corresponding axes
            # if "Corresponding axes:" in l:
            #     global_axes = l.split(":")[1]
            # get predicted median
            if method == "mean":
                if "Predicted mean:" in l:
                    proba = round(float(l.split(": ")[1]), 2)
            # get predicted mean
            elif method == "median":
                if "Predicted median:" in l:
                    proba = round(float(l.split(": ")[1]), 2)
            # get percentage of slices predicted 1
            else:
                if "Repartition of predictions:" in l:
                    proba = round(float((l.split("; ")[1]).split(" ")[0])/100, 2)
    table = (pd.DataFrame(table)).sort_values(["axis", "slice"]).reset_index(drop=True)
    global_slices = table[table.pred == 1].slice.tolist()
    global_axes = table[table.pred == 1].axis.tolist()
    # Get predictions at a local scale
    ### x
    x_slices = table[(table.axis == "x")].slice
    min_slices_min1 = x_slices.min() - 1
    max_slices_plus1 = x_slices.max() + 1
    slices = [x for x in x_slices if ((x > min_slices_min1) & (x < max_slices_plus1))]
    x_proba = get_proba_per_axis(table, 'x', slices, method)
    # by area
    n = int((max_slices_plus1-1)/n_areas)
    x_results_locally = get_results_locally(n, min_slices_min1, max_slices_plus1, x_slices, table, "x", method)
    ### y
    y_slices = table[(table.axis == "y")].slice
    min_slices_min1 = y_slices.min() - 1
    max_slices_plus1 = y_slices.max() + 1
    slices = [x for x in y_slices if ((x > min_slices_min1) & (x < max_slices_plus1))]
    y_proba = get_proba_per_axis(table, 'y', slices, method)
    # by area
    n = int((max_slices_plus1-1)/n_areas)
    y_results_locally = get_results_locally(n, min_slices_min1, max_slices_plus1, y_slices, table, "y", method)
    ### z
    z_slices = table[(table.axis == "z")].slice
    min_slices_min1 = z_slices.min() - 1
    max_slices_plus1 = z_slices.max() + 1
    slices = [x for x in z_slices if ((x > min_slices_min1) & (x < max_slices_plus1))]
    z_proba = get_proba_per_axis(table, 'z', slices, method)
    # by area
    n = int((max_slices_plus1-1)/n_areas)
    z_results_locally = get_results_locally(n, min_slices_min1, max_slices_plus1, z_slices, table, "z", method)
    # Returns dictionary and table
    return [{"subid": [subid], "proba": [proba], "t": [t], "slices": [global_slices], "axes": [global_axes], 
            "x_proba": [x_proba], "y_proba": [y_proba], "z_proba": [z_proba]}, 
            table, x_results_locally, y_results_locally, z_results_locally]


def compute_proba_to_pred(proba):
    return 1 if proba > 0.5 else 0


if __name__ == '__main__':
    # Define inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('-indir', nargs=1, type=str)
    parser.add_argument('-subid', nargs=1, type=str)
    parser.add_argument('-method', nargs=1, type=str, default="percentage")
    parser.add_argument('-n_areas', nargs=1, type=str, default="3")
    # Load inputs
    args = parser.parse_args()
    indir = args.indir[0]
    subid = args.subid[0]
    method = args.method[0]
    n_areas = int(args.n_areas[0])
    # Get predictions from log file
    tot_preds, table, x_results_locally, y_results_locally, z_results_locally = get_predictions_from_log(indir, subid, "global_analysis.log", method, n_areas)
    tot_df = pd.DataFrame(tot_preds)
    tot_df["pred"] = tot_df["proba"].apply(compute_proba_to_pred) 
    tot_df["x_pred"] = tot_df["x_proba"].apply(compute_proba_to_pred) 
    tot_df["y_pred"] = tot_df["y_proba"].apply(compute_proba_to_pred) 
    tot_df["z_pred"] = tot_df["z_proba"].apply(compute_proba_to_pred)
    x_results_locally["pred"] = x_results_locally["proba"].apply(compute_proba_to_pred) 
    y_results_locally["pred"] = y_results_locally["proba"].apply(compute_proba_to_pred) 
    z_results_locally["pred"] = z_results_locally["proba"].apply(compute_proba_to_pred)  
    # Save in a csv table
    tot_df.to_csv(os.path.join(indir, "tot_df.csv"), index=False)
    table.to_csv(os.path.join(indir, "table.csv"), index=False)
    x_results_locally.to_csv(os.path.join(indir, "x_results_locally.csv"), index=False)
    y_results_locally.to_csv(os.path.join(indir, "y_results_locally.csv"), index=False)
    z_results_locally.to_csv(os.path.join(indir, "z_results_locally.csv"), index=False)
