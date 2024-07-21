import pandas as pd
import numpy as np
import os.path
from math import floor, log10

"""
GENERAL UTILITY (I.E., PRINTING, SAVING, ETC.)
"""
# Save results to a specified folder
def save_results(solutions, solutions2 = []):
    print("In order to plot or export results, they must be saved.")
    if not os.path.exists("./Results"):
        while True:
            print("No Results folder exists at the current file level in which to save results.")
            makeResults = str(input("Create Results folder? (Y/N) "))
            if makeResults == "Y":
                os.makedirs("./Results")
                break
            elif makeResults == "N":
                print("Results cannot be saved.", "\n")
                return
            else:
                print("Invalid response. Please re-enter.")
    valid = False
    while not valid:
        temp = str(input("Would you like to save results to a .TXT file? (Y/N) "))
        if (temp == "Y"):
            subFName = ""
            while True:
                subF = str(input("Save to Results (Y) or Results subfolder? Enter Y or subfolder name: "))
                if subF == "Y":
                    print("File(s) will save to Results folder.")
                    break
                elif os.path.exists("./Results/" + subF):
                    subFName = subF + "/"
                    break
                else:
                    crSubF = str(input(("Subfolder does not exist. Create one? (Y/N) ")))
                    if crSubF == "Y":
                        os.makedirs("./Results/" + subF)
                        subFName = subF + "/"
                        break
                    elif crSubF == "N":
                        print("File(s) will save to Results folder.")
                        break
                    else:
                        print("Invalid response. Please re-enter.")

            fileName = str(input("Enter a file name without the .txt ending (NO SPACES): "))
            print("")
            if (len(fileName) <= 4) or (fileName[len(fileName)-4:] != ".txt"):
                if solutions2 != []:
                    fileName2 = fileName + "_INTENSITY.txt"
                    fileName += "_PRESSURE.txt"
                else:
                    fileName += ".txt"
            elif (fileName[len(fileName)-4:] == ".txt") and (solutions2 != []):
                fileName2 = fileName[:len(fileName)-4] + "_INTENSITY.txt"
                fileName = fileName[:len(fileName)-4] + "_PRESSURE.txt"
            f = open("./Results/" + subFName + fileName, "w")
            string_res = ""
            cols = len(solutions[0])
            for line in solutions:
                for val in range(cols):
                    if (val == cols - 1):
                        string_res += str(line[val]) + "\n"
                    else:
                        string_res += str(line[val]) + " "
            f.write(string_res)
            f.close()
            if (solutions2 != []):
                f = open("./Results/" + subFName + fileName2, "w")
                string_res = ""
                cols = len(solutions2[0])
                for line in solutions2:
                    for val in range(cols):
                        if (val == cols - 1):
                            string_res += str(line[val]) + "\n"
                        else:
                            string_res += str(line[val]) + " "
                f.write(string_res)
                f.close()
            savedFolder = "Results"
            if subFName != "":
                savedFolder = subFName[:-1]
            if (solutions2 != []):
                print("Pressure results were saved to " + fileName + " in the " + savedFolder + " folder.")
                print("Matching intensity locations were saved to " + fileName2 + " in the " + savedFolder + " folder.")
            else:
                print("Results were saved to " + fileName + " in the " + savedFolder + " subfolder.")
            valid = True
        elif (temp == "N"):
            valid = True
        else:
            print("Invalid response. Please enter Y or N.")
    print("")
    return

# Determines number of frequency samples
def f_step(dfN, xSteps):
    fstep = int(len(dfN.index)/xSteps)
    print("Number of Frequency Samples: ", fstep, "\n")
    return fstep

# Determines number of x samples per frequency.
def x_step(dfN, maxX):
    count, idx = 0, 0
    # Loop through first frequency sample
    threshold = abs(dfN.iat[idx, 0] - dfN.iat[idx + 1, 0]) * 2
    height = len(dfN.index)
    while True:
        """if idx == height - 1:
            count += 1
            break
        elif abs(dfN.iat[idx, 0] - dfN.iat[idx + 1, 0]) > threshold:
            count += 1
        idx += 1"""
        if (abs(float(dfN.iat[count, 0]) - float(dfN.iat[count+1, 0])) > threshold) or\
            (float(dfN.iat[count + 1, 0]) - float(dfN.iat[count, 0]) < 0):
            count += 1
            break
        count += 1
    print()
    print("X Samples per Frequency: ", count)
    return count

# Determines number of x samples per frequency for each Point Source
def x_stepPS(x_locs, fStep):
    eps = 0.01   # May not be exactly equal, so we test with some epsilon
    idx, count, f_count = 0, 0, 0 # count tracks number of steps per frequency, f_count tracks number of frequencies in each ps
    threshold = (x_locs[1] - x_locs[0]) * 2
    #threshold = (maxX-minX)/2

    xS = []
    while True:
        count += 1
        if idx == len(x_locs) - 1:
            xS.append(count)
            break
        if (abs(x_locs[idx] - x_locs[idx+1]) > threshold):   # Reset after each frequency
            f_count += 1
            if (f_count == fStep):  # Reached the end of this PS location. Hard reset
                threshold = (x_locs[idx + 2] - x_locs[idx + 1]) * 2
                xS.append(count)
                f_count = 0
            count = 0
        idx += 1
    print()
    print("X Samples per Frequency: ", str(xS))
    return xS

# Helper function to print out lists of data to terminal
def helperPrint(solutions, isIntensity, gap_size, minF, minPS):
    cluster_count, counter = 0, 0
    prev_f, prevPS = minF, solutions[0][0]
    if gap_size == -1:
        counter = 0
    gap_f = minF
    cluster_count += 1
    for sol_idx in range(len(solutions)):
        sol = solutions[sol_idx]
        if sol[0] >= 0:
            ps_print = str(sol[0])
        else:
            ps_print = "---"
        if sol[0] != solutions[sol_idx-1][0] and sol_idx != 0 and minPS != -1:
            print("/////////////////////////////////")
            prev_f = sol[1]
            cluster_count += 1
            counter = 0
        elif (gap_size == -1):
            if (counter == 2):
                print("--------------------")
                counter = 0
        elif ((sol[1]-gap_f > gap_size) and (sol_idx != 0)):
            print("--------------------")
            prev_f = sol[1]
            cluster_count += 1
        elif (sol[1] != prev_f) and (gap_f != -1):
            print("")
            prev_f = sol[1]
        gap_f = sol[1]
        if (not isIntensity):
            if (sol[2] == -1):
                l_loc, r_loc, par, conf = "--", "--", "--", "--"
            else:
                l_loc, r_loc, conf = str(sol[2]), str(sol[3]), str(sol[5])
                if (sol[4] == -1):
                    par = "--"
                elif (sol[4] == 0):
                    par = "even"
                else:
                    par = "odd"
        else:
            l_loc, r_loc, par, conf = sol[2], sol[3], "", ""
        print(ps_print, ":", sol[1], ": ", l_loc, " ", r_loc, par, conf)
        counter += 1
    return cluster_count

# Utility function to convert a pandas dataframe to a nested list
def convert_to_list(df):
    result = []
    for r in range(len(df.index)):
        temp = dfRow_to_list(df, r)
        result.append(temp)
    return result

# Function which merges two intensity or two pressure results dataframes and returns the result as a nested list
def merge_lists(df1, df2):
    idx1, idx2 = 0, 0
    result = []
    while True:
        if idx1 >= len(df1.index):
            for i in range(idx2, len(df2.index)):
                temp = dfRow_to_list(df2, i)
                result.append(temp)
            break
        elif idx2 == len(df2.index):
            for i in range(idx1, len(df1.index)):
                temp = dfRow_to_list(df1, i)
                result.append(temp)
            break

        if df1.iat[idx1, 0] < df2.iat[idx2, 0]:
            temp = dfRow_to_list(df1, idx1)
            result.append(temp)
            idx1 += 1
        elif df1.iat[idx1, 0] > df2.iat[idx2, 0]:
            temp = dfRow_to_list(df2, idx2)
            result.append(temp)
            idx2 += 1
        else:
            if df1.iat[idx1, 1] < df2.iat[idx2, 1]:
                temp = dfRow_to_list(df1, idx1)
                result.append(temp)
                idx1 += 1
            elif df1.iat[idx1, 1] > df2.iat[idx2, 1]:
                temp = dfRow_to_list(df2, idx2)
                result.append(temp)
                idx2 += 1
            
    return result

# Converts a single row of a dataframe into a list
def dfRow_to_list(df, r):
    temp = []
    for c in range(len(df.columns)):
            temp.append(df.iat[r, c])
    return temp



"""
DATA UTILITY METHODS (I.E., SORTING, MATH, ETC.)
"""
# Helper function to find maximum in an array by analyzing a sublist index
def find_max(arr, sub_idx = -1):
    if sub_idx != -1:
        max = arr[0][sub_idx]
        idx = 0
        for i in range(1, len(arr)):
            if arr[i][sub_idx] > max:
                max = arr[i][sub_idx]
                idx = i
    else:
        max = arr[0]
        idx = 0
        for i in range(1, len(arr)):
            if arr[i] > max:
                max = arr[i]
                idx = i
    return idx, max
# Helper function to find minimum in an array by analyzing a sublist index
def find_min(arr, sub_idx = -1):
    if sub_idx != -1:
        min = arr[0][sub_idx]
        idx = 0
        for i in range(1, len(arr)):
            if arr[i][sub_idx] < min:
                min = arr[i][sub_idx]
                idx = i
    else:
        min = arr[0]
        idx = 0
        for i in range(1, len(arr)):
            if arr[i] < min:
                min = arr[i]
                idx = i
    return idx, min

# Merge Sort implementation that compares values in sublists at sublist index sub_ind
def pair_merge(sub, sub_ind):
    if (len(sub) <= 1):
        return sub
    else:
        m = len(sub)//2
        L = pair_merge(sub[:m], sub_ind)
        R = pair_merge(sub[m:], sub_ind)
        l_ind, r_ind = 0, 0
        result = []
        for i in range(len(sub)):
            if l_ind >= len(L):
                for j in R:
                    result.append(j)
                break
            elif r_ind >= len(R):
                for j in L:
                    result.append(j)
                break
            elif L[l_ind][sub_ind] <= R[r_ind][sub_ind]:
                result.append(L[l_ind])
                l_ind += 1
            else:
                result.append(R[r_ind])
                r_ind += 1
        return result
    
# Find best match for each individual peak. Delete redundant values
def pair_sift(pairs, ex_idx):
    result = []
    selected_l = []
    # Iterate through all left locations
    for l in range(len(pairs)):
        l_val = pairs[l][ex_idx]
        # If left location not processed yet, add to list and process
        if l_val not in selected_l:
            selected_l.append(l_val)
            l_min = [l, pairs[l][2]]    # Index of current best match, and the corresponding penalty.
            # Iterate over remaining left locations looking for lowest penalty with matching location.
            for r in range(l+1, len(pairs)):
                if (pairs[r][ex_idx] == l_val) and (pairs[r][2] < l_min[1]):
                    l_min = [r, pairs[r][2]]
            result.append(pairs[l_min[0]])
    return result

# Check if a point is a global peak by checking general trend in nearby slopes.
def is_global_peak(f_range, peak_idx, depth_check):
    is_peak = 0
    if (f_range[peak_idx + 1] - f_range[peak_idx] > 0) == (f_range[peak_idx] - f_range[peak_idx - 1] > 0):
        return False
    leftSide = np.sign(f_range[peak_idx] - f_range[peak_idx - 1])
    rightSide = np.sign(f_range[peak_idx + 1] - f_range[peak_idx])
    currL, currR = 0, 0
    for i in range(1, depth_check+1):
        currL, currR = np.sign(f_range[peak_idx - i] - f_range[peak_idx - i - 1]), np.sign(f_range[peak_idx + i + 1] - f_range[peak_idx + i])
        if currL == leftSide:
            is_peak += 1
        else:
            is_peak -= 1
        if currR == rightSide:
            is_peak += 1
        else:
            is_peak -= 1
    if is_peak > 0:
        return True
    else:
        return False

# Function which returns the log10 of a value. Useful for determining how precisely to round values.
def decimal_precision(val):
    return abs(floor(log10(val)))
    