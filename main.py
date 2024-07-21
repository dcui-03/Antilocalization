import pandas as pd
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from utils import *
from userInput import *

# Sort for numPeaks amount of local maxima in a specified half. Returns the locations of these local maxima in an array
# Also filters out particular peaks if they are too low
def peak_sorts(local_half, numPeaks, heightW, overall_avg):
    avg_height = sum(local_half)/len(local_half)
    padding = int(len(local_half) * 1/6)             # Boundary for valid results (tune padding as desired)
    maxima = [[-1, -1]]*numPeaks
    curr_max = [-1, -1]                         # Maximum peak of local region
    is_above = local_half[0] >= avg_height      # Regions are defined by being above the average height
    temp = 0
    
    local_peaks = []
    for j in range(1, len(local_half) - 1):
        current = local_half[j]
        if (current >= avg_height) and (is_above == False):  # Starting a new local region
            is_above = True
        elif (current < avg_height) and (is_above == True): # End of local region
            if (curr_max[0] != -1) and (curr_max[0] >= padding) and (curr_max[0] <= len(local_half)-padding): # Save our recorded current max
                local_peaks.append(curr_max)
                temp += 1
            is_above = False
            curr_max = [-1, -1]

        # Within a region, test to see if is greater than current max
        if (is_above):
            if (current > curr_max[1]) and (current >= heightW*avg_height) and (current > overall_avg):
                curr_max = [j, current]                     # Overwrite current local max

    # We now have a set of peaks that are above the threshold. We filter for the top num_peaks.
    if len(local_peaks) > numPeaks:
        for peak in local_peaks:
            idx, min = find_min(maxima, 1)
            if peak[1] >= min:
                maxima[idx] = peak
    else:
        maxima = local_peaks
    
    # Copy non-empty values into final list
    maxima2 = []
    for val in maxima:
        if val[1] != -1:
            maxima2.append(val)

    return maxima2

# Driver code to process each frequency's data
# Returns a list containing selected peaks for the frequency
def f_intensity_process(f_range, numPeaks, heightW, symW):
    overall_avg = sum(f_range)/len(f_range)
    # Find the highest peaks on left and right sides.
    halfway = len(f_range)//2
    half_len = len(f_range) - halfway
    quarter_len = half_len / 2
    l_maxima = peak_sorts(f_range[:halfway], numPeaks, heightW, overall_avg)
    # Right side peaks will usually be lower, so our height weight is weaker
    r_maxima = peak_sorts(f_range[halfway:], numPeaks, heightW, overall_avg)
    # Early reject if there are no valid maxima to compare
    if (l_maxima == []) or (r_maxima == []):
        return []
    
    # Compare the peaks using the distance and symmetry weights (O(n**2))
    # We store only the top 3 matches, which we can later visually confirm
    max_val, max_idx = float('inf'), 0
    top_pairs = [[-1, -1, max_val]] * 3

    for l in l_maxima:
        for r in r_maxima:
            # Distance penalty is the sum of the squared distances from the quarter location
            distPenalty = (l[0]-quarter_len)**2 + (r[0]-quarter_len)**2
            
            # Symmetry penalty is simply squared distance between the two index locations
            symPenalty = ((half_len - r[0]) - l[0])**2
            # Padding for symmetry
            if abs((half_len - r[0]) - l[0]) <= (len(f_range)//35):
                curr_penalty = distPenalty + symW * symPenalty
                if curr_penalty < max_val:
                    max_idx, max_val = find_max(top_pairs, 2)
                    top_pairs[max_idx] = [l[0], halfway + r[0], curr_penalty]
    # Take only top matches if they exist.
    top_pairs2 = []      
    for val in top_pairs:
        if val[0] != -1:
            top_pairs2.append(val)
    # Use merge sort to sort by ascending penalty
    top_pairs2 = pair_merge(top_pairs2, 2)
    # Sift for redundant values (We want bijective results)
    top_pairs2 = pair_sift(top_pairs2, 0)
    top_pairs2 = pair_sift(top_pairs2, 1)

    return top_pairs2

# Intensity Processing
def intensity(dfN, minF):
    # Get some user input
    numPeaks, heightW, symW = user_inputI()

    sol_count = 0
    solutions = []
    indices = []

    if (minPS == -1):   # For the plane wave case
        # Call helper function to process each frequency
        for n in range(fSteps):
            freq = round(minF + n * fStepSize, fStepScale)

            print("Working on frequency ", n+1, " of ", fSteps, end='\r')
            f_range = np.array(dfN["I"])[n*xSteps:(n+1)*xSteps]
            # If a solution is produced, increment counter and add a row to our solutions list
            n_matches = f_intensity_process(f_range, numPeaks, heightW, symW)
            if len(n_matches) != 0:
                sol_count += 1
                for match in n_matches:
                    solutions.append([-1, freq, minX + xStepSize*match[0], minX + xStepSize*match[1]])
        indices = [0]*len(solutions)
    else:   # POINT SOURCE
        running_idx = 0
        # Call helper function to process each frequency
        for i in range(len(xSteps)):
            for n in range(fSteps):
                freq = round(minF + n * fStepSize, fStepScale)

                print("Working on Point Source ", i+1, " of ", len(xSteps), end='\r')
                f_range = np.array(dfN["I"])[running_idx + n*xSteps[i]:running_idx + (n+1)*xSteps[i]]
                # If a solution is produced, increment counter and add a row to our solutions list
                n_matches = f_intensity_process(f_range, numPeaks, heightW, symW)
                if len(n_matches) != 0:
                    sol_count += 1
                    for match in n_matches:
                        solutions.append([round(minPS + i*PSStepSize, PSStepScale), freq, minX + xStepSize[i]*match[0], minX + xStepSize[i]*match[1]])
                        indices.append([running_idx, i, n, match[0], match[1]])
            running_idx += xSteps[i] * fSteps
    
    print("\n")

    # Ask user if they would like to print the results
    if (print_results()):
        # Specify a gap size between clusters 
        print("For each frequency, matches are listed in descending fit.", "\n")
        if (sol_count != 0):
            gap_size = 1.5
            print("                 RESULTS")

            print("PS : Freq : Left Location      Right Location")
            print("--------------------------------------------------")
            cluster_count = helperPrint(solutions, True, gap_size, minF, minPS)
        # Take note of solution clusters!
        print("\n")

        # Print out final results
        if (sol_count == 1):
            print(sol_count, "frequency found.")
        else:
            print(sol_count, "frequencies found.")
            if (sol_count != 0):
                print(cluster_count, "clusters in interval " + str(gap_size), "\n")

    # Give option to save results to a txt file
    if (sol_count != 0):
        save_results(solutions)
    else:
        print("No results available to save.")
    print("Intensity Processing Completed.", "\n")
    return solutions, indices

# Rough parity check by applying a weighted sum of different tests
def parity_check(f_range, n = 15, ps_loc = -1):
    if len(f_range)%2 == 0:
        halfwayr = len(f_range)//2
        halfwayl = halfwayr-1
    else:
        halfwayr = len(f_range)//2
        halfwayl = halfwayr
    range_avg = sum(f_range)/len(f_range)
    abs_range_avg = sum([abs(val) for val in f_range])/len(f_range)
    t1, t2, t3, t4, t5, t6, t7 = 0.25, 0.25, 0.2, 0.7, 0.4, n*0.025, 0.2
    
    curr_peakLeft, lPeak_idx = 0, 0
    curr_peakRight, rPeak_idx = 0, 0
    l_zero, r_zero = 0, 0
    max_peak = 0
    idx_counter = 0
    match = 0
    
    # TEST 1: Iterate until we get the nearest peaks from the left and right of the center.
    while True:
        if (curr_peakLeft != 0) and (curr_peakRight != 0) and ((np.sign(curr_peakLeft) != np.sign(l_val)) and (np.sign(curr_peakRight) != np.sign(r_val))):
            l_zero, r_zero = halfwayl - (idx_counter-1), halfwayr + (idx_counter-1)
            break
        elif idx_counter >= len(f_range)//60:
            break
        l_val, r_val = f_range[halfwayl-idx_counter], f_range[halfwayr + idx_counter]
        if abs(l_val) > max_peak:
            max_peak = abs(l_val)
        elif abs(r_val) > max_peak:
            max_peak = abs(r_val)
        if (abs(l_val) > abs(curr_peakLeft)) and (is_global_peak(f_range, halfwayl - idx_counter, 20)):
            curr_peakLeft = l_val
            lPeak_idx = idx_counter
        if (abs(r_val) > abs(curr_peakRight)) and (is_global_peak(f_range, halfwayl + idx_counter, 20)):
            curr_peakRight = r_val
            rPeak_idx = idx_counter
        idx_counter += 1
    
    if (curr_peakLeft == 0) or (curr_peakRight == 0):
        return -1, -1   # Handles edge case
    # Test if there is a peak near the center. If so, then there is likely an even func.
    if (abs(rPeak_idx - lPeak_idx) > 15) and ((rPeak_idx <= len(f_range)/90) or (lPeak_idx <= len(f_range)/90)) and (max_peak > abs_range_avg) :
        match += 0.6
    elif np.sign(curr_peakLeft) != np.sign(curr_peakRight):
        match -= t1
    else:
        match += t1

    # TEST 2: Check slope at nearest 0's from center
    l_slope, r_slope = f_range[l_zero + 1] - f_range[l_zero], f_range[r_zero - 1] - f_range[r_zero]
    if (np.sign(l_slope) != np.sign(r_slope)):  # ODD
        match -= t2
    else:   # EVEN
        match += t2

    # TEST 3: Find greatest peaks and compare their signs
    templ, tempr = [abs(val) for val in f_range[:halfwayr]], [abs(val2) for val2 in f_range[halfwayr:]]
    maxlidx, maxl = find_max(templ, -1)
    maxridx, maxr = find_max(tempr, -1)
    maxl, maxr = f_range[maxlidx], f_range[halfwayr + maxridx]
    m0 = (np.sign(maxl)) == (np.sign(maxr))

    if m0 == True:
        match += t3
    else:
        match -= t3

    # TEST 4: End behavior
    ss, se = np.sign(f_range[1] - f_range[0]), np.sign(f_range[-2] - f_range[-1])
    ss2, se2 = np.sign(f_range[2] - f_range[1]), np.sign(f_range[-3] - f_range[-2])
    ss3, se3 = np.sign(f_range[3] - f_range[2]), np.sign(f_range[-4] - f_range[-3])
    stotal, sl, sr = 0, [ss, ss2, ss3], [se, se2, se3]
    for i in range(len(sl)):
        if sl[i] == sr[i]:
            stotal += 1/(i+2)
        else:
            stotal -= 1/(i+2)
    if stotal < 0:
        match -= (12*stotal/(13)) * t4
    else:
        match += (12*stotal/(13)) * t4

    # TEST 5: Test slopes at some contributor point or at the point source
    if ps_loc == -1:
        ps_idx = int(round(0.5 * (float(len(f_range))/30)))
    else:
        ps_idx = halfwayl - int(round(ps_loc * float(len(f_range))/30))
    if (np.sign(f_range[halfwayl - ps_idx] - range_avg) != np.sign(f_range[halfwayr + ps_idx] - range_avg)):
        match -= t5
    else:
        match += t5

    # TEST 6: Take slopes at a set of random contribution points
    l_list = []
    for i in range(n):
        while True:
            next_val = random.randrange(halfwayr)
            if next_val not in l_list:
                l_list.append(next_val)
                break
    for l in l_list:
        if (np.sign(f_range[l] - range_avg) == np.sign(f_range[l-1] - range_avg)):
            match += t6/n
        else:
            match -= t6/n
    
    # TEST 7: Compare averages for each half
    l_avg, r_avg = sum(f_range[:halfwayl+1])/len(f_range[:halfwayl+1]), sum(f_range[halfwayr:])/len(f_range[halfwayr:])
    if np.sign(l_avg) == np.sign(r_avg):
        match += t7
    else:
        match -= t7

    even_total = sum([t1, t2, t3, t4, t5, t6, t7])
    odd_total = -1 * sum([t1, t2, t3, t4, t5, t6, t7])

    if match == 0:
        return -1, -1
    if match > 0:
        if match > even_total:
            conf = 1
        else:
            conf = round(match/even_total, 3)
        return 0, conf
    return 1, round(match/odd_total, 3)

# Code to process each frequency's pressure
def f_pressure_process(f_range, lidx, ridx, parity, heightW, distW):
    temp = [abs(val) for val in f_range]
    overall_avg = sum(temp)/len(temp)
    
    subrange = int(len(f_range)//(distW*2))
    sol = [[-1, 0]]*2

    idx = [lidx, ridx]

    # For the left and right locations, find a matching peak or valley in the subrange.
    curr_idx = idx[0]

    idx_min, idx_max = int(curr_idx-subrange), int(curr_idx+subrange)
    if idx_min <= 0:
        idx_min = 0
    if idx_max >= len(f_range)//2 + 1:
        idx_max = len(f_range)//2 + 1
        
    for subloc in range(idx_min, idx_max):
        curr_peak = f_range[subloc]
        if (curr_peak - f_range[subloc - 1] > 0) != (f_range[subloc+1] - curr_peak > 0):
            if (abs(f_range[subloc]) >= sol[0][1]) and (abs(f_range[subloc]) >= heightW*overall_avg):
                sol[0] = [subloc, f_range[subloc]]

    curr_idx = idx[1]
    idx_min, idx_max = curr_idx-subrange, curr_idx+subrange
    if idx_min <= len(f_range)//2:
        idx_min = len(f_range)//2
    if idx_max >= len(f_range):
        idx_max = len(f_range)

    for subloc in range(int(idx_min), int(idx_max)):
        curr_peak = f_range[subloc]
        # Check if peak
        if is_global_peak(f_range, subloc, 20):
            # Check if abs val greater than some threshold
            if (abs(curr_peak) >= heightW*overall_avg):
                # Check based on parity
                if (sol[0][1] > 0) and (parity == 0): # Pos, Even
                    if ((curr_peak) >= sol[1][1]) and curr_peak > 0:
                        sol[1] = [subloc, curr_peak]
                elif (sol[0][1] > 0) and (parity == 1): # Pos, Odd
                    if (curr_peak <= sol[1][1]) and curr_peak < 0:
                        sol[1] = [subloc, curr_peak]
                elif (sol[0][1] <= 0) and (parity == 0): # Neg, Even
                    if (curr_peak <= sol[1][1]) and curr_peak < 0:
                        sol[1] = [subloc, f_range[subloc]]
                elif (sol[0][1] <= 0) and (parity == 1):
                    if (curr_peak >= sol[1][1]) and curr_peak > 0: # Neg, Odd
                        sol[1] = [subloc, f_range[subloc]]
    return sol

# Pressure Processing
def pressure(dfP, i_res, idxs, minF):
    if i_res[0][0] == -1:
        min_idxs = xSteps
    else:
        min_idxs = min(xSteps)
    heightW, distW, num_samples = user_inputP(min_idxs)
    solutions = []
    sol_count = 0

    if minPS == -1:     # NON POINT SOURCE
        # Call helper function to process each frequency
        for res in range(len(i_res)):
            freq = i_res[res][1]
            lidx, ridx = (i_res[res][2]-minX)//xStepSize, (i_res[res][3]-minX)//xStepSize
            n = int((freq - minF)/fStepSize)

            print("Working on result ", res+1, " of ", len(i_res), end='\r')

            f_range = np.array(dfP["P"])[idxs[res] + n*xSteps:idxs[res] + (n+1)*xSteps]
            parity, confidence = parity_check(f_range, num_samples)
            # If a solution is produced, increment counter and add a row to our solutions list
            n_match = f_pressure_process(f_range, lidx, ridx, parity, heightW, distW)
            if (n_match[0][0] != -1) and (n_match[1][0] != -1):
                solutions.append([i_res[res][0], freq, minX + xStepSize*n_match[0][0], minX + xStepSize*n_match[1][0], parity, confidence])
                sol_count += 1
            else:
                solutions.append([i_res[res][0], freq, -1, -1, parity, confidence])
    else:   # POINT SOURCE
        for res in range(len(i_res)):
            freq = i_res[res][1]
            print("Working on result ", res+1, " of ", len(i_res), end='\r')

            lidx, ridx = idxs[res][3], idxs[res][4]
            f_range = np.array(dfP["P"])[idxs[res][0] + xSteps[idxs[res][1]] * idxs[res][2]: idxs[res][0] + xSteps[idxs[res][1]] * (idxs[res][2]+1)]
            parity, confidence = parity_check(f_range, num_samples, i_res[res][0])
            # If a solution is produced, increment counter and add a row to our solutions list
            n_match = f_pressure_process(f_range, lidx, ridx, parity, heightW, distW)
            if (n_match[0][0] != -1) and (n_match[1][0] != -1):
                solutions.append([i_res[res][0], freq, minX + xStepSize[idxs[res][1]]*n_match[0][0], minX + xStepSize[idxs[res][1]]*n_match[1][0], parity, confidence])
                sol_count += 1
            else:
                solutions.append([i_res[res][0], freq, -1, -1, parity, confidence])
    print("\n")

    if (print_results()):
        # Specify a gap size between clusters
        gap_size = 1.5
        print("Each pressure result corresponds to a singular intensity result.")
        print("If there is no suitable match, locations are listed as --.", "\n")
        print("                    RESULTS")

        print("PS : Freq : Left Location      Right Location    Parity    Confidence")
        print("----------------------------------------------------------------------")
        helperPrint(solutions, False, gap_size, minF, minPS)
    print("\n")
    # Give option to save results to a txt file
    save_results(solutions)
    
    print("Acoustic Pressure Processing Completed.", "\n")
    return solutions

# Search for possible doublets based off of our search results
def searchDoublets(i_res, p_res, num_results, padding):
    solutions = []
    solutionsI = []
    threshold, confW = -1 * float('inf'), 0.5
    currBest = [[-1, -1, threshold]] * num_results

    for i in range(len(p_res)):
        currPS, currF, parity, inv_conf = p_res[i][0], p_res[i][1], p_res[i][4], 1-p_res[i][5]
        for j in range(i+1,len(p_res)):
            if (currPS != p_res[j][0]):
                break
            if (parity != p_res[j][4]) and (p_res[j][2] != -1) and (parity != -1) and (p_res[j][4] != -1) and (abs(p_res[i][2] - p_res[j][2]) <= 1.5) and\
                (abs(currF - p_res[j][1]) >= padding):
                score = (p_res[i][2] - p_res[j][2])**2 + (p_res[i][3] - p_res[j][3])**2 + confW * (inv_conf + (1-p_res[j][5]))
                currBestMax = 0
                for k in range(len(currBest)):
                    if (currBest[currBestMax][2] > currBest[k][2]) or (currBest[k][2] < 0):
                        currBestMax = k
                if score < abs(currBest[currBestMax][2]):
                    currBest[currBestMax] = [i, j, score]
        if (i == len(p_res) - 1):
            currBest = pair_merge(currBest, 2)
            for i in range(len(currBest)):
                best = currBest[i]
                if best[0] != -1:
                    solutions.append(p_res[best[0]])
                    solutions.append(p_res[best[1]])
                    solutionsI.append(i_res[best[0]])
                    solutionsI.append(i_res[best[1]])
        elif (currPS != p_res[i+1][0]):
            currBest = pair_merge(currBest, 2)
            for i in range(len(currBest)):
                best = currBest[i]
                if best[0] != -1:
                    solutions.append(p_res[best[0]])
                    solutions.append(p_res[best[1]])
                    solutionsI.append(i_res[best[0]])
                    solutionsI.append(i_res[best[1]])
            currBest = [[-1, -1, threshold]] * num_results
    
    if len(solutions) == 0:
        print("No results found.")
    else:
        if (print_results()):
            print("The locations shown are associated with pressure only.")
            print("                    RESULTS")
            print("PS : Freq : Left Location      Right Location    Parity")
            print("-------------------------------------------------------")
            helperPrint(solutions, False, -1, minF, minPS)
        print("")
        save_results(solutions, solutionsI)

    print("Doublet Search Completed.", "\n")
    return solutions, solutionsI

# NEED TO FIX SCALING ON GRAPHS
# Helper function for plotting the intensity
def plot_intensity(f, PS):
    n = int((f/fStepSize)-(minF/fStepSize))
    if minPS != -1:
        ps_idx = int(round((PS-minPS)/(PSStepSize)))
        start_idx = 0
        for v in range(len(xSteps)):
            if v < ps_idx:
                start_idx += xSteps[v] * fSteps
        graph_range = np.array(dfN["I"])[start_idx + n*xSteps[ps_idx]:start_idx + xSteps[ps_idx]*(n+1)]
        scalar = 30/int(xSteps[ps_idx])
        title_end = " (Point Source: " + str(PS) + ")"
    else:
        graph_range = np.array(dfN["I"])[n*xSteps:xSteps*(n+1)]
        scalar = 30/xSteps
        title_end = ""
    maxInt = np.max(graph_range)
    color_list = ['g--', 'b--', 'c--', 'm--', 'y--', 'k--']

    scale, power = 0.1 * 10**global_scale, global_scale
    while True:
        if (maxInt * scale)//1 > 10:
            scale /= 10
            power -= 1
        elif (maxInt * scale)//1 < 1:
            scale *= 10
            power += 1
        else:
            break
    temp = []
    
    if minPS!=-1:
        plt.plot(scalar * np.array(range(xSteps[ps_idx])), scale * graph_range, 'r-')
    else:
        plt.plot(scalar * np.array(range(xSteps)), scale * graph_range, 'r-')
    plt.xlim(minX, maxX)
    plt.ylim(0, int(maxInt*scale + 1))

    if intensity_p:
        cc = 0
        for i in range(len(IntList)):
            if cc >= 6:
                break
            elif (IntList[i][1] == f) and (IntList[i][0] == PS):
                l_loc, r_loc = IntList[i][2], IntList[i][3]
                if [l_loc, r_loc] not in temp:
                    temp.append([l_loc, r_loc])
                    plt.plot([l_loc, l_loc], [0, 55], color_list[cc])
                    plt.plot([r_loc, r_loc], [0, 55], color_list[cc])
                cc += 1
    plt_title = str(f) + "kHz Intensity Plot" + title_end
    plt.title(plt_title)
    yLabel = "Intensity (a.u., scaled by 10^" + str(power) + ")"
    plt.ylabel(yLabel)
    plt.xlabel("x/a")
    
    # Call plt.show() at the end
    plt.show()

# Helper function for plotting the pressure
def plot_pressure(f, PS):
    n = int((f/fStepSize)-(minF/fStepSize))
    if minPS != -1:
        ps_idx = int(round((PS-minPS)/(PSStepSize)))
        start_idx = 0
        for v in range(len(xSteps)):
            if v < ps_idx:
                start_idx += xSteps[v] * fSteps
        graph_range = np.array(dfP["P"])[start_idx + n*xSteps[ps_idx]:start_idx + xSteps[ps_idx]*(n+1)]
        scalar = 30/int(xSteps[ps_idx])
        title_end = " (Point Source: " + str(PS) + ")"
    else:
        graph_range = np.array(dfP["P"])[n*xSteps:xSteps*(n+1)]
        scalar = 30/xSteps
        title_end = ""
    maxP, minP =  np.max(graph_range), np.min(graph_range)
    color_list = ['g--', 'b--', 'c--', 'm--', 'y--', 'k--']
    temp = []

    scale, power = 0.1 * 10**global_scale, global_scale
    while True:
        if (((maxP * scale)//1 > 1) and ((abs(minP) * scale)//1 > 1) and ((maxP * scale)//1 < 10)) or \
            (((maxP * scale)//1 > 1) and ((abs(minP) * scale)//1 > 1) and ((abs(minP) * scale)//1 < 10)):
            break
        elif ((maxP * scale)//1 < 1) or ((abs(minP) * scale)//1 < 1):
            scale *= 10
            power += 1
        elif ((maxP * scale)//1 > 10) or ((abs(minP) * scale)//1 > 10):
            scale /= 10
            power -= 1
    
    if minPS != -1:
        plt.plot(scalar * np.array(range(xSteps[ps_idx])), scale * graph_range, 'r-')
    else:
        plt.plot(scalar * np.array(range(xSteps)), scale * graph_range, 'r-')
    plt.xlim(minX, maxX)
    plt.ylim(int(minP * scale) - 1, int(maxP * scale) + 1)

    if pressure_p:
        cc = 0
        for i in range(len(PreList)):
            if cc >= 6:
                break
            elif (PreList[i][1] == f) and (PreList[i][0] == PS):
                l_loc, r_loc = PreList[i][2], PreList[i][3]
                if [l_loc, r_loc] not in temp:
                    temp.append([l_loc, r_loc])
                    plt.plot([l_loc, l_loc], [-55, 55], color_list[cc])
                    plt.plot([r_loc, r_loc], [-55, 55], color_list[cc])
    plt_title = str(f) + "kHz Acoustic Pressure Plot" + title_end
    plt.title(plt_title)
    yLabel = "Pressure (Pa., scaled by 10^" + str(power) + ")"
    plt.ylabel(yLabel)
    plt.xlabel("x/a")
    
    # Call plt.show() at the end
    plt.show()

# Main Loop for Plotting Results
def plot_loop(minPS, maxPS, PSStepScale, minF, maxF, fStepScale):
    contd = True
    PS = -1
    while contd:
        chose_int = get_plot()
        if minPS != -1:
            PS = round(input_PS(minPS, maxPS), PSStepScale)
        f = round(get_freq(minF, maxF), fStepScale)
        if chose_int:
            plot_intensity(f, PS)
        else:
            plot_pressure(f, PS)
        contd = get_plot_contd()
    print("Data plotting complete.")
    return

# MAIN DRIVER CODE
# Get some user input (i.e., frequency interval); Define min/max x
minF, maxF, I_File, P_File, minPS, maxPS, fStepSize, minX, maxX = user_inputF()
restricted = True   # A variable to determine if restricted mode should be on or not

# Load in data as necessary and calculate other variables (xSteps, etc.)
if (minF != 0) and (maxF != 0):
    restricted = False
    print("Loading in our Intensity data...", "\n")
    # Read our intensity data file and store in a pandas dataframe.
    global_scale = 7

    dfN = pd.read_fwf(I_File, header = None)
    dfN.columns = ["x","I"]

    dfN=dfN.astype('float')
    dfN["I"] = dfN["I"]*(10**global_scale)
    print("Opened Intensity File: ", I_File, "\n")

    print("Loading in our Acoustic Pressure data...")
    dfP = pd.read_fwf(P_File, header = None)
    dfP.columns = ["x","P"]

    dfP=dfP.astype('float')
    dfP["P"] = dfP["P"]*(10**global_scale)
    print("Opened Acoustic Pressure File: ", P_File, "\n")

    # If we are using plane wave, determine step size as usual
    if minPS == -1:
        # Find step size of the horizontal x.
        xSteps = x_step(dfN, maxX)
        xStepSize = float(maxX - minX)/(xSteps)

        # Find step size of the frequency
        fSteps = f_step(dfN, xSteps)
        fStepSize = (maxF-minF)/(fSteps-1)
        PSStepSize, PSStepScale = -1, -1
    # If we are using point source, use point source step sizes
    else:
        fSteps = int(round((maxF-minF)/fStepSize) + 1)
        xSteps = x_stepPS(np.array(dfN["x"]), fSteps)   # Returns a list of the number of indices in each point source's frequency.
        xStepSize = [float(maxX-minX)/val for val in xSteps]
        PSSteps = len(xSteps)
        PSStepSize = (maxPS - minPS)/(PSSteps - 1)
        PSStepScale = decimal_precision(PSStepSize)
    fStepScale = decimal_precision(fStepSize)
    print("")

# Main loop. Asks user to select an option and then carries out selected option.
while True:
    chosen = print_options(restricted)
    dfInt, dfPre, IntList, PreList = [], [], [], []     # Reset large variables to free up space
    intensity_p, pressure_p = False, False

    # Standard Processing
    if chosen == 1:
        print("--STANDARD PROCESSING--", "\n")
        while True:
            # Run intensity data
            print("Starting Intensity Processing...", "\n")
            while True:
                i_results, idxs = intensity(dfN, minF)
                contd = get_contd()
                print("")
                if (contd == False):
                    IntList = i_results
                    intensity_p = True
                    break

            pressureProcess = True

            # If no results, immediately terminate
            if (len(i_results) == 0):
                print("No results produced from parameters. Please try again.")
            elif (not pressureProcess): # Skip pressure processing
                print("Acoustic Pressure Processing skipped.")
            else:
                print("Starting Acoustic Pressure Processing...", "\n")

                while True:
                    # Pass intensity results to pressure
                    p_results = pressure(dfP, i_results, idxs, minF)
                    contd = get_contd()
                    print("")
                    if (contd == False):
                        PreList = p_results
                        pressure_p = True
                        break
                
                # Search for Doublets
                sDoublets = user_searchDoublets()
                if sDoublets:
                    while True:
                        num_results, doublet_padding = num_par_res(maxF - minF)
                        doubletListP, doubletListI = searchDoublets(i_results, p_results, num_results, doublet_padding)
                        contd = get_contd()
                        print("")
                        if (contd == False):
                            IntList = doubletListP
                            PreList = doubletListI
                            break
                # Plot results
                if get_standard_plot():
                    plot_loop(minPS, maxPS, PSStepScale, minF, maxF, fStepScale)
                
            print("Processing Complete.")
            if not ui_Restart():
                break
    
    # Plot data using saved results files
    elif chosen == 2:
        print("--PLOT DATA--", "\n")
        # Retrieve and load in intensity and/or pressure results data files.
        intensity_p, intFileName = get_int_data()
        intFileName = "./Results/" + intFileName
        print("")
        pressure_p, preFileName = get_pre_data()
        preFileName = "./Results/" + preFileName
        if intensity_p:
            dfInt = pd.read_csv(intFileName, sep = " ", header = None)
            dfInt = dfInt.astype('float')
            IntList = convert_to_list(dfInt)

        if pressure_p:
            dfPre = pd.read_csv(preFileName, sep = " ", header = None)
            dfPre = dfPre.astype('float')
            PreList = convert_to_list(dfPre)
        
        # Call helper subroutine to carry out plotting
        plot_loop(minPS, maxPS, PSStepScale, minF, maxF, fStepScale)
    
    # Doublet Search From Results Files
    elif chosen == 3:
        print("--DOUBLET SEARCH--", "\n")
        # Retrieve intensity and pressure results data files
        intensity, intFileName = get_int_data(True)
        intFileName = "./Results/" + intFileName
        print("")
        pressure, preFileName = get_pre_data(True)
        preFileName = "./Results/" + preFileName
        dfInt = pd.read_csv(intFileName, sep = " ", header = None)
        dfInt = dfInt.astype('float')
        i_res = convert_to_list(dfInt)

        dfPre = pd.read_csv(preFileName, sep = " ", header = None)
        dfPre = dfPre.astype('float')
        p_res = convert_to_list(dfPre)

        # Allow user to keep entering parameters until they get a set of results they want
        while True:
            num_results, padding = num_par_res(maxF - minF)
            searchDoublets(i_res, p_res, num_results, padding)
            contd = get_contd()
            print("")
            if (contd == False):
                break

    # Merge Data
    elif chosen == 4:
        print("--MERGE DATA--", "\n")
        # Retrieve first file
        print("--File One--")
        df1FileName = get_data()
        df1FileName = "./Results/" + df1FileName
        print("")
        # Retrieve second file
        print("--File Two--")
        df2FileName = get_data()
        df2FileName = "./Results/" + df2FileName
        df1 = pd.read_csv(df1FileName, sep = " ", header = None)
        df1 = df1.astype('float')

        df2 = pd.read_csv(df2FileName, sep = " ", header = None)
        df2 = df2.astype('float')

        # Call helper functions to merge the dataframes and ask user to save results
        df = merge_lists(df1, df2)
        save_results(df)
        df, df1, df2 = [], [], []   # Clear variables
    
    # QUIT
    else:
        break

    print("\n", "\n")

print("Program Complete.")