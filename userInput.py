import os

"""
This file contains methods for obtaining and passing along user inputs to other python files.
"""

# Get some input information on the frequency from the user
def user_inputF():
    valid = False
    while not valid:
        minF = float(input("Enter minimum frequency tested (kHz): "))
        maxF = float(input("Enter maximum frequency tested (kHz): "))

        if (minF == 0.0) and (maxF == 0.0):
            print("No files loaded in. Restricted mode on.")
            temp, minPS, maxPS, fStepSize, minX, maxX = ["", ""], -1, -1, -1, 0, 0
            valid = True
        elif (maxF <= minF):
            print("Maximum frequency is less than or equal to minimum frequency. Please re-enter.")
        else:
            temp, minPS, maxPS, fStepSize, minX, maxX = validate_F(minF, maxF)
            if temp != ["", ""] and os.path.exists(temp[0]) and os.path.exists(temp[1]):
                valid = True
            else:
                print("No files corresponding to frequency inputs.")
    if (minF != 0) and (maxF != 0):
        print("Frequency Range: ", str(minF) + "-" + str(maxF), "kHz")
        print("X Range: ", str(minX) + "-" + str(maxX), "\n")
    return minF, maxF, temp[0], temp[1], minPS, maxPS, fStepSize, minX, maxX

# Validates filenames inputted into user_inputF()
def validate_F(f1, f2):
    ifile, pfile = "", ""
    temp = True
    minPS, maxPS, fStepSize, minX, maxX = -1, -1, -1, 0.0, 30.0
    f1, f2 = float(f1), float(f2)
    if (f1 == 400.0) and (f2 == 500.0):
        ifile = "./PW_400-500kHz/Two Row_30 Period Crystal_Hard Boundaries_400-500kHz_Intensity data.txt"
        pfile = "./PW_400-500kHz/Two Row_30 Period Crystal_Hard Boundaries_400-500kHz_Total Acoustic Pressure data.txt"
    else:
        temp = False
        print("No pre-existing data on file.")
        ifile = input("Enter a file name or path for the Intensity File: ")
        pfile = input("Enter a file name or path for the Pressure File: ")
        minX, maxX = get_x_range()
        minPS, maxPS, fStepSize = get_PS(minX, maxX)
    if temp == True:
        print("Using pre-loaded parameters.")
    return [ifile, pfile], minPS, maxPS, fStepSize, minX, maxX

# Get x range tested from user.
def get_x_range():
    while True:
        minX = float(input("Enter minimum x location: "))
        maxX = float(input("Enter maximum x location: "))

        if (maxX <= minX):
            print("Maximum x location is less than or equal to minimum location. Please re-enter.")
        else:
            return minX, maxX

# Get point source, if the file refers to a point source
def get_PS(minX, maxX):
    while True:
        mode = int(input("Using plane wave(1) or point source(2)? "))
        if (mode == 2):
            while True:
                minPS = float(input("Enter minimum point source location: "))
                maxPS = float(input("Enter maximum point source location: "))
                intf = float(input("Enter the frequency step size: "))
                if (maxPS <= minPS):
                    print("Maximum location is less than or equal to minimum location. Please re-enter.")
                elif (minPS < minX) or (maxPS > maxX):
                    print("Point Sources are not within x location range. Enter values between " + str(minX) + " and " + str(maxX))
                else:
                    print("Minimum Point Source Location:", minPS)
                    print("Maximum Point Source Location:", maxPS)
                    print("Frequency Interval Step Size:", intf, "\n")
                    return minPS, maxPS, intf
        elif (mode == 1):
            return -1, -1, -1
        else:
            print("Invalid selection. Enter only 1 or 2.")

# Get some input information on the desired Intensity processing from the user
"""
numPeaks: Describes the number of largest peaks that we will compare for each half, if such a number exists.
        This is a maximum, so other weights may cause it to be less than that.
heightW: A weight value which penalizes shorter peaks more compared to larger peaks.
symW: A weight value which penalizes peak pairs that are less symmetric, compared to those that are more symmetric.
"""
def user_inputI():
    valid = False
    while not valid:
        numPeaks = int(input("Enter max number of peaks to be compared at each frequency: "))
        heightW = float(input("Enter a penalty weight for intensity height: "))
        symW = float(input("Enter a penalty weight for symmetry: "))

        if (numPeaks > 15):
            print("Number of peaks is too high. For a reasonable runtime, select a value less than 15.", "\n")
        else:
            valid = True
    
    print("Number of Peaks: ", numPeaks)
    print("Height Weight: ", heightW)
    print("Symmetry Weight: ", symW, "\n")
    return numPeaks, heightW, symW

# General user input for Pressure processing.
def user_inputP(min_i):
    while True:
        heightW = float(input("Enter a penalty weight for pressure height (magnitude): "))
        distW = float(input("Enter a search weight for x location: "))
        numSamples = input("Enter the number of random sample points to test in parity, or enter Y for default(15): ")

        if numSamples != "Y":
            numSamples = int(float(numSamples))
        else:
            numSamples = 15

        if numSamples > (min_i//4):
            print("Too many sample points. Enter a smaller integer value.")
        elif (numSamples < 0):
            print("Invalid number of samples. Enter a positive integer value.")
        elif (heightW <= 0):
            print("Invalid height weight. Please re-enter.")
        elif (distW <= 4) or (distW >= min_i):
            print("Invalid distance weight. Please re-enter.")
        else:
            break
    print("Height Weight: ", heightW)
    print("Distance Weight: ", distW)
    print("Number of Sample Points: ", numSamples)
    return heightW, distW, numSamples

# Asks user if they would like to run a stage of processing again.
def get_contd():
    while True:
        cont = input("Would you like to run this processing again with different parameters? (Y/N) ")
        if (cont == "Y"):
            return True
        elif (cont == "N"):
            return False
        else:
            print("Invalid response. Please enter Y or N.")

# Asks user if they would like to run a different test.
def get_contd_separate():
    while True:
        cont = input("Would you like to run a different test? (Y/N) ")
        if (cont == "Y"):
            return True
        elif (cont == "N"):
            return False
        else:
            print("Invalid response. Please enter Y or N.")

# Asks user if they would like to search for doublets.
def user_searchDoublets():
    while True:
        search = input("Search for doublets based on these results? (Y/N) ")
        if (search == "Y"):
            return True
        elif (search == "N"):
            return False
        else:
            print("Invalid response. Please enter Y or N.")

# Asks the user how many doublet results per point source they would like to search for
# Also asks for a padding value.
def num_par_res(fInterval):
    while True:
        num_res = int(input("Enter the maximum number of results (for each point source, if applicable): "))
        padding = input("Due to parity inaccuracy, a padding is recommended between doublet frequencies. Enter a padding range or press Y for default(0.15): ")
        if (padding == "Y") and (num_res > 0):
            return num_res, 0.15
        elif (float(padding) <= fInterval) and (num_res > 0):
            return num_res, float(padding)
        elif num_res <= 0:
            print("Invalid number of results. Please re-enter.")
        else:
            print("Invalid padding. Please re-enter.")
        
# Asks users if they would like to repeat standard processing
def ui_Restart():
    while True:
        restart = input("Restart entire standard processing using the current intensity and pressure files? (Y/N) ")
        if restart == "Y":
            return True
        elif restart == "N":
            return False
        else:
            print("Invalid response. Please re-enter.")

# Asks user if they would like to print their results to the terminal
def print_results():
    while True:
        temp = str(input("Would you like to print results to terminal? (Y/N) "))
        if (temp == "Y") or (temp == "N"):
            break
        else:
            print("Invalid response. Please enter Y or N.")
    if (temp == "Y"):
        return True
    print("")
    return False

# Fetches user's intensity data files to be plotted
def get_int_data(req = False):
    if not req:
        while True:
            i = str(input("Would you like to show intensity processing results? (Y/N) "))
            if (i == "Y"):
                while True:
                    int_file = str(input("Enter the path from the Results parent folder to the results .TXT you would like to use: ./Results/"))
                    if len(int_file) <= 4:
                        print("File is not a suitable file format. Please re-enter.")
                    elif (os.path.isfile("./Results/" + int_file)) and (int_file[len(int_file) - 4:] == ".txt"):
                        return True, int_file
                    else:
                        print("File does not exist or is not in the specified folder, or is not a .TXT. Please re-enter.")
            elif (i == "N"):
                return False, ""
            else:
                print("Invalid response. Please re-enter.")
    while True:
        print("Intensity File")
        int_file = str(input("Enter the path from the Results parent folder to the results .TXT you would like to use: ./Results/"))
        if len(int_file) <= 4:
            print("File is not a suitable file format. Please re-enter.")
        elif (os.path.isfile("./Results/" + int_file)) and (int_file[len(int_file) - 4:] == ".txt"):
            return True, int_file
        else:
            print("File does not exist or is not in the specified folder, or is not a .TXT. Please re-enter.")

# Fetches user's pressure data files to be plotted
def get_pre_data(req = False):
    if not req:
        while True:
            p = str(input("Would you like to show acoustic pressure processing results? (Y/N) "))
            if (p == "Y"):
                while True:
                    pre_file = str(input("Enter the path from the Results parent folder to the results .TXT you would like to use: ./Results/"))
                    if len(pre_file) <= 4:
                        print("File is not a suitable file format. Please re-enter.")
                    elif (os.path.isfile("./Results/" + pre_file)) and (pre_file[len(pre_file) - 4:] == ".txt"):
                        return True, pre_file
                    else:
                        print("File does not exist or is not in the specified folder. Please re-enter.")
            elif (p == "N"):
                return False, ""
            else:
                print("Invalid response. Please re-enter.")
    while True:
        print("Acoustic Pressure File")
        pre_file = str(input("Enter the path from the Results parent folder to the results .TXT you would like to use: ./Results/"))
        if len(pre_file) <= 4:
            print("File is not a suitable file format. Please re-enter.")
        elif (os.path.isfile("./Results/" + pre_file)) and (pre_file[len(pre_file) - 4:] == ".txt"):
            return True, pre_file
        else:
            print("File does not exist or is not in the specified folder. Please re-enter.")

# Fetches a user's intensity or pressure file for merging
def get_data():
    while True:
        print("Intensity or Pressure File")
        pre_file = str(input("Enter the path from the Results parent folder to the results .TXT you would like to use: ./Results/"))
        if len(pre_file) <= 4:
            print("File is not a suitable file format. Please re-enter.")
        elif (os.path.isfile("./Results/" + pre_file)) and (pre_file[len(pre_file) - 4:] == ".txt"):
            return pre_file
        else:
            print("File does not exist or is not in the specified folder. Please re-enter.")

# Asks user if they would like to plot the results of their standard processing.
def get_standard_plot():
    while True:
        should_graph = str(input("Would you like to plot these results? (Y/N) "))
        if (should_graph == "Y"):
            return True
        elif (should_graph == "N"):
            return False
        else:
            print("Invalid response. Please re-enter.")

# Asks users if they would like to plot intensity or pressure
def get_plot():
    while True:
        i_p = str(input("Plot Intensity or Pressure? (I/P) "))
        if (i_p == "I"):
            return True
        elif (i_p == "P"):
            return False
        else:
            print("Invalid response. Please re-enter.")

# Asks user for frequency they would like to plot
def get_freq(minF, maxF):
    while True:
        text = "Enter a frequency b/w " + str(minF) +  " and " + str(maxF) + " (Rounded to nearest valid response): "
        freq = float(input(text))
        if (freq <= maxF) and (freq >= minF):
            print("")
            return round(freq, 2)
        else:
            print(freq, "is an invalid frequency. Please re-enter.")

# Asks user for the point source location they would like to plot
def input_PS(minPS, maxPS):
    while True:
        text = "Enter a point source location b/w " + str(minPS) +  " and " + str(maxPS) + " (Rounded to nearest valid response): "
        PSloc = float(input(text))
        if (PSloc <= maxPS) and (PSloc >= minPS):
            print("")
            return round(PSloc, 2)
        else:
            print(PSloc, "is an invalid location. Please re-enter.")

# Asks user if they would like to continue plotting graphs
def get_plot_contd():
    while True:
        contd = str(input("View another plot? (Y/N) "))
        if (contd == "Y"):
            return True
        elif (contd == "N"):
            return False
        else:
            print("Invalid response. Please re-enter.")

# Provides users options in main loop
def print_options(restricted):
    while True:
        if restricted:
            print("Restricted mode. Options 1 and 2 require a file to be loaded and cannot be run.")
        else:
            print("NOTE: Options 1 and 2 require the loaded data files. Options 3 and 4 can be run arbitrarily.")
        print("Select one of the following options: ")
        print("(1) Standard Intensity, Acoustic Pressure, Doublet Processing")
        print("(2) Plot Results from Results Data")
        print("(3) Doublet Search From Results Data")
        print("(4) Merge Intensity or Pressure Data")
        print("(5) Quit")

        print("")
        select_option = int(float(input("Select one of the above options: ")))
        print("")
        if (select_option < 1) or (select_option > 5):
            print("Invalid response. Please re-enter.", "\n")
        elif (restricted) and (select_option <= 2):
            print("Restricted mode. Options 1 and 2 are unavailable.", "\n")
        else:
            return select_option