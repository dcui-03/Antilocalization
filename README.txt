DATA ANALYSIS PROGRAM FOR COMSOL-SIMULATED PHONONIC CRYSTAL ANTILOCALIZATION*

Most Recent Update: 07.16.2024

This program was developed for the purpose of analyzing COMSOL Simulation data produced by the University of North Texas (UNT) Physics Department's Phononic Crystal Antilocalization project.
Below are general details and instructions for use. If using or modifying for research purposes, please provide credit to the original programmer (Daniel Cui). Additionally, any questions regarding
the code or suggestions for improvement can be emailed to him (dcui@bu.edu). Additional details regarding the project and programming logic can be requested directly from the proper UNT authorities.

In order to get started with the code, it is recommended that a minimum of the General Descriptions, How to Use, and the File Tree sections are read.

------------------------------------------

GENERAL DESCRIPTIONS
- main.py:
    Main.py is the file containing runnable code. Its main job is to provide the user with options, and execute them. These options
    include standard analysis, plotting data, searching for doublets, and merging data.

    1. Standard Processing: This is the primary tool for analyzing data. After taking some user raw user-input data, it will return the best pairs of locations based on a weighted combination
    of both symmetry and centrality. It also matches these locations with acoustic pressure and assigns a parity to each graph. Lastly, it can search for doublets using these results.
    The resulting locations and frequencies can be saved as .TXT files in a designated Results folder.
        - The saved .txt file format is in the following format: [POINT SOURCE]  [FREQUENCY]  [LEFT LOCATION]  [RIGHT LOCATION] ([PARITY]  [CONFIDENCE]).
            - 0 means EVEN parity. 1 means ODD parity.
            - Confidence is a score of how likely the chosen parity is to be correct.
            - Parity and Confidence are not shown in intensity results.
    After obtaining results, the user will be asked if they would like to graph these results.
    In general, if the user if working with plane wave data and would like to filter for point source data, they should simply run intensity processing and get a handful
    of results for export, and then skip pressure processing. If the user is working with point source data and would like to search for doublets, they should run intensity
    processing with much lower parameters, then run pressure processing and doublet search.

        There are several user inputs which the user can control during this process:
        i. Intensity Processing:
            - Max Number of Peaks: This is the number of peaks on each side of the crystal that will be compared.
                RECOMMENDED INPUT: 10-15
            - Intensity Height Penalty: Only peaks that are this factor above the average will be considered in the peak comparison.
                This parameter may require some fiddling and rerunning, so don't be afraid to try many different values.
                RECOMMENDED (FIRST) INPUT: 4.5 +/- 1.0
            - Symmetry Weight: This parameter weights how important symmetry is in scoring pairs.
                RECOMMENDED INPUT: 30-45

            If seeking to generate doublets from point source data, generally a large number of results is preferred (~1500). If seeking to find clusters, tightening
            user inputs may help to narrow down the top ~10 candidates.
        
        ii. Pressure Processing:
            - Pressure Height: Only peaks that are this factor above the average will be considered. Note that pressure averages tend to be near zero.
                RECOMMENDED INPUT: 1-1.5
            - X Location Search Weight: This is the inverse distance that the program will look for a matching peak. Larger values will yield a tighter search range.
                RECOMMENDED INPUT: 30
            - Parity Sample Points: This is the number of random points to sample when determining parity. A middling number is recommended, as values too low will not produce
                accurate results, while values too high will overweigh this test compared to the other parity tests.
                RECOMMENDED INPUT: 15

            Pressure Processing may also rule out frequencies that seem to have an unclear parity or do not produce any matching peaks.
            Confidence scores will also trend toward low values because of the breadth of tests run (see section below). Again, visual checks will be a necessary final step.
        
        iii. Doublet Search
            - Number of Results: Limits the number of results which can be returned by each frequency. Since there are normally only a handful of promising
                results in each frequency that already has candidates, this number should be relatively low.
                RECOMMENDED INPUT: 3-5
            - Padding: Doublets tend to be in a close range, but if the program produces frequency pairs that are too close (within a range of ~0.15kHz), these tend
                to be false positives. Notice that this is not always true, but in order to filter out most of these false positives, the user can enforce a minimum
                difference in frequencies required to make a doublet pair.
                RECOMMENDED INPUT: 0.15-0.25

    2. Plotting Data: This is used to plot the input data loaded in. Additionally, users can load in intensity and/or pressure results files obtained from standard
    processing. If graphing a frequency containing a solution in the corresponding results file, a dashed line will mark that location pair. Since Matplot is used
    to graph the data, any plots can be saved as .PNG files.

    3. Search for Doublets: This option simply performs a doublet search as outlined in Standard Processing, but does so based on results file inputs. It is important
    that the user inputs only 2 intensity results files or 2 pressure results files. Mixing may cause issues.

    4. Merge Data: In the case where there are two files that were processed separately but need to be merged (i.e., for plotting or doublet search), the user can use
    this option to merge the data. They will then be given the option to save it.

    5. Quit: Ends the menu option loop and quits the program.

- userInput.py:
    This file simply stores functions which prompt the user for some input. It also contains functions which validate user inputs.

- utils.py
    This file contains general operational functions, such as functions for saving data, as well as mathematical and sorting operations.

------------------------------------------

HOW TO USE/RUN
To use, download files in the same folder from the associated GitHub or a legitimate UNT source. Load into your
favorite IDE. If you are using an IDE that does not primarily support Python, be sure to have a Python Interpreter extension downloaded.
Next, don't forget to build your python environment. Since I use VSCode, I will link the instructions to build a
Python VSCode environment here: https://code.visualstudio.com/docs/python/environments.

Before running the program, add a COMSOL data file to the folder. In order to properly load the .TXT file in, you will need to make the following alterations to these files:
    - Remove any header text. Leave only the 2 columns of numbers.
    - In the first row of each column, add extra zeros such that the first values in each column are much longer than the values below it.
        - Ex. If the first location reads as 0.123456E10, instead extend it as 0.1234560000000E10
Additionally, be sure to have some information about the data on hand, including the frequency range, the location range, and the step sizes run. Data entered must 
also satisfy some basic requirements, such as needing a regular, positive interval for the locations of the same point source. Graphs for each frequency must also be continuous in the
location range. Parametric sweeps over 1 point source location range and 1 frequency range is preferred, since the code is not equipped to handle gaps between ranges.

You can now run the program on the terminal as expected; be sure to follow the prompts and give the correct type of response when prompted. When prompted to enter
a file, be sure to enter the correct filepath, or the code may fail. For a brief overview of filepaths, see the first NOTE below. Additionally,
users should be wary of the EXPECTED FILE TREE (see below section). Upon obtaining results in standard processing, the user will be given the option to create a Results folder in which to store
the results of their processing if they have not already created one. Similarly, the program will also provide the option to create Results subfolders.

When plotting data, it is important to remember that only one plot can be viewed at a time. Therefore, you MUST close the currently displayed graph before attempting to view and/or save
another one. 

Also note that if users would like to bypass loading in data and would only like to search for doublets with existing files or merge data from existing files, the user can enter 0
and 0 as the minimum and maximum frequency. 

NOTE: If the user is unfamiliar with filepaths, I will provide a very quick rundown here:
    - The period (.) in a filepath refers to the current folder.
    - Forward slash (/) is used to separate hierarchies in a filetree.
    Ex. If you are currently at the "Main" folder level, which contains a "Results" subfolder, then the following
    would be a potential filepath to a particular result .TXT file:
        ./Results/result.txt
    Similarly, if you had a subfolder in Results called "Example" which contained a file called "example.txt", then the
    filepath to example.txt from Main would look like this:
        ./Results/Example/example.txt

NOTE: Although many safety measures have been put in place to avoid issues, edge cases may cause long wait times or infinite loops.
A user may also enter a wrong input that they cannot later re-enter. In order to kill a currently running program in these cases, you can enter ctrl+C at any time.
This will effectively end the program and empty out any stored values, letting you rerun the program from the start once you are ready again.

------------------------------------------

EXPECTED FILE TREE
The following file tree should be used to avoid issues:
- Main Folder
    - main.py
    - userInput.py
    - utils.py

    - (Folders for raw data)
    - Results: A folder to store the .TXT results of IPSort.py
        - (Subfolders for each frequency range)
    - Images: A folder to store images resulting from plot_results.py
        - (Subfolders for each frequency range)

Pre-loading a file: Instead of entering in the file path and point source parameters manually, users can "pre-load" this data into utils.py.
    To do so, go to the validate_f() function in utils.py. Following the format of the other pre-loaded files, create an elif statement and
    enter in the frequency range, file path to both intensity and acoustic pressure, as well as minimum/maximum point source location and
    frequency step size if applicable.

Restricted Mode: If the user only wishes to find doublets or merge data in existing results files, they may enter 0 and 0 for the minimum and maximum
    frequency inquiries, which will begin restricted mode. Since loading in the raw data tends to take the longest amount of time, restricted mode
    avoids doing this and jumps immediately to the options menu. However, users will not be able to do standard processing or plot data during this
    run of the program.

------------------------------------------

DETAILED DESCRIPTION OF PARITY TESTS
Several tests are run and weighted in order to determine a frequency's parity. They are as follows:
    1. Find nearest peaks from center (weight: +/- 0.25): Since graphs with a peak near the line of symmetry tend to be even, this test tries to catch this
        by discovering these 2 nearest peaks and then seeing how far each is from the line of symmetry.
    2. Find slope at nearest zeros from center (weight: +/- 0.25): Although checking slope at the line of symmetry rarely gives good results, checking slopes
        at nearest zeros to this line of symmetry can show an inclination towards odd or even parity.
    3. Compare signs of greatest peaks (weight: +/- 0.2): Since the largest peaks tend to be (but are not always) the peaks which give the clearest indicator of parity,
        the sign of the greatest peak on the left side of the crystal is compared with the sign of the greatest peak on the right side.
    4. End behavior (weight: +/- 0.7): Many graphs (but not all) have very clear indicators of parity at the far ends of the crystal. Therefore, this graph compares the trends
        of slope at both ends of the crystal.
    5. Test at some contributor point (weight: +/- 0.4): If there is a point source, then we expect there to be a peak at that point. We also expect some peak at a similar 
        location on the right side of the crystal, so we check to see if their signs match.
    6. Random contributor points (weight: +/- n*0.025): Plots with very strong parity tend to express that parity throughout the entire crystal, so we simply take a user-defined
        n number of unique, random sample points for testing and check to see if their signs match on both sides of the crystal. The more samples the better, although too many
        samples may incorrectly affect certain plots which have some skew.
    7. Compare averages of each half (weight: +/- 0.2): Since odd parity should differ from even only in negation, we can compare the sign of the average value in the left half
        with the sign of the average value in the right half.

To obtain parity, a positive running sum of the tests indicates even parity, while a negative running sum indicates odd parity. Similarly, confidence is determined
by dividing the running sum by the maximum number of earnable points for a particular parity.

------------------------------------------

KNOWN ISSUES AND FUTURE IMPLEMENTATIONS
- No strict type-checks means that some incorrect user inputs will cause the code to fail. BE WARY OF INCORRECT INPUTS!
- Parity check and thus confidence still have some inaccuracies, especially as frequency increases.

------------------------------------------

* This code was developed by Daniel Cui (Boston University) as part of a research project conducted at the UNT Physics Department REM program, with mentorship from Dmytro Shymkiv (UNT).