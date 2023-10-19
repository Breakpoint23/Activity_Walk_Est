# Activity and Walking Speed Estimation

- This library contain modules for activity detection (Standing,Walking,Running and Squatting) along with Walking speed Estimation. 
- The module that connects and run polar is polar.run.start
- The module that runs the prediction scripts and GUI is main.run.start 
- Running main.run.start also creates two pylsl streams , one for Activity Estimation and one for Walking speed estimation
- Currently Walking Estimation supports 2 different deep learning architecture 
    1. ![Self - Attention](Merged2.jpg)
    2. ![Vanilla CNN](Merged.jpg)
    - with two different input length (0.5 s data, 1 s data) and two different acceleration combinations (Vertical only, Vertical and lateral)
- Working on adding more walking speed estimation models and model comparision GUI.


# Installation and Running

1. Activate your python environment and cd into the parent directory and run ``` pip install -r requirements.txt ```

2. To install the Activity and Walking speed estimation library  in the parent directory run ``` pip install . ``` . This will install module named **ActWalkEst** in your current environment. The structure of which is as following,
    - ActWalkEst
        - ActivityRec
            - model
            - prediction
        - SpeedEst
            - model1
            - model2
            - model3
            - model4
            - model5
            - prediction
        - polar
            - polar
            - utils
            - search_polar
            - collect_polar
            - run
        - UI
            - ui1
        - main
            - run
        - resources
            - saved models for activity and speed estimation
            - saved normalizer for speed estimation
            - images for ui
            - polar address file
        - utils
            - lsl_imu


3. For now to run the estimation, 

    - start the data stream from polar. For which one needs to call function *start* from **polar.run** module (takes no arguments).
    - start the estimation algorithms . For which one needs to call function *start* from **main.run** module (takes no arguments).
    - while initializing main.run module, it will pop up the GUI but it'll also ask the user to select a walking speed estimation model to run in the terminal.

4. Directory test contains few example scripts for this library. 
