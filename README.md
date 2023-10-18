# Activity and Walking Speed Estimation

- This library contain modules for activity detection (Standing,Walking,Running and Squatting) along with Walking speed Estimation. 
- The module that connects and run polar is polar.run.start
- The module that runs the prediction scripts and GUI is main.run.start


# Installation and Running

1. Activate your python environment and cd into the parent directory and run ``` pip install -r requirements.txt ```

2. To install the Activity and Walking speed estimation library  in the parent directory run ``` pip install . ``` . This will install module named **ActWalkEst** in your current environment. The structure of which is as following,
    - ActWalkEst
        - ActivityRec
            - model
            - prediction
        - SpeedEst
            - model
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

4. Directory test contains few example scripts for this library. 