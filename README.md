# Visual-OddBall-Task-tDCS
This repository includes a visual "oddball" task designed for use during a transcranial direct current stimulation (tDCS) session, developed as part of the research for my Master's Dissertation titled "Enhancing Brain-Computer Interface (BCI) Performance with tDCS."

## Requirements
- Python
- PsychoPy

## Task Description
The task we used was adapted from an original task developed by D’Amico A. [1], which was itself based on the paradigm by Li et al. [2]. We modified the number of runs, and blocks, and adjusted the types and frequency of stimuli to suit the specific needs of our study. Furthermore, we configured the task for synchronous integration with the Neuroelectrics stimulation software.

As shown in Figure 1, the task involved a visual counting exercise conducted over 12 runs. Each run presented a continuous sequence of stimuli, followed by a fixation cross. Participants received instructions via guidance screens to count only the target figures, distinguished by their green colour, and to disregard figures of other colours (yellow, blue, and white). After each run, participants reported their count of the target figures by choosing one of two options displayed on the screen and pressing the corresponding button.

![image](https://github.com/user-attachments/assets/81b95652-5fa3-4b8f-993c-00038143db08)

The experiment was divided into three blocks of about 5 minutes each, with each block containing four runs. After each set of four runs, participants were given a 1.5-minute break to rest, helping to reduce fatigue while maintaining their focus. The first block featured triangles, the second block included rhombuses, and the final block used stars, introducing variety to prevent task monotony.

Each run consisted of 28 to 48 trials, with target stimuli appearing in 25% of the trials and non-target stimuli in the remaining 75%. Each trial included a figure displayed for 800 to 1000 milliseconds, followed by a fixation cross shown for another 800 to 1000 milliseconds. The number of target figures per run varied between 7 and 12 to keep the task unpredictable and engaging for participants. To ensure synchronization between stimuli presentation and data collection, custom Python scripts were used to send codes/triggers, integrated with the Neuroelectrics software. This setup ensured precise alignment of visual stimuli with neurophysiological data, allowing for accurate ERP analysis later on. Participants were given a task explanation and brief demonstration prior to the start of the experiment.


## Usage
1. Download ...
2. Run main.py


## License
Free to use, share, and adapt the material, provided proper credit is given.

## References
[1] D’Amico A “Ollie.” ollie-d/PsychoPyP300: Simple PsychoPy visual oddball paradigm. Designed to be a test of LSL marker and photosensor latency. Accessed September 6, 2024. https://github.com/ollie-d/PsychoPyP300
[2] Li F, Yi C, Jiang Y, et al. Different Contexts in the Oddball Paradigm Induce Distinct Brain Networks in Generating the P300. Front Hum Neurosci. 2019;12. doi:10.3389/FNHUM.2018.00520/FULL
