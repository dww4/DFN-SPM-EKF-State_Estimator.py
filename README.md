# Extended Kalman Filter w/ Reduced Order Model State Estimator

############################### Description ###############################

Uses the Single Particle Model (SPM) combined with an Extended Kalman Filter (EKF) to estimate state of charge (SOC). Uses a thermal-electrochemical Doyle/Fuller/Newman (DFN/P2D) model to obtain test data for the estimator. 

############################### Dependencies ###############################

0. PyBamm Installation (pip install pybamm)

1. DFN-SPM-EKF-State_Estimator.py

############################### Executing Program ###############################

1. Run Simulation:
	a. Run: python DFN-SPM-EKF-State_Estimator.py

	b. Should Output:
		i. Plot of DFN simulation results for Voltage, State of Charge, and Temperature
		ii. Plot of "True" SOC from DFN and Estimated SOC from EKF/SPM

############################### Help ###############################

Ask DJ (dejuante1503@gmail.com) 
