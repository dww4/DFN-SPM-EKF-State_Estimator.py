import pybamm
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

#### Simulate "Real" Data Using DFN Model ####
dfn_model = pybamm.lithium_ion.DFN(options={"thermal": "lumped"}) #Define DFN model with lumped thermal eqn
pv = pybamm.ParameterValues("Chen2020") #Use LGM50 Cell Parameters (NMC-SiGraphite-LiPF6)
sim_dfn = pybamm.Simulation(dfn_model,parameter_values=pv) #Create simulation with model


#Apply a dynamic current profile
time = np.linspace(0, 3600, 1000)  #1-hour simulation
current_profile = -2 * np.sin(2 * np.pi * time / 3600)  #Sinusoidal current [A] to model realistic conditions
input_parameters = {"Current function": pybamm.Interpolant(time, current_profile, pybamm.t)}

#Run DFN simulation
sim_dfn.solve(t_eval=time, inputs=input_parameters)

#Extract "measured" terminal voltage, SOC, and temperature
voltage_dfn = sim_dfn.solution["Terminal voltage [V]"].data

#Calculate SOC based on surface lithium concentration
c_s_n = sim_dfn.solution["X-averaged negative particle surface concentration [mol.m-3]"].data
soc_dfn = c_s_n /33133.0 #33133.0 is maximum conc on surface from Chen2020 paramter set

#Extract OCV vs SOC from DFN Model
ocv_dfn = sim_dfn.solution["Battery open-circuit voltage [V]"].data

#Compute dV/dSOC using finite differences
H_values = np.gradient(ocv_dfn, soc_dfn)
H_interp = lambda soc: np.interp(soc, soc_dfn, H_values)

temp_dfn = sim_dfn.solution["X-averaged cell temperature [K]"].data

#Add measurement noise
voltage_measured = voltage_dfn + np.random.normal(0, 0.01, size=voltage_dfn.shape)  #10 mV noise
temp_measured = temp_dfn + np.random.normal(0, 0.5, size=temp_dfn.shape)  #0.5 K noise
time_sim = sim_dfn.solution["Time [min]"].data

#Plot DFN results
plt.figure(figsize=(10, 7))
plt.subplot(3, 1, 1)
plt.plot(time_sim, voltage_measured, label="Measured Voltage (DFN + Noise)")
plt.xlabel("Time [min]")
plt.ylabel("Voltage [V]")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(time_sim, temp_measured, label="Measured Temperature (DFN + Noise)", color='r')
plt.xlabel("Time [min]")
plt.ylabel("Temperature [K]")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(time_sim, soc_dfn, label="True SOC (DFN)", color='g')
plt.xlabel("Time [min]")
plt.ylabel("State of Charge")
plt.legend()
plt.tight_layout()
plt.show()

#### Set up the SPM for EKF #####
spm_model = pybamm.lithium_ion.SPM()
sim_spm = pybamm.Simulation(spm_model,parameter_values=pv)

##### Implement EKF for SOC Estimation #####

#EKF parameters
Q = np.array([[1e-5]])  # Process noise covariance (SOC)
R = np.array([[1e-4]])  # Measurement noise covariance (Voltage)

#Initial estimates
state_est = np.zeros(len(time))  #SOC
state_est[0] = 0.9  #Initial SOC guess
P = np.array([[0.01]])  #Initial error covariance
execution_time = np.zeros(len(time)) #Average execution time per step

for k in range(1, len(time_sim)):
    start_time = datetime.now()
    dt = time[k] - time[k - 1]
    current = current_profile[k]

    #Prediction step using SPM model
    sim_spm.solve(t_eval=[0, time[k]], inputs={"Current function": current})
    c_s_n_spm = sim_spm.solution["X-averaged negative particle surface concentration [mol.m-3]"].data[-1]
    voltage_pred = sim_spm.solution["Terminal voltage [V]"].data[-1]
    soc_pred = c_s_n_spm /33133.0 #33133.0 is maximum conc on surface from Chen2020 paramter set
    P_pred = P + Q

    #Correction step
    z_k = voltage_measured[k]  #Measurement
    h_x = voltage_pred  #Model prediction
    y_k = z_k - h_x  #Innovation

    #Measurement sensitivity matrix H computed from OCV-SOC curve
    H = np.array([[H_interp(soc_pred)]])

    #Kalman Gain
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)

    #State update
    state_est[k] = soc_pred + (K * y_k)[0]
    P = (np.eye(1) - K @ H) @ P_pred
    end_time = datetime.now()
    execution_t = end_time - start_time
    execution_time[k] = execution_t.total_seconds()

#Plot SOC estimation results
plt.figure(figsize=(10, 5))
plt.plot(time_sim, soc_dfn, label="True SOC (DFN)", color='g')
plt.plot(time/60, state_est, label="Estimated SOC (SPM+EKF)", linestyle='--')
plt.xlabel("Time [min]")
plt.ylabel("State of Charge")
plt.legend()
plt.show()
print("Average Execution Time: %g | #Estimates/sec: %g" %(np.mean(execution_time),1/np.mean(execution_time)))
