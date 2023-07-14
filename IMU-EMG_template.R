# Código demostración trabajo práctico (Code demonstration practical work) (Code démonstration travaux pratiques)
# Template for the experiment
# Prepared by Andres Torres, Mariana Ochoa Franco and Frederic Domingue
# Version 1.0 (draft)
# Starting point, the code could be improved 

# Cargar librerias(load libraries)(Charger)
#(zoo), (pracma), (signal), (tidyverse), (seewave),(gsignal), (readxl)
#...

library(zoo)
library(pracma)
library(signal)
library(tidyverse)
library(gsignal)
library(readxl)

#########################################
### CODE FOR IMUS ( Codigo para 2 IMU`s`) 
#########################################
# SIGNAL PROCESSING FOR 2 IMUS ( Procesamiento de la señal 2 IMUS)

# Set up a 3x1 plotting area ( configuración de area de graficado)
#...

# 1. Read data ( lectura de datos)

# Load IMU data into a data frame (Cargar los datos a un data frame)
# remove the empty rows
#...

IMU1_data <- #load the first IMU 

IMU2_data <- #load de second IMU




# Name the columns
#...
  
names(IMU1_data) <- #PacketCounter,SampleTimeFine,Acc_X,Acc_Y,Acc_Z,
                    # Gyr_X, Gyr_Y, Gyr_Z, Mag_X,Mag_Y, 
                    # Mag_Z

names(IMU2_data) <- #PacketCounter,SampleTimeFine,Acc_X,Acc_Y,Acc_Z,
  # Gyr_X, Gyr_Y, Gyr_Z, Mag_X,Mag_Y, 
  # Mag_Z


# Convert to numeric vector ( convertir a un vector numerico )
time1 <- seq(0, (nrow(IMU1_data)-1)/60, by=1/60)  #as.numeric(IMU1_data$PacketCounter)/60 
Acc_X1 <- as.numeric(IMU1_data$Acc_X)
Acc_Y1 <- as.numeric(IMU1_data$Acc_Y)
Acc_Z1 <- as.numeric(IMU1_data$Acc_Z)
Gyr_X1 <- as.numeric(IMU1_data$Gyr_X)
Gyr_Y1 <- as.numeric(IMU1_data$Gyr_Y)
Gyr_Z1 <- as.numeric(IMU1_data$Gyr_Z)
Mag_X1 <- as.numeric(IMU1_data$Mag_X)
Mag_Y1 <- as.numeric(IMU1_data$Mag_Y)
Mag_Z1 <- as.numeric(IMU1_data$Mag_Z)

time2 <- seq(0, (nrow(IMU2_data)-1)/60, by=1/60)
Acc_X2 <- as.numeric(IMU2_data$Acc_X)
Acc_Y2 <- as.numeric(IMU2_data$Acc_Y)
Acc_Z2 <- as.numeric(IMU2_data$Acc_Z)
Gyr_X2 <- as.numeric(IMU2_data$Gyr_X)
Gyr_Y2 <- as.numeric(IMU2_data$Gyr_Y)
Gyr_Z2 <- as.numeric(IMU2_data$Gyr_Z)
Mag_X2 <- as.numeric(IMU2_data$Mag_X)
Mag_Y2 <- as.numeric(IMU2_data$Mag_Y)
Mag_Z2 <- as.numeric(IMU2_data$Mag_Z)


# Take the same number of samples (we would't have to do it if both IMUs save data at the same time)Tomar el mismo número de muestras (no tendríamos que hacerlo si ambas IMU guardan datos al mismo tiempo)
# if IMU1 is longer 
time1 <- time2 
Acc_X1 <- Acc_X1[1:nrow(IMU2_data)]
Acc_Y1 <- Acc_Y1[1:nrow(IMU2_data)]
Acc_Z1 <- Acc_Z1[1:nrow(IMU2_data)]
Gyr_X1 <- Gyr_X1[1:nrow(IMU2_data)]
Gyr_Y1 <- Gyr_Y1[1:nrow(IMU2_data)]
Gyr_Z1 <- Gyr_Z1[1:nrow(IMU2_data)]
Mag_X1 <- Mag_X1[1:nrow(IMU2_data)]
Mag_Y1 <- Mag_Y1[1:nrow(IMU2_data)]
Mag_Z1 <- Mag_Z1[1:nrow(IMU2_data)]


# 2. Filter data ( filtrado de datos ) 

# Define the filter parameters ( definicion de parametros el filtro) 
#...
fs <-   # sampling frequency Frecuencia de muestreo 

fc <-    # cutoff frequency (below the typical frequency range of 
          # human motion - about 20 Hz, and the Earth's magnetic
          # field variations)

order <-   # filter order ( Orden del filtro)  

# Design the buttherworth filter ( diseño del filtro) 
b <- butter(order, fc/(fs/2), type="low")

# Apply the filter ( implementacion del filtro )
#...
filt_Acc_X1 <-
filt_Acc_Y1 <- 
filt_Acc_Z1 <- 
filt_Gyr_X1 <- 
filt_Gyr_Y1 <- 
filt_Gyr_Z1 <- 
filt_Mag_X1 <-
filt_Mag_Y1 <- 
filt_Mag_Z1 <- 

filt_Acc_X2 <- 
filt_Acc_Y2 <- 
filt_Acc_Z2 <- 
filt_Gyr_X2 <- 
filt_Gyr_Y2 <- 
filt_Gyr_Z2 <- 
filt_Mag_X2 <- 
filt_Mag_Y2 <- 
filt_Mag_Z2 <- 



# Calculate IMU angle from accelerometer, gyroscope, and magnetometer data (Calculo del ángulo del IMU del aceletrometro, gyroscopo y magnetometro)


# Obtain raw sensor readings from accelerometer, magnetometer, and gyroscope (# Obtener las lecturas brutas de los sensores acelerómetro, magnetómetro y giroscopio. )
# Merge filt_Acc_X1, filt_Acc_Y1, filt_Acc_Z1
# Merge filt_Mag_X1, filt_Mag_Y1, filt_Mag_Z1
# Merge filt_Gyr_X1, filt_Gyr_Y1, filt_Gyr_Z1  
# merge filt_Acc_X2, filt_Acc_Y2, filt_Acc_Z2)
# merge filt_Mag_X2, filt_Mag_Y2, filt_Mag_Z2)
# merge filt_Gyr_X2, filt_Gyr_Y2, filt_Gyr_Z2)
#...

accel1 <- 
mag1 <-  
gyro1 <- 

accel2 <- 
mag2 <-  
gyro2 <- 



# Initialize orientation estimate
#...
  
roll_est1 <- 
pitch_est1 <-
yaw_est1 <- 
accel_roll1 <- 
accel_pitch1 <- 

roll_est2 <- 
pitch_est2 <- 
yaw_est2 <- 
accel_roll2 <- 
accel_pitch2 <- 

# Initialize complementary filter constants
#...
  
alpha1 <- #Value for the filter
beta1 <- 1 - alpha1

alpha2 <- alpha1
beta2 <- 1 - beta1


# Loop over sensor readings

# Loop over sensor readings
for (i in 1:dim(accel1)-1) {
  
  # Estimate orientation using complementary filter
  accel_roll1[i] <- atan2(accel1[i,2], accel1[i,3]) * 180 / pi
  accel_pitch1[i] <- atan2(-accel1[i,1], sqrt(accel1[i,2]^2 + accel1[i,3]^2)) * 180 / pi
  
  roll_est1[i] <- alpha1 * (roll_est1 + gyro1[i,1]) + beta1 * accel_roll1
  pitch_est1[i] <- alpha1 * (pitch_est1 + gyro1[i,2]) + beta1 * accel_pitch1
  yaw_est1[i] <- yaw_est1 + gyro1[i,3]
  
}

for (i in 1:dim(accel2)-1) {
  
  # Estimate orientation using complementary filter
  accel_roll2[i] <- atan2(accel2[i,2], accel2[i,3]) * 180 / pi
  accel_pitch2[i] <- atan2(-accel2[i,1], sqrt(accel2[i,2]^2 + accel2[i,3]^2)) * 180 / pi
  
  roll_est2[i] <- alpha2 * (roll_est2 + gyro2[i,1]) + beta2 * accel_roll2
  pitch_est2[i] <- alpha2 * (pitch_est2 + gyro2[i,2]) + beta2 * accel_pitch2
  yaw_est2[i] <- yaw_est2 + gyro2[i,3]
  
}

#Time scale
time <- time1[1:length(time1)-1]

###
# Graph of time , pitch_est1 (2), time , roll_est1 (2) and time , yaw_est1 (2)
###

#...



# Angles from sensor 1 (thigh)
roll1 <- roll_est1
pitch1 <- pitch_est1
yaw1 <- yaw_est1

# Angles from sensor 2 (lower leg)
roll2 <- roll_est2
pitch2 <- pitch_est2
yaw2 <- yaw_est2



# Function to convert roll, pitch, and yaw angles to a rotation matrix (Función para convertir los ángulos de balanceo, cabeceo y rotacion en una matriz de rotación)

anglesToRotationMatrix <- function(roll, pitch, yaw) {
  # Convert angles to radians
  roll_rad <- roll * pi / 180
  pitch_rad <- pitch * pi / 180
  yaw_rad <- yaw * pi / 180
  
  # Calculate sin and cos values
  sr <- sin(roll_rad)
  cr <- cos(roll_rad)
  sp <- sin(pitch_rad)
  cp <- cos(pitch_rad)
  sy <- sin(yaw_rad)
  cy <- cos(yaw_rad)
  
  # Calculate rotation matrix
  R <- matrix(0, nrow = 3, ncol = 3)
  R[1, 1] <- cp * cy
  R[1, 2] <- cp * sy
  R[1, 3] <- -sp
  R[2, 1] <- sr * sp * cy - cr * sy
  R[2, 2] <- sr * sp * sy + cr * cy
  R[2, 3] <- sr * cp
  R[3, 1] <- cr * sp * cy + sr * sy
  R[3, 2] <- cr * sp * sy - sr * cy
  R[3, 3] <- cr * cp
  
  return(R)
}


# Function to calculate the angle between two sensors ( funcion para calcular el angulo entre dos vectores )
calculateAngleBetweenSensors <- function(roll1, pitch1, yaw1, roll2, pitch2, yaw2) {
  n <- length(roll1)  # Number of angles
  
  # Initialize result vectors
  roll_relative <- numeric(n)
  pitch_relative <- numeric(n)
  yaw_relative <- numeric(n)
  
  for (i in 1:n) {
    # Convert roll, pitch, and yaw angles to rotation matrices
    R1 <- anglesToRotationMatrix(roll1[i], pitch1[i], yaw1[i])
    R2 <- anglesToRotationMatrix(roll2[i], pitch2[i], yaw2[i])
    
    # Calculate the relative rotation matrix
    R_relative <- solve(R2) %*% R1
    
    # Extract roll, pitch, and yaw angles from the relative rotation matrix
    roll_relative[i] <- atan2(R_relative[2, 3], R_relative[3, 3])
    pitch_relative[i] <- -asin(R_relative[1, 3])
    yaw_relative[i] <- atan2(R_relative[1, 2], R_relative[1, 1])
  }
  
  return(list(roll_relative = roll_relative, pitch_relative = pitch_relative, yaw_relative = yaw_relative))
}


# Call the function with the given input vectors # Call the function with the given input vectors (Llamar a la función con los vectores de entrada dados) 

roll1 <- as.matrix(roll1)
pitch1 <- as.matrix(pitch1)
yaw1 <- as.matrix(yaw1)

roll2 <- as.matrix(roll2)
pitch2 <- as.matrix(pitch2)
yaw2 <- as.matrix(yaw2)

result <- calculateAngleBetweenSensors(roll1, pitch1, yaw1, roll2, pitch2, yaw2)

pitch_relative = result$roll_relative*180/pi
roll_relative = result$pitch_relative*180/pi
yaw_relative = result$yaw_relative*180/pi




# Set up a 3x1 plotting area ( configuracion de area de graficado)
#
#...

##
# Plot pitch_relative , time, roll_relative , time, yaw_relative , time
##
#...


# Define the filter parameters ( definicion de parametros el filtro) 
#...
fs <-   # sampling frequency Frecuencia de muestreo 
  
fc <-    # cutoff frequency (below the typical frequency range of 
  # human motion - about 20 Hz, and the Earth's magnetic
  # field variations)
  
order <-   # filter order ( Orden del filtro)  
  
  # Design the buttherworth filter ( diseño del filtro) 
b <- butter(order, fc/(fs/2), type="low")

# Apply the filter (implementacion del filtro)
filt_pitch <- filtfilt(b, pitch_relative)
filt_roll <- filtfilt(b, roll_relative)
filt_yaw <- filtfilt(b, yaw_relative)

# Plot the filtered angles filt_pitch,filt_roll, filt_yaw 

#...







### CODE FOR EMG ( codigo EMG)

# Set up a 2x1 plotting area

par(mfrow=c(3,1))

# 1. Read data

# Load EMG data into a data frame (Cargar datos EMG en una estructura de datos)


data_EMGMVCBF <- read_excel("R_BFMVC.xlsx")  
data_EMGMVCRF <- read_excel("R_RFMVC.xlsx") 
data_MOVEMENT <- read_excel("EMG_MOVEMENT.xlsx")

# Convert to numeric vector (Conversión de datos a una vectores numericos)

emg_MVCBF<- as.numeric(data_EMGMVCBF$`R BICEPS FEMORIS: EMG 2 [V]`)
timeMVCBF <- as.numeric(data_EMGMVCBF$`X [s]`)

emg_MVCRF<- as.numeric(data_EMGMVCRF$`R RECTUS FEMORIS: EMG 1 [V]`)
timeMVCRF <- as.numeric(data_EMGMVCRF$`X [s]`)

emg_MOVEMENTBF<- as.numeric(data_MOVEMENT$`R BICEPS FEMORIS: EMG 2 [V]`)
emg_MOVEMENTRF<- as.numeric(data_MOVEMENT$`R RECTUS FEMORIS: EMG 1 [V]`)
timeMOVEMENT <- as.numeric(data_MOVEMENT$`X [s]`)


# Plot raw (Gráfico señal bruta )

plot(timeMVCBF,emg_MVCBF, type= "l", xlab="Tiempo(s)", ylab="Amplitud(mV)")
title("EMG MVC Biceps Femoris")

plot(timeMVCRF,emg_MVCRF, type= "l", xlab="Tiempo(s)", ylab="Amplitud(mV)")
title("EMG MVC Rectus Femoris")

plot(timeMOVEMENT,emg_MOVEMENTBF, type= "l", xlab="Tiempo(s)", ylab="Amplitud(mV)")
title("EMG Movement Biceps Femoris")

plot(timeMOVEMENT,emg_MOVEMENTRF, type= "l", xlab="Tiempo(s)", ylab="Amplitud(mV)")
title("EMG Movement Rectus Femoris")

# 2. Filter EMG data (Filtrado datos EMG )

# Set filter parameters (Definición de los parametros del filtro)

fs <- 2160
lowcut <- 20
highcut <- 450
order <- 4


# Design a butterworth filter ( Diseño del filtro butterworth)

bf <- butter(order, c(lowcut, highcut)/(fs/2), type= "pass", plane ="z")


# Apply the filter to the data (Implementación de flitro a los datos) 

filteredMVCBF <- filtfilt(bf, emg_MVCBF)
filteredMVCRF <- filtfilt(bf, emg_MVCRF)
filteredMoveBF <- filtfilt(bf, emg_MOVEMENTBF)
filteredMoveRF <- filtfilt(bf, emg_MOVEMENTRF)

# Signal plot  ( Graficado de la señal )

plot(timeMVCBF,filteredMVCBF, type= "l", xlab="Tiempo(s)", ylab="Amplitud(mV)")
title("EMG MVC Biceps Femoris filtered")

plot(timeMVCRF,filteredMVCRF, type= "l", xlab="Tiempo(s)", ylab="Amplitud(mV)")
title("EMG MVC Rectus Femoris filtered")

plot(timeMOVEMENT,filteredMoveBF, type= "l", xlab="Tiempo(s)", ylab="Amplitud(mV)")
title("EMG Movement Biceps Femoris filtered")

plot(timeMOVEMENT,filteredMoveRF, type= "l", xlab="Tiempo(s)", ylab="Amplitud(mV)")
title("EMG Movement Rectus Femoris filtered")

# 3. Amplitude analysis ( Análisis de la amplitud) 


# 3.2 Envelope (Curva envolvente)

#Rectify

rectifiedMVCBF <- abs(filteredMVCBF)
rectifiedMVCRF <- abs(filteredMVCRF)
rectifiedMOVEBF <- abs(filteredMoveBF)
rectifiedMOVERF <- abs(filteredMoveRF)


plot(timeMVCBF,rectifiedMVCBF, type= "l", xlab="Tiempo(s)", ylab="Amplitud(mV)")
title("EMG MVC Biceps Femoris rectified")

plot(timeMVCRF,rectifiedMVCRF, type= "l", xlab="Tiempo(s)", ylab="Amplitud(mV)")
title("EMG MVC Rectus Femoris rectified")

plot(timeMOVEMENT,rectifiedMOVEBF, type= "l", xlab="Tiempo(s)", ylab="Amplitud(mV)")
title("EMG Movement Biceps Femoris rectified")

plot(timeMOVEMENT,rectifiedMOVERF, type= "l", xlab="Tiempo(s)", ylab="Amplitud(mV)")
title("EMG Movement Rectus Femoris rectified")

# Compute the power of the signal( Cálculo de la potencia de la señal, suavizado)

data_powerMVCBF <- filteredMVCBF^2
data_powerMVCRF <- filteredMVCRF^2
data_powerMoveBF <- filteredMoveBF^2
data_powerMoveRF <- filteredMoveRF^2

# Apply a moving window average to smooth the signal ( Aplicar una media de ventana móvil para suavizar la señal  )

window_size <- 0.1
window_size_samples <- window_size*fs
data_power_smoothedMVCBF <- stats::filter(data_powerMVCBF,rep(1/window_size_samples, window_size_samples), method = "conv")
data_power_smoothedMVCRF <- stats::filter(data_powerMVCRF,rep(1/window_size_samples, window_size_samples), method = "conv")
data_power_smoothedMoveBF <- stats::filter(data_powerMoveBF,rep(1/window_size_samples, window_size_samples), method = "conv")
data_power_smoothedMoveRF <- stats::filter(data_powerMoveRF,rep(1/window_size_samples, window_size_samples), method = "conv")



# Compute the square root of the averaged signal to obtain the RMS envelope ( Calcula la raíz cuadrada de la señal promediada para obtener la envolvente RMS )

data_rms_envMVCBF <- sqrt(data_power_smoothedMVCBF)
data_rms_envMVCRF <- sqrt(data_power_smoothedMVCRF)
data_rms_envMOVEBF <- sqrt(data_power_smoothedMoveBF)
data_rms_envMOVERF <- sqrt(data_power_smoothedMoveRF)

#crear un vector con na

non_naMVCBF <- !is.na(data_rms_envMVCBF)
non_naMVCRF <- !is.na(data_rms_envMVCRF)
non_naMoveBF <- !is.na(data_rms_envMOVEBF)
non_naMoveRF <- !is.na(data_rms_envMOVERF)

#subgrupo con el vector logico

data_rms_envMVCBF <- data_rms_envMVCBF[non_naMVCBF]
data_rms_envMVCRF <- data_rms_envMVCRF[non_naMVCRF]
data_rms_envMOVEBF <- data_rms_envMOVEBF[non_naMoveBF]
data_rms_envMOVERF <- data_rms_envMOVERF[non_naMoveRF]

# Plot the results ( Gráfica de resuktados) 

time_rmsMVCBF <- 1:length(data_rms_envMVCBF)/fs   # time definition (deficion del tiempo) 
time_rmsMVCRF <- 1:length(data_rms_envMVCRF)/fs
time_rmsMOVE <- 1:length(data_rms_envMOVEBF)/fs

plot(time_rmsMVCBF,data_rms_envMVCBF, type= "l", xlab="Tiempo(s)", ylab="Amplitud(mV)")
title("EMG MVC Biceps Femoris Smoothed")

plot(time_rmsMVCRF,data_rms_envMVCRF, type= "l", xlab="Tiempo(s)", ylab="Amplitud(mV)")
title("EMG MVC Rectus Femoris Smoothed")

plot(time_rmsMOVE,data_rms_envMOVEBF, type= "l", xlab="Tiempo(s)", ylab="Amplitud(mV)")
title("EMG Movement Biceps Femoris Smoothed")

plot(time_rmsMOVE,data_rms_envMOVERF, type= "l", xlab="Tiempo(s)", ylab="Amplitud(mV)")
title("EMG Movement Rectus Femoris Smoothed")


# 3.1 Normalization (Normalización)

# Normalize the signal by dividing by the maximum value (Normalizar la señal dividiéndola por el valor máximo)

max_MVCBF <- max(data_rms_envMVCBF)
max_MVCRF <- max(data_rms_envMVCRF)
max_MoveRF <- max(data_rms_envMOVERF)

normalized_dataMoveBF <- data_rms_envMOVEBF/max_MVCBF
normalized_dataMoveRF <- data_rms_envMOVERF/max_MoveRF


# Plot the original and normalized signals (# graficado  las señales filtrada  y normalizada )


par(mfrow=c(2,1))

plot(time_rmsMOVE,data_rms_envMOVEBF, type= "l", xlab="Tiempo(s)", ylab="Amplitud(mV)")
title("EMG Movement Biceps Femoris Smoothed")

plot(time_rmsMOVE,normalized_dataMoveBF, type= "l", xlab="Tiempo(s)", ylab="Amplitud(mV)")
title("EMG Movement Biceps Femoris Normalizada")

plot(time_rmsMOVE,data_rms_envMOVERF, type= "l", xlab="Tiempo(s)", ylab="Amplitud(mV)")
title("EMG Movement Rectus Femoris Smoothed")

plot(time_rmsMOVE,normalized_dataMoveRF, type= "l", xlab="Tiempo(s)", ylab="Amplitud(mV)")
title("EMG Movement Rectus Femoris Normalizada")


# 4. Spectral analysis ( Análisis espectral) 

# Perform Fourier Transform (Ejecución de la transformada rapida de fourier)

# Perform Fourier transform on filtered_data (Ejecución de la transformada rapida de fourieren la señal filtrada)

fft_data <- abs(fft(filtereddata))

# Calculate power spectrum ( Calculo del espectro de potencia)

power_spectrum <- abd(fft_data)^2

# Calculate the power spectrum using spec.pgram(Calculo del espectro de potencia usando sepc.gram)

psd <- spec.pgram(filtereddata, taper = 0, log = "no", fast= FALSE, detrend = FALSE,
                  xlab= "Fraquency (KHz)", ylab = "Spectrum (mV2/Hz)")

# Plot the power spectrum ( Gráfico del espectro de potencia) 

plot(psd$freq*1000, psd$spec, type = "l", xlab="Frequency (Hz)",
     ylab= "Power (mV2/Hz)", main = "Espectro de potencia EMG")

# Calculate frequency axis ( Cálculo de frecuencias) metricas

sampling_rate <- 2160
n <- length(fft_data)
frequency <-seq(0,100, length.out = length(power_spectrum))

# Calculate mean and median frequency ( Cálculo de la frecuencia media y la mediana)

fft_data <- Re(fft_data)
mean_frequency <- sum(frequency*fft_data[1:(n/2)]) / sum(fft_data[1:(n/2)])
cumulative_sum <- cumsum(power_spectrum)
median_frequency <- frequency[min(which(cumulative_sum >= sum(power_spectrum)/2))]

# Calculate peak power and total power ( Cálculo del pico máximo y total de la potencia)

peak_power <- max(fft_data[1:(n/2+1)])^2 / n
total_power <- sum(fft_data[1:n/2+1])^2 /n

# Print the results( Presentación de resultados )

cat("Mean Frequency:", mean_frequency, "Hz\n")
cat("Median Frequency:", median_frequency, "Hz\n")
cat("Peak Power:", peak_power, "mV2/Hz\n")
cat("Total Power:", total_power, "mV2/Hz\n")

# Time syncronization between IMU and EMG ( Sincronizacion de datos IMU y EMG)
# set a time constant to adjust the offset between IMU and EMG
#...

time_shift <-  # seconde
tRF <- timeRF + time_shift
tBF <- timeBF + time_shift
tIMU <- (1:length(filt_yaw)/60)

# Set up a 3x1 plotting area ( configuracion de area de graficado 3*1)
#...

#tRF , normalized_dataRF


#tBF , normalized_dataBF

#tIMU , filt_yaw



## Code to resample EMG to 60Hz to be at the same frequency as xsens data ( Resampleo de datos de EMG a la frecuencia de los IMU)

# Convert the time series object to a vector( # Convierte el objeto serie temporal en un vector) 

RF_rms_vec <- as.vector(normalized_dataRF)
BF_rms_vec <- as.vector(normalized_dataBF)


# Resample EMG signal to 60Hz ( resampleo a 60 Hz)

original_frequency <- 2160  

# Target frequency for resampling ( Frecuencia objetivo )

target_frequency <- 60  

# Compute the resampling ratio ( Calculo de proporcion de resampleo)

resampling_ratio <- original_frequency / target_frequency

# Resample the RMS signal vector to the target frequency (# Remuestrear el vector de señal RMS a la frecuencia objetivo.  )

resampled_rms_RF <- downsample(RF_rms_vec, resampling_ratio)
resampled_rms_BF <- downsample(BF_rms_vec, resampling_ratio)
resampled_tRF <- downsample(tRF, resampling_ratio)
resampled_tBF <- downsample(tBF, resampling_ratio)


############################################
# PLot EMG RMS and angle ( Dibujo RMS y EMG)
############################################


# Set up a 2x1 plotting area ( Configuracion de area de graficado )
#...
# tIMU, filt_yaw
plot(tIMU, filt_yaw, type = "l",  xlab="Time (s)", ylab="Angle (deg)", xlim = c(0,25))
title("Yaw")

# resampled_tRF, resampled_rms_RF

# Set up a 2x1 plotting area ( Configuracion de area de graficado )
#...

#Define de range for the analysis
#Define a windows around the test 
window_start_IMU <- #row number start
window_stop_IMU <- #row number stop
window_start_EMG <- #row number start (consider offset)
window_stop_EMG <- #row number stop (consider offset)

plot(filt_yaw[window_start_IMU:window_stop_IMU],resampled_rms_RF[window_start_IMU:window_stop_IMU],  xlab="Angle (Degrees)", ylab="EMG (rms)")
title("Relative angle Vs. Recto femoral RMS")

plot(filt_yaw[window_start_IMU:window_stop_IMU],resampled_rms_BF[window_start_IMU:window_stop_IMU],  xlab="Angle (Degrees)", ylab="EMG (rms)")
title("Relative angle Vs. Biceps femoral RMS")

###
# Find a relation  between filt_yaw and resampled_rms_BF or RF
###




