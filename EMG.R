# 0. Install and call the packages (Instalacion y cargado de paquetes : signal, tidyverse, seewave,zoo,pracma,gsignal) 
install.packages("signal")
install.packages("tidyverse")
install.packages("seewave")
install.packages("zoo")
install.packages("pracma")
install.packages("gsignal")
install.packages("readxl")

library(signal)
library(tidyverse)
library(seewave)
library(zoo)
library(pracma)
library(gsignal)
library(readxl)



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
