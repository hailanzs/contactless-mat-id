-- Radar Settings
-- 2 Tx 4 Rx | complex 1x
--------------------------------------------------------------
COM_PORT = 20 -- TODO: Put the COM port that the UART/User port appears at
RADARSS_PATH = "C:\\ti\\mmwave_studio_02_01_01_00\\rf_eval_firmware\\radarss\\xwr18xx_radarss.bin"
MASTERSS_PATH = "C:\\ti\\mmwave_studio_02_01_01_00\\rf_eval_firmware\\masterss\\xwr18xx_masterss.bin"
SAVE_DATA_PATH = "C:\\robotic-sensing-1\\raw-data\\adc_data.bin" -- TODO: put datapath to dump raw data (not used, just needs to be a valid path)

-------- VERY IMPORTANT AND SERIOUS RADAR SETTINGS --------
-- General
NUM_TX = 2
NUM_RX = 4

-- ProfileConfig
START_FREQ = 77 -- GHz
IDLE_TIME = 7 -- us
RAMP_END_TIME = 26.19 -- us
ADC_START_TIME = 7.07 --usn
FREQ_SLOPE = 149.957 -- MHz/us
ADC_SAMPLES = 64
SAMPLE_RATE = 3600 -- ksps
RX_GAIN = 40 -- dB

-- FrameConfig
START_CHIRP_TX = 0
END_CHIRP_TX = 1 -- 
NUM_FRAMES = 0 --
CHIRP_LOOPS = 1 --    
PERIODICITY = 2 -- ms
-----------------------------------------------------------

-------- THIS IS FINE --------
ar1.FullReset()
ar1.SOPControl(2)
ar1.Connect(COM_PORT,921600,1000)
------------------------------

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
ar1.Calling_IsConnected()
ar1.SelectChipVersion("AR1642")
ar1.frequencyBandSelection("77G")
ar1.SelectChipVersion("XWR1843")
ar1.SelectChipVersion("AR1642")
ar1.SelectChipVersion("XWR1843")
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

-------- DOWNLOAD FIRMARE --------
ar1.DownloadBSSFw(RADARSS_PATH)
ar1.GetBSSFwVersion()
ar1.GetBSSPatchFwVersion()
ar1.DownloadMSSFw(MASTERSS_PATH)
ar1.PowerOn(0, 1000, 0, 0)
ar1.RfEnable()
ar1.GetMSSFwVersion()
--------

-------- STATIC CONFIG STUFF --------
ar1.ChanNAdcConfig(1, 1, 1, 1, 1, 1, 1, 2, 1, 0) 
ar1.LPModConfig(0, 0)
ar1.RfInit()
--------------------------------------

-------- DATA CONFIG STUFF --------
ar1.DataPathConfig(1, 1, 0)
ar1.LvdsClkConfig(1, 1)
ar1.LVDSLaneConfig(0, 1, 1, 0, 0, 1, 0, 0)

-------- SENSOR CONFIG STUFF --------
ar1.ProfileConfig(0, START_FREQ, IDLE_TIME, ADC_START_TIME, RAMP_END_TIME, 0, 0, 0, 0, 0, 0, FREQ_SLOPE, 0, ADC_SAMPLES, SAMPLE_RATE, 0, 0, RX_GAIN)
ar1.ChirpConfig(0, 0, 0, 0, 0, 0, 0, 0, 1, 0)
ar1.ChirpConfig(1, 1, 0, 0, 0, 0, 0, 1, 0, 0)
ar1.FrameConfig(START_CHIRP_TX, END_CHIRP_TX, NUM_FRAMES, CHIRP_LOOPS, PERIODICITY, 0, 0, 1)
-------------------------------------

-------- ETHERNET STUFF --------
ar1.SelectCaptureDevice("DCA1000")
ar1.CaptureCardConfig_EthInit("192.168.33.30", "192.168.33.180", "12:34:56:78:90:12", 4096, 4098)
ar1.CaptureCardConfig_Mode(1, 2, 1, 2, 3, 30)
ar1.CaptureCardConfig_PacketDelay(75)
--------------------------------

ar1.CaptureCardConfig_StartRecord(SAVE_DATA_PATH, 1)
ar1.StartFrame()

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 

-------- CALCULATED AND NOT TOO SERIOUS PARAMETERS --------
CHIRPS_PER_FRAME = (END_CHIRP_TX - START_CHIRP_TX + 1) * CHIRP_LOOPS
NUM_DOPPLER_BINS = CHIRPS_PER_FRAME / NUM_TX
NUM_RANGE_BINS = ADC_SAMPLES
RANGE_RESOLUTION = (3e8 * SAMPLE_RATE * 1e3) / (2 * FREQ_SLOPE * 1e12 * ADC_SAMPLES)
MAX_RANGE = (300 * 0.9 * SAMPLE_RATE) / (2 * FREQ_SLOPE * 1e3)
DOPPLER_RESOLUTION = 3e8 / (2 * START_FREQ * 1e9 * (IDLE_TIME + RAMP_END_TIME) * 1e-6 * NUM_DOPPLER_BINS * NUM_TX)
MAX_DOPPLER = 3e8 / (4 * START_FREQ * 1e9 * (IDLE_TIME + RAMP_END_TIME) * 1e-6 * NUM_TX)

print("\n\nI wanted to make the outputted parameters easy to find so I put them inbetween two memes:")

print("Chirps Per Frame:", CHIRPS_PER_FRAME)
print("Num Doppler Bins:", NUM_DOPPLER_BINS)
print("Num Range Bins:", NUM_RANGE_BINS)
print("Range Resolution:", RANGE_RESOLUTION)
print("Max Unambiguous Range:", MAX_RANGE)
print("Doppler Resolution:", DOPPLER_RESOLUTION)
print("Max Doppler:", MAX_DOPPLER)
