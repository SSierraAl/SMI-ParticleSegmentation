#bokeh serve --show .\Interactive_Server.py     
#Necessary Libraries
from libraries_Import import *
from functions import *
from Individual_Anaysis import *

###############################################################################################
#Parameters  ##################################################################################
###############################################################################################
NewData=True
Adq_Freq=2000000
Time_Captured_ms=16384
order_filt = 4
freq_cut_inf=7000 
freq_cut_sup=100000 


Particle_Params = {
  "laser_lambda": 1.55e-6,
  "angle": 22,
  "beam_spot_size": 90e-6,
}

###############################################################################################
#Load Data  ###################################################################################
###############################################################################################

path_folder="./Particle_Segmentation/Particles_Data/"


Name_Folder=['Raw_2um',
             'Raw_4um',
             'Raw_10um'
             ]

Upper_Lim_Freq,  Lower_Lim_Freq, Upper_Lim_Freq_Soft, Lower_Lim_Freq_Soft = Get_Tab_0_1(path_folder, Name_Folder, freq_cut_inf, freq_cut_sup,order_filt,Adq_Freq,NewData)

# Overwrite the same bandwidth filter to have the same for each different particles
Upper_Lim_Freq=[freq_cut_sup,freq_cut_sup,freq_cut_sup]
Lower_Lim_Freq=[freq_cut_inf,freq_cut_inf,freq_cut_inf]

tab2 = Get_Time_Tab(path_folder, Name_Folder[0],order_filt,Adq_Freq,NewData,Upper_Lim_Freq[0],Lower_Lim_Freq[0], Time_Captured_ms,Particle_Params)

tab3 = Get_Time_Tab(path_folder, Name_Folder[1], order_filt,Adq_Freq,NewData,Upper_Lim_Freq[1],Lower_Lim_Freq[1],Time_Captured_ms,Particle_Params)

tab4 = Get_Time_Tab(path_folder, Name_Folder[2], order_filt,Adq_Freq,NewData,Upper_Lim_Freq[2],Lower_Lim_Freq[2],Time_Captured_ms,Particle_Params)

#tab5 = Get_Analysis(path_folder, Name_Folder, freq_cut_inf, freq_cut_sup,order_filt,Adq_Freq,Time_Captured_ms,Particle_Params)
    
Tabs_Final=Tabs(tabs=[tab2, tab3, tab4])

# Description
desc = Div(text=open("description.html").read(), sizing_mode="stretch_width")

lay=layout([[desc],
            [Tabs_Final],])

doc = curdoc()
doc.add_root(lay)
