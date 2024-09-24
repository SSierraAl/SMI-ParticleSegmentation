#Necessary Libraries
from libraries_Import import *
from functions import *



def Get_Time_Tab(path_folder, Name_Folder,order_filt,Adq_Freq,NewData, Upper_Lim_Freq, Lower_Lim_Freq,Time_Captured_ms,Particle_Params):
    
    global Counter_Column
    #global filterImpact
    Counter_Column=1
    path_folder=path_folder+Name_Folder

    Original_Dataframe=Load_New_Data(path_folder,NewData)
    columnas = Original_Dataframe.columns.tolist()
    selected_column=columnas[0]


    # Random Data from begining
    data_Y = Original_Dataframe[selected_column]
    # Milliseconds scale
    data_X_max=Time_Captured_ms * 1000 / Adq_Freq
    data_X=np.linspace(0, data_X_max, Time_Captured_ms)

    #Create Source Column
    ampFFT1, freqfft1,_ =FFT_calc(data_Y, Adq_Freq)
    source_filtered_FFT = ColumnDataSource(data= dict(data_x_axis=freqfft1, daya_y_axis=ampFFT1))
    source_original_data= ColumnDataSource(data= dict(data_x_axis=data_X, daya_y_axis=data_Y))
    source_filtered_data= ColumnDataSource(data= dict(data_x_axis=data_X, daya_y_axis=data_Y))

    ###############################################################################################
    #Widgets  ####################################################################################
    ###############################################################################################

    scroll_menu = Select(title="Select a column:", options=columnas)
    Low_Freq = TextInput(title="Low Freq [Hz]:", value=str(Lower_Lim_Freq))
    High_Freq = TextInput(title="High Freq [Hz]:", value=str(Upper_Lim_Freq))
    button_refresh = Button(label="Refresh")
    button_load_sample = Button(label="Load Sample")


    ###############################################################################################
    #Graphics  ####################################################################################
    ###############################################################################################
    #Global FFT Filtered Plot Info ########################################################################################
    FilterFFT = figure(title=f'FFT Filtered', x_axis_label='[Hz]', y_axis_label='Amplitud',width=1400, height=200)
    FilterFFT1=FilterFFT.line('data_x_axis', 'daya_y_axis', source=source_filtered_FFT, line_width=2, line_color='blue')
    hover_tool = HoverTool(tooltips=[("Freq", "@data_x_axis"), ("Value", "@daya_y_axis")])
    FilterFFT.add_tools(hover_tool)
    FilterFFT.xaxis.formatter = NumeralTickFormatter(format="0a")

    # Filter Plot Info ########################################################################################
    filterImpact = figure(title='File', x_axis_label='[ms]', y_axis_label='[V]',width=1400, height=250)
    filterImpact2=filterImpact.line('data_x_axis','daya_y_axis',source=source_filtered_data,line_width=2, legend_label='Filtered', line_color='blue')

    hover_tool = HoverTool(tooltips=[("ms: ", "@data_x_axis"), ("V: ", "@daya_y_axis")])
    filterImpact.add_tools(hover_tool)
    filterImpact.legend.location = 'top_left'

    # Filter Plot Info ########################################################################################
    filterImpact_Second_Peak = figure(title='Second peak detected', x_axis_label='[ms]', y_axis_label='[V]',width=1400, height=250)
    filterImpact2_Second_Peak=filterImpact_Second_Peak.line('data_x_axis','daya_y_axis',source=source_filtered_data,line_width=2, legend_label='Filtered', line_color='blue')

    hover_tool = HoverTool(tooltips=[("ms: ", "@data_x_axis"), ("V: ", "@daya_y_axis")])
    filterImpact_Second_Peak.add_tools(hover_tool)
    filterImpact_Second_Peak.legend.location = 'top_left'

    # Spectogram ########################################################################################
    spectogram_fig = figure(title='Spectogram', x_axis_label='[ms]', y_axis_label='[Hz]',width=1400, height=250)
    
    # Histogram Selection
    TOOLS="pan,wheel_zoom,box_select,lasso_select,reset"
    Histo_Select = figure(tools=TOOLS, width=800, height=600, min_border=10, min_border_left=50,
           toolbar_location="above", x_axis_location=None, y_axis_location=None,
           title="Linked Histograms")
    Histo_Select.select(BoxSelectTool).continuous = False
    Histo_Select.select(LassoSelectTool).continuous = False

    Histo_X_Axis = figure(title='X_Axis', x_axis_label='X_Axis', y_axis_label='Frequency',x_range=Histo_Select.x_range, min_border=10, min_border_left=50, y_axis_location="right",width=870, height=250)
    Histo_X_Axis.yaxis.major_label_orientation = np.pi/4
    Histo_Y_Axis = figure(title='Y_Axis', x_axis_label='Frequency', y_axis_label='Y_Axis',y_range=Histo_Select.y_range, min_border=10, y_axis_location="right",width=250)
    Histo_Y_Axis.xaxis.major_label_orientation = np.pi/4

    TexInfo=' Doppler Peak: '+'0.00'+' [Hz]'+ ' Particle Speed: '+'0'+' [m/s]' +' '+'Passage time: '+' 0 '+'[ms]'+' '+'Estimate Size: '+' 0 '+'[um]'
    Data_Info = PreText(text=TexInfo,width=60, height=30)


    ###############################################################################################
    #Functions  ####################################################################################
    ###############################################################################################
   
    def refresh_Filter():
        #Clear Graph
        filterImpact.renderers = []
        filterImpact_Second_Peak.renderers = []

        Green_Group=0
        Yellow_Group=0
        Red_Group=0

        data_X=np.linspace(0, data_X_max, Time_Captured_ms)
        #Load New Column
        global Counter_Column
        Counter_Column=1
        selected_column = scroll_menu.value
        data_Y = Original_Dataframe[selected_column]
        freq_cut_inf = int(Low_Freq.value)
        freq_cut_sup = int(High_Freq.value)

        #Remove Mean
        data_Y_NoOffset=data_Y
        data_Y_NoOffset = data_Y_NoOffset - data_Y_NoOffset.mean()
        y_Filtrada= butter_bandpass_filter(data_Y_NoOffset, freq_cut_inf, freq_cut_sup, order_filt, Adq_Freq)
        filterImpact.line(data_X,y_Filtrada,line_width=2, line_color='blue')

        #Time Dommain
        filterImpact.title.text = str(selected_column)
        # Calcular la Transformada de Hilbert de la señal
        signal_hilbert = hilbert(y_Filtrada)
        # La señal analítica es la combinación de la señal original y su Transformada de Hilbert
        signal_analytic = y_Filtrada + 1j * signal_hilbert
        instantaneous_phase = (np.angle(signal_analytic))
        instantaneous_phase = (instantaneous_phase + np.pi) % (2 * np.pi) - np.pi
        phase_derivative = np.diff(instantaneous_phase) / np.diff(data_X)
        phase_derivative=phase_derivative/max(phase_derivative)
        
        filterImpact2.data_source.data['data_x_axis'] = data_X
        filterImpact2.data_source.data['daya_y_axis'] = y_Filtrada
        #Frequency Domain
        ampFFT1, freqfft1,_ =FFT_calc(y_Filtrada, Adq_Freq)
        FilterFFT1.data_source.data['data_x_axis'] = freqfft1
        FilterFFT1.data_source.data['daya_y_axis'] = ampFFT1

        #Spectogram #####################################################################################
        #find number of peaks
        std_dev = 20  # Desviación estándar para el filtro gaussiano
        cut_index=int(len(ampFFT1)/10)
        new_FFT=ampFFT1[0:cut_index]
        filtered_fft = gaussian_filter1d(new_FFT, sigma=std_dev)
        
        # Calcular la amplitud media de la señal suavizada
        #mean_amplitude = (max(filtered_fft)/3)*2
        mean_amplitude = (max(filtered_fft)/2)
        peaks, properties = find_peaks(filtered_fft, height=mean_amplitude)
        peak_amplitudes = properties["peak_heights"]
        # Ordenar los picos por su amplitud de forma descendente
        sorted_indices = np.argsort(peak_amplitudes)[::-1]  # Agrega [::-1] para orden descendente
        sorted_peaks = peaks[sorted_indices]
        # Filtrar los picos para asegurar un espaciado mínimo de 12 kHz
        min_distance = 12000  # Espaciado mínimo en Hz
        valid_peaks = []
        for peak in sorted_peaks:
            # Comprobar espaciado con respecto a cada pico ya en valid_peaks
            if all(abs(freqfft1[peak] - freqfft1[vp]) >= min_distance for vp in valid_peaks):
                valid_peaks.append(peak)

        # Si se desea mantener el orden original de los picos (descendente por amplitud) después de aplicar el filtro de distancia, este paso ya logra eso.
        print("Frecuencias de los picos principales (orden descendente por amplitud):", freqfft1[valid_peaks])
        #################################
        #################################

        t, f_new, Sxx_new=find_spectogram(y_Filtrada, Adq_Freq,ampFFT1, freqfft1, Particle_Params)
        general_mean=(np.mean(Sxx_new, axis=0))
        #Reduce limit for selection selection
        #mean_spectogram= np.mean(Sxx_new[Sxx_new < mean_spectogram],axis=0)
        t = np.linspace(0, max(data_X), len(general_mean))
        color_mapper = LinearColorMapper(palette=Viridis256, low=(np.min(Sxx_new)), high=(np.max(Sxx_new)))
        img_spectogram=spectogram_fig.image(image=[Sxx_new], x=0, y=0, dw=t[-1], dh=f_new[-1], color_mapper=color_mapper)

        #Anomalie its if there are multiple pixels together over the mean spectogra
        valid_zones, anomalies = classify_zones(general_mean, [5,10])
        _,_,_=particle_anomalies_validation(anomalies,valid_zones,data_X,y_Filtrada,t,filterImpact,Green_Group, Yellow_Group, Red_Group)

        #########################################################
        ##########################################################

        
        if len(valid_peaks)>1:
            print('picos')
            print(freqfft1[valid_peaks][1])
            print(freqfft1[valid_peaks][0])
            two_peak=int(freqfft1[valid_peaks][1])
            min_lim=(two_peak)-12000
            if min_lim<7000:
                min_lim=7000
            max_lim=two_peak+12000  
            data_Y_NoOffset=data_Y
            data_Y_NoOffset = data_Y_NoOffset - data_Y_NoOffset.mean()
            y_Filtrada= butter_bandpass_filter(data_Y_NoOffset, min_lim, max_lim, order_filt, Adq_Freq)

            data_X=np.linspace(0, data_X_max, Time_Captured_ms)
            filterImpact_Second_Peak.line(data_X, y_Filtrada, line_width=2, line_color="blue")
            filterImpact2_Second_Peak.data_source.data['data_x_axis'] = data_X
            filterImpact2_Second_Peak.data_source.data['daya_y_axis'] = y_Filtrada
            

            ampFFT1, freqfft1,_ =FFT_calc(y_Filtrada, Adq_Freq)
            #New Spectogram
            t, f_new, Sxx_new=find_spectogram(y_Filtrada, Adq_Freq,ampFFT1, freqfft1, Particle_Params)
            general_mean=(np.mean(Sxx_new, axis=0))
            #Reduce limit for sleection selection
            t = np.linspace(0, max(data_X), len(general_mean))
            color_mapper = LinearColorMapper(palette=Viridis256, low=(np.min(Sxx_new)), high=(np.max(Sxx_new)))
            #Anomalie if there are multiple pixels together over the mean spectogram
            valid_zones, anomalies = classify_zones(general_mean, [5,10])
            _,_,_=particle_anomalies_validation(anomalies,valid_zones,data_X,y_Filtrada,t,filterImpact_Second_Peak,Green_Group, Yellow_Group, Red_Group)


    def load_sample():

        selected_row_indices = source_Table.selected.indices
        df_data = pd.DataFrame(source_Table.data)
        sel_col=df_data['name'][selected_row_indices]
        sel_col=sel_col.values[0]
        data_Y = Original_Dataframe[sel_col]
        
        freq_cut_inf = int(Low_Freq.value)
        freq_cut_sup = int(High_Freq.value)
        #Remove Mean
        data_Y_NoOffset=data_Y
        data_Y_NoOffset = data_Y_NoOffset - data_Y_NoOffset.mean()
        y_Filtrada= butter_bandpass_filter(data_Y_NoOffset, freq_cut_inf, freq_cut_sup, order_filt, Adq_Freq)

        #Time Dommain
        filterImpact.title.text = str(sel_col)
        filterImpact2.data_source.data['data_x_axis'] = data_X
        filterImpact2.data_source.data['daya_y_axis'] = y_Filtrada

        #Frequency Domain
        ampFFT1, freqfft1,_ =FFT_calc(y_Filtrada, Adq_Freq)
        FilterFFT1.data_source.data['data_x_axis'] = freqfft1
        FilterFFT1.data_source.data['daya_y_axis'] = ampFFT1


        #Spectogram #####################################################################################
        t, f_new, Sxx_new=find_spectogram(y_Filtrada, Adq_Freq,ampFFT1, freqfft1, Particle_Params)
        general_mean=(np.mean(Sxx_new, axis=0))
        mean_spectogram=np.mean(general_mean)
        #Reduce limit for sleection selection
        #mean_spectogram= np.mean(Sxx_new[Sxx_new < mean_spectogram],axis=0)
        t = np.linspace(0, max(data_X), len(general_mean))
        color_mapper = LinearColorMapper(palette=Viridis256, low=(np.min(Sxx_new)), high=(np.max(Sxx_new)))
        img_spectogram=spectogram_fig.image(image=[Sxx_new], x=0, y=0, dw=t[-1], dh=f_new[-1], color_mapper=color_mapper)
        
        ########################################################################################################

    ###############################################################################################
    #Structure & Connections  #####################################################################
    ###############################################################################################
    button_refresh.on_click(refresh_Filter)
    button_load_sample.on_click(load_sample) 


    lay=layout([[scroll_menu, High_Freq, Low_Freq,button_refresh],
                [FilterFFT],
                [filterImpact],
                [spectogram_fig],
                [filterImpact_Second_Peak]
                ])
    

    lay = TabPanel(child=lay, title=Name_Folder)
    return lay




def Get_Tab_0_1(path_folder, Name_Folder, freq_cut_inf, freq_cut_sup,order_filt,Adq_Freq,NewData):

    Upper_Lim_Freq=[]
    Lower_Lim_Freq=[]
    Upper_Lim_Freq_Soft=[]
    Lower_Lim_Freq_Soft=[] 

    for i in range(0,len(Name_Folder)):
        dir_folder=path_folder+Name_Folder[i]
        Data_Array=GetLoad_Folder(dir_folder,Name_Folder[i],NewData)
        Avg_FFT, Avg_Freq=AverageFFT(Data_Array,freq_cut_inf, freq_cut_sup, order_filt, Adq_Freq)
        if Avg_FFT.empty:
            print('empty folder ' + Name_Folder[i])
        else:
            window_size = 5
            window = np.ones(window_size) / window_size
            Avg_FFT = np.convolve(Avg_FFT, window, mode='valid')
            Avg_Freq = np.convolve(Avg_Freq, window, mode='valid')

            source_Avg_FFT = ColumnDataSource(data= dict(data_x_axis=Avg_Freq, daya_y_axis=Avg_FFT))
            OriginalFFT = figure(title=(Name_Folder[i]), x_axis_label='[Hz]', y_axis_label='Amplitud',width=1400, height=250)
            OriginalFFT1 =OriginalFFT.line('data_x_axis', 'daya_y_axis', source=source_Avg_FFT, line_width=2)
            max_fwhm, min_fwhm = get_full_width(OriginalFFT1.data_source.data['data_x_axis'], OriginalFFT1.data_source.data['daya_y_axis'])
            ####################################################################
            #Limits for Particle Detection
            Upper_Lim_Freq.append(int(max_fwhm))
            Lower_Lim_Freq.append(int(min_fwhm))
            #Limits for analysis of fringes, soft limits
            Upper_Lim_Freq_Soft.append(int(max_fwhm)+40000)
            Lower_Lim_Freq_Soft.append(int(min_fwhm)-5000)
            ####################################################################
    return Upper_Lim_Freq,  Lower_Lim_Freq, Upper_Lim_Freq_Soft, Lower_Lim_Freq_Soft
