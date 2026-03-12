from bokeh.models import Select, NumberFormatter,NumeralTickFormatter,Button, ColumnDataSource, TextInput, DataTable, TableColumn, Div, NumeralTickFormatter, HoverTool, BoxAnnotation, RangeTool
from bokeh.plotting import figure, curdoc
from bokeh.models import Button, TextInput,DataTable, NumberFormatter, StringFormatter, StringEditor, NumberEditor,TableColumn
from bokeh.models import TapTool, Span,LinearColorMapper,LassoSelectTool, CustomJS, MultiChoice, HoverTool,PreText, NumeralTickFormatter,BoxSelectTool, TextInput,TabPanel, Tabs


from bokeh.layouts import layout, column
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import hilbert
from Support_Functions import Load_New_Data, FFT_calc, butter_bandpass_filter
import os
from scipy.signal import find_peaks, hilbert

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-(x - mean)**2 / (2 * stddev**2))

def Set_Reader_Data(path, parameters):
    global Original_Dataframe  #
    path_folder = path
    Adq_Freq = int(parameters['acquisition_frequency'])
    order_filt = int(parameters['bandpass_filter_order'])
    Low_Freq = int(parameters['band_low_filter_frequency'])
    High_Freq = int(parameters['band_high_filter_frequency'])

    # Load Data and Initial Setup
    Original_Dataframe = Load_New_Data(path_folder, True)
    columnas = Original_Dataframe.columns.tolist()
    Counter_Column = 0
    selected_column = columnas[Counter_Column]
    data_Y = Original_Dataframe[selected_column]
    data_X = (np.arange(len(data_Y)) / Adq_Freq) * 1000  # Convert to milliseconds

    
    # ColumnDataSource Setup
    def create_source(data_X, data_Y):
        return ColumnDataSource(data=dict(data_x_axis=data_X, daya_y_axis=data_Y))

    source_original_data = create_source(data_X, data_Y)
    source_filtered_data = create_source(data_X, data_Y)
    source_gaussian_data = create_source(data_X, data_Y)
    ampFFT1, freqfft1, _ = FFT_calc(data_Y, Adq_Freq)

    # FFT sources
    source_original_FFT = create_source(freqfft1, ampFFT1)
    source_filtered_FFT = create_source(freqfft1, ampFFT1)
    source_original_section_FFT = create_source(freqfft1, ampFFT1)
    source_filtered_section_FFT1 = create_source(freqfft1, ampFFT1)
    source_filtered_section_FFT2 = create_source(freqfft1, ampFFT1)

    # Widget Creation
    scroll_menu = Select(title="Select a column:", options=columnas)
    button_refresh = Button(label="Refresh")
    button_next = Button(label="Next")
    button_section = Button(label="Section")
    button_prev = Button(label="Prev")
    button_save_table = Button(label="Save Table")

    N_bins = TextInput(title="N_bins", value="20")

    # Plot Creation Function to Avoid Redundancy
    def create_figure(title, x_axis, y_axis, width, height, source, color="blue", hover=True):
        p = figure(title=title, x_axis_label=x_axis, y_axis_label=y_axis, width=width, height=height)
        line = p.line('data_x_axis', 'daya_y_axis', source=source, line_width=2, line_color=color)
        if hover:
            p.add_tools(HoverTool(tooltips=[(x_axis, "@data_x_axis"), (y_axis, "@daya_y_axis")]))
        p.xaxis.formatter = NumeralTickFormatter(format="0.00a")
        return p, line


    def adjust_axis_labels_and_ticks(fig):
        """Helper function to set font and size for all axes and titles in a figure."""
        
        # Adjust axis labels (font and size)
        fig.xaxis.axis_label_text_font = "times"
        fig.yaxis.axis_label_text_font = "times"
        fig.xaxis.axis_label_text_font_size = "12pt" #18
        fig.yaxis.axis_label_text_font_size = "12pt" #18
        
        fig.xaxis.major_label_text_font = "times"
        fig.yaxis.major_label_text_font = "times"
        fig.xaxis.major_label_text_font_size = "14pt" #20
        fig.yaxis.major_label_text_font_size = "14pt" #20

        # Set the font style to 'normal' to avoid italic (cursive) for axis labels
        fig.xaxis.axis_label_text_font_style = "normal"
        fig.yaxis.axis_label_text_font_style = "normal"
        fig.xaxis.major_label_text_font_style = "normal"
        fig.yaxis.major_label_text_font_style = "normal"
        
        
        # Adjust title (font, size, and style)
        fig.title.text_font = "times"
        fig.title.text_font_size = "14pt" #20
        fig.title.text_font_style = "normal"  # Set title to normal (not bold)



    # Figures with glyphs
    FilterFFT, FilterFFT1 = create_figure('FFT Global', 'Freq. [Hz]', 'Amp. [a.u.]', 700, 250, source_filtered_FFT, color='blue')
    Section_Particle, line_Particle = create_figure('OFI Signal', 'Time [ms]', 'Amp. [mV]', 700, 250, source_original_data)
    Section_Particle_Filtered, line_Particle_Filtered = create_figure('OFI Filtered', 'Time [ms]', 'Amp. [mV]', 700, 250, source_filtered_data, color='orange')
    filterImpact, filterImpact1 = create_figure('Filter vs Original', '[ms]', '[mV]', 1400, 250, source_original_data)
    filterImpact3 = filterImpact.line('data_x_axis', 'daya_y_axis', source=source_gaussian_data, legend_label='Gaussian fit', line_color='green')
    filterImpact2 = filterImpact.line('data_x_axis', 'daya_y_axis', source=source_filtered_data, legend_label='Filtered', line_color='orange')
    
    Section_Particle_Filtered.x_range=Section_Particle.x_range
    # Range Tool Setup
    range_overlay = BoxAnnotation(fill_color="navy", fill_alpha=0.1)
    range_tool = RangeTool(x_range=Section_Particle.x_range, overlay=range_overlay)
    #range_tool = RangeTool(x_range=Section_Particle_Filtered.x_range, overlay=range_overlay)
    filterImpact.add_tools(range_tool)
    
    # Section FFT Figures
    SectionFFT, Section_FFT1 = create_figure('FFT Filtered', '[Hz]', 'Amp. [a.u.]', 700, 250, source_original_section_FFT)
    Section_FilteredFFT, Section_FilterFFT1 = create_figure('Section FFT Filtered', ' Freq. [Hz]', 'Amp. [a.u.]', 700, 250, source_filtered_section_FFT1, color='blue')
    Section_FilterFFT2 = Section_FilteredFFT.line('data_x_axis', 'daya_y_axis', source=source_filtered_section_FFT2, line_width=2, line_color='orange')

    #Histogram
    Histogram = figure(title=('Histogram Analysis'), x_axis_label='Bins', y_axis_label='Events',width=1300, height=400)



    # Apply the formatting to all figures
    for fig in [FilterFFT, Section_Particle, Section_Particle_Filtered, filterImpact, SectionFFT, Section_FilteredFFT, Histogram]:
        adjust_axis_labels_and_ticks(fig)



    #############################################################################
    # TABLE Widget
    #############################################################################

    button_load_sample = Button(label="Load Sample")
    Particle_Data_Table = {
        'name': ['--'],
        'val_Freq': [0.1],
        'amp_Freq': [0.1],
        'amp_Time': [0.1],
        'num_burst': [0.1],
        'passage' : [0.1]
    }
    df_moments = pd.DataFrame(Particle_Data_Table)
    source_Table = ColumnDataSource(df_moments)

    tabla = DataTable(source=source_Table, width=1400, height= 250,columns=[
        TableColumn(field='name', title='Name File', editor=StringEditor(), formatter=StringFormatter()),
        TableColumn(field='val_Freq', title='Mean Freq [kHz]', editor=NumberEditor(), formatter=NumberFormatter(format='0.000000')),
        TableColumn(field='amp_Freq', title='Amp Peak Freq', editor=NumberEditor(), formatter=NumberFormatter(format='0.000000')),
        TableColumn(field='amp_Time', title='Amp Fringes', editor=NumberEditor(), formatter=NumberFormatter(format='0.000000')),
        TableColumn(field='num_burst', title='Num Fringes', editor=NumberEditor(), formatter=NumberFormatter(format='0.000000')),
        TableColumn(field='passage', title='Passage', editor=NumberEditor(), formatter=NumberFormatter(format='0.000000')),
        ])

    columnas = Original_Dataframe.columns.tolist()
    for i in columnas:
        data_Y = Original_Dataframe[i]
        data_Y = data_Y.fillna(0)
        Adq_Freq = int(parameters['acquisition_frequency'])
        order_filt = int(parameters['bandpass_filter_order'])
        Low_Freq = int(parameters['band_low_filter_frequency'])
        High_Freq = int(parameters['band_high_filter_frequency'])
        # Remove Mean and Apply Filter
        data_Y_NoOffset = data_Y - data_Y.mean()
        y_Filtrada = butter_bandpass_filter(data_Y_NoOffset, Low_Freq, High_Freq, order_filt, Adq_Freq)
        ampFFT1, freqfft1,_ =FFT_calc(y_Filtrada, Adq_Freq)


        # Count the number of fringes within the FWHM range

        #Previous version Sebas
        """
        def count_peaks_with_gaussian(signal, data_X, main_freq):
            # Step 1: Preprocess the signal by setting all values less than zero to zero
            processed_signal = np.maximum(signal, 0)
            valid_P_y2 = butter_bandpass_filter(processed_signal, main_freq-3000, main_freq+3000, order_filt, Adq_Freq)
            # Step 2: Perform Gaussian fitting on the envelope of the signal
            #valid_P_y2 = np.abs(hilbert(valid_P_y2))  # Hilbert transform for envelope
            valid_P_y2_count=np.maximum(valid_P_y2, 0)

            try:
                # Fit Gaussian to the signal envelope
                optimized_params, _ = curve_fit(gaussian, data_X, valid_P_y2_count, maxfev=10000)
                amplitude, mean, stddev = optimized_params
                amplitude_normalized = (amplitude / np.max(amplitude)) * max(signal)
                #stddev=stddev*2/3
                y_curve = gaussian(data_X, amplitude_normalized, mean, stddev)


            except Exception as e:
                print("Gaussian fit failed:", e)
                return 0, 0  # Return zero peaks if the fit fails

            # Calcular el umbral como la mitad de la altura máxima de la gaussiana
            threshold = y_curve.max() / 4
            # Encuentra los índices donde la curva cruza el umbral
            above_threshold = y_curve > threshold
            crossing_indices = np.where(np.diff(above_threshold.astype(int)) != 0)[0]

            # Verificamos que hay dos cruces
            if len(crossing_indices) >= 2:
                # Primer y segundo cruce del umbral
                first_crossing = crossing_indices[0]
                second_crossing = crossing_indices[1]
                
                # Rango delimitado para el conteo de picos
                valid_range = valid_P_y2_count[first_crossing:second_crossing + 1]

                # Detectar y contar los picos en el rango seleccionado
                peak_count = 0
                for i in range(1, len(valid_range) - 1):
                    # Condición de un pico: el valor actual es mayor a cero y es un punto de subida
                    if valid_range[i-1] <= 0.00000 and valid_range[i] > 0.0000000:
                        peak_count += 1

                # Calcular el ancho en milisegundos entre los cruces
                width_points = ((len(valid_range)/Adq_Freq) * 1000)
            else:
                peak_count = 0
                width_points = 0


            return peak_count, width_points
        """
        def count_zero_crossing_peaks_numpy(signal, min_amplitude=0.05):
            signal = np.asarray(signal)

            # Find zero crossings: -1 → +1 transitions and back
            below_zero = signal <= 0
            above_zero = signal > 0

            # Find rising edge (crossing up) and falling edge (crossing down)
            crossings_up = np.where((signal[:-1] <= 0) & (signal[1:] > 0))[0]
            crossings_down = np.where((signal[:-1] > 0) & (signal[1:] <= 0))[0]

            # Count full peak cycles (up then down)
            peak_count = 0
            i = j = 0
            while i < len(crossings_up) and j < len(crossings_down):
                if crossings_down[j] > crossings_up[i]:
                    # Check the peak height between up and down
                    peak_region = signal[crossings_up[i]+1:crossings_down[j]+1]
                    if peak_region.max() >= min_amplitude:
                        peak_count += 1
                    i += 1
                j += 1

            return peak_count
        
        def count_peaks_with_gaussian(signal, data_X, main_freq):
            # Parameters
            Adq_Freq = int(parameters['acquisition_frequency'])  # Sampling frequency
            order_filt = int(parameters['bandpass_filter_order'])
            
            # Step 1: Preprocess the signal
            processed_signal = signal  # Keep original signal for peak counting
            # Apply bandpass filter around main frequency
            valid_P_y2 = butter_bandpass_filter(processed_signal, main_freq-2500, main_freq+2500, 3, Adq_Freq)
            
            # Step 2: Compute the signal envelope using Hilbert transform
            envelope = np.abs(hilbert(valid_P_y2))
            
            # Step 3: Fit Gaussian to the envelope to identify the signal region
            try:
                # Initial guess for Gaussian parameters
                initial_guess = [np.max(envelope), data_X[np.argmax(envelope)], np.std(data_X)/2]
                optimized_params, _ = curve_fit(gaussian, data_X, envelope, p0=initial_guess, maxfev=10000)
                amplitude, mean, stddev = optimized_params
                y_curve = gaussian(data_X, amplitude, mean, stddev)
            except Exception as e:
                print("Gaussian fit failed:", e)
                return 0, 0  # Return zero peaks if fit fails
            
            # Step 4: Define threshold to select the wavelet burst region
            threshold = y_curve.max() * 0.15   # Use 10% of max Gaussian amplitude as threshold
            above_threshold = valid_P_y2 > threshold
            crossing_indices = np.where(np.diff(above_threshold.astype(int)) != 0)[0]
            
            # Step 5: Identify the burst region
            if len(crossing_indices) >= 2:
                first_crossing = crossing_indices[0]
                second_crossing = crossing_indices[-1]  # Use last crossing to capture full burst
                # Ensure indices are within bounds
                first_crossing = max(0, first_crossing)
                second_crossing = min(len(valid_P_y2) - 1, second_crossing)
                
                # Extract the signal in the burst region
                valid_range = valid_P_y2[first_crossing:second_crossing + 1]
                valid_time = data_X[first_crossing:second_crossing + 1]
                
                # Step 6: Count peaks using scipy.signal.find_peaks
                # Normalize signal to improve peak detection
                valid_range = valid_range / np.max(np.abs(valid_range)) if np.max(np.abs(valid_range)) > 0 else valid_range
                # Find peaks with prominence and height thresholds
                peak_count = count_zero_crossing_peaks_numpy(valid_range, 0.05)
                
                # Calculate the duration of the burst in milliseconds
                width_points = (len(valid_time) / Adq_Freq) * 1000
            else:
                peak_count = 0
                width_points = 999
            
            return peak_count, width_points


        num_bursts, passage = count_peaks_with_gaussian(y_Filtrada,data_X, freqfft1[np.argmax(ampFFT1)])


        #condition to clean data ####################### PILAS
        if passage<999:

            nueva_fila = pd.DataFrame({'name': [i], 'val_Freq': freqfft1[np.argmax(ampFFT1)]/1000, 'amp_Freq': max(ampFFT1) ,
                                    'amp_Time': max(abs(y_Filtrada)), 'num_burst': num_bursts, 'passage': passage}) 
            source_Table.stream(new_data=nueva_fila,rollover=max(1,len(columnas)))


    #############################################################################
    # Histogram Selection Widget
    #############################################################################

    TOOLS="pan,wheel_zoom,box_select,lasso_select,reset, tap"
    Histo_Select = figure(tools=TOOLS, width=700, height=700, min_border=10, min_border_left=50,
        toolbar_location="above", x_axis_location=None, y_axis_location=None,
        title="Linked Histograms")
    Histo_Select.select(BoxSelectTool).continuous = False
    Histo_Select.select(LassoSelectTool).continuous = False


    Histo_X_Axis = figure(title='X_Axis', x_axis_label='X_Axis', y_axis_label='Events',x_range=Histo_Select.x_range, min_border=10, min_border_left=50, y_axis_location="right",width=770, height=250)
    #Histo_X_Axis.yaxis.major_label_orientation = np.pi/4
    Histo_Y_Axis = figure(title='Y_Axis', x_axis_label='Events', y_axis_label='Y_Axis',y_range=Histo_Select.y_range, min_border=10, y_axis_location="right",width=250)
    Histo_Y_Axis.xaxis.major_label_orientation = np.pi/4


    #Histo_X_Axis.yaxis.major_label_orientation = np.pi / 4
    Histo_X_Axis.xaxis.major_label_text_font_size = "20pt"
    Histo_X_Axis.yaxis.major_label_text_font_size = "20pt"
    Histo_X_Axis.xaxis.major_label_text_font = "times"  # Set font family for x-axis
    Histo_X_Axis.yaxis.major_label_text_font = "times"

    
    #Histo_Y_Axis.xaxis.major_label_orientation = np.pi / 4
    Histo_Y_Axis.xaxis.major_label_text_font_size = "20pt"
    Histo_Y_Axis.yaxis.major_label_text_font_size = "20pt"
    Histo_Y_Axis.xaxis.major_label_text_font = "times"  # Set font family for x-axis
    Histo_Y_Axis.yaxis.major_label_text_font = "times"  # Set font family for y-axis


    button_comparision = Button(label="Update Parameters")
    OPTIONS = ["val_Freq", "amp_Freq", "amp_Time","num_burst", "passage"]
    multi_choice = MultiChoice(value=["val_Freq","amp_Freq"], options=OPTIONS,width=500,height=40 )
   
    #############################################################################
    # Interactive Functions
    #############################################################################


    # Helper Function to Update Data Sources

    """
    def update_data_sources(data_Y):
        
        Adq_Freq = int(parameters['acquisition_frequency'])
        order_filt = int(parameters['bandpass_filter_order'])
        Low_Freq = int(parameters['band_low_filter_frequency'])
        High_Freq = int(parameters['band_high_filter_frequency'])
        # Remove Mean and Apply Filter
        data_Y_NoOffset = data_Y - data_Y.mean()
        y_Filtrada = butter_bandpass_filter(data_Y_NoOffset, Low_Freq, High_Freq, order_filt, Adq_Freq)
        y_filtrada = y_Filtrada - y_Filtrada.mean()
        
        # Update Sources for Time and Frequency Domains
        filterImpact1.data_source.data = dict(data_x_axis=data_X, daya_y_axis=data_Y_NoOffset)
        filterImpact2.data_source.data = dict(data_x_axis=data_X, daya_y_axis=y_filtrada)

        # Update FFT
        ampFFT1, freqfft1, _ = FFT_calc(y_Filtrada, Adq_Freq)
        FilterFFT1.data_source.data = dict(data_x_axis=freqfft1, daya_y_axis=ampFFT1)

        # Gaussian fitting time signal
        valid_P_y2 = np.abs(hilbert(y_filtrada))
        try:
            optimized_params, _ = curve_fit(gaussian, data_X, valid_P_y2)
            # Obtener los parámetros optimizados
            amplitude, mean, stddev = optimized_params
            amplitude = amplitude/ np.max((amplitude))
            amplitude=amplitude*max(y_filtrada)
            y_curve = gaussian(data_X, amplitude, mean, stddev)
            filterImpact3.data_source.data = dict(data_x_axis=data_X, daya_y_axis=y_curve)
        except:
            y_curve=np.zeros(len(data_X))
            filterImpact3.data_source.data = dict(data_x_axis=data_X, daya_y_axis=y_curve)
    """



    def update_data_sources(data_Y):
        Adq_Freq = int(parameters['acquisition_frequency'])
        order_filt = int(parameters['bandpass_filter_order'])
        Low_Freq = int(parameters['band_low_filter_frequency'])
        High_Freq = int(parameters['band_high_filter_frequency'])

        data_Y_NoOffset = data_Y - data_Y.mean()
        y_Filtrada = butter_bandpass_filter(data_Y_NoOffset, Low_Freq, High_Freq, order_filt, Adq_Freq)
        y_filtrada = y_Filtrada - y_Filtrada.mean()
        
        filterImpact1.data_source.data = dict(data_x_axis=data_X, daya_y_axis=data_Y_NoOffset)
        filterImpact2.data_source.data = dict(data_x_axis=data_X, daya_y_axis=y_filtrada)

        ampFFT1, freqfft1, _ = FFT_calc(y_Filtrada, Adq_Freq)
        FilterFFT1.data_source.data = dict(data_x_axis=freqfft1, daya_y_axis=ampFFT1)

        #Gaussian filter
        if freqfft1[np.argmax(ampFFT1)]-7000 < 0:
            y_Filtrada = butter_bandpass_filter(y_filtrada, 0, freqfft1[np.argmax(ampFFT1)]+7000, order_filt, Adq_Freq)
        elif freqfft1[np.argmax(ampFFT1)]+7000 > Adq_Freq/2:
            y_Filtrada = butter_bandpass_filter(y_filtrada, freqfft1[np.argmax(ampFFT1)]-7000, (Adq_Freq/2)-2000, order_filt, Adq_Freq)  
        else:
            y_Filtrada = butter_bandpass_filter(y_filtrada, freqfft1[np.argmax(ampFFT1)]-7000, freqfft1[np.argmax(ampFFT1)]+7000, order_filt, Adq_Freq)


        valid_P_y2 = np.abs(hilbert(y_filtrada))

        try:
            initial_guess = [np.max(valid_P_y2), data_X[np.argmax(valid_P_y2)], 10]
            optimized_params, _ = curve_fit(gaussian, data_X, valid_P_y2, p0=initial_guess, maxfev=10000)
            amplitude, mean, stddev = optimized_params

            y_curve = gaussian(data_X, amplitude, mean, stddev)
            filterImpact3.data_source.data = dict(data_x_axis=data_X, daya_y_axis=y_curve)

            threshold = 0.30 * amplitude
            #print(f"Gaussian amplitude = {amplitude}, Threshold = {threshold}")
            peak_idx = np.argmax(y_curve)

            left_idx = peak_idx
            while left_idx > 0 and y_curve[left_idx] >= threshold:
                left_idx -= 1

            right_idx = peak_idx
            while right_idx < len(y_curve) - 1 and y_curve[right_idx] >= threshold:
                right_idx += 1

            new_start = data_X[left_idx]
            new_end = data_X[right_idx]

            print(f"Gaussian range: {new_start:.2f} ms → {new_end:.2f} ms")

            Section_Particle.x_range.start = new_start
            Section_Particle.x_range.end = new_end
            range_tool.x_range.start = new_start
            range_tool.x_range.end = new_end

        except Exception as e:
            print("Gaussian fit or range tool update failed:", e)
            y_curve = np.zeros(len(data_X))
            filterImpact3.data_source.data = dict(data_x_axis=data_X, daya_y_axis=y_curve)




    # Function to Update Section Data Based on Range Selection
    def update_section_data():
        data_Y=Original_Dataframe[scroll_menu.value]
        start, end = Section_Particle.x_range.start, Section_Particle.x_range.end
        visible_indices = [i for i, x in enumerate(data_X) if start <= x <= end]
        visible_x = data_X[visible_indices]
        visible_y = data_Y[visible_indices]
        
        # Update Section Particle Filtered data
        y_Filtrada = butter_bandpass_filter(visible_y - np.mean(visible_y), Low_Freq, High_Freq, order_filt, Adq_Freq)
        
        # Update FFT for Section FFT Figures
        amp_section_FFT, freq_section_FFT, _ = FFT_calc(visible_y, Adq_Freq)
        Section_FFT1.data_source.data = dict(data_x_axis=freq_section_FFT, daya_y_axis=amp_section_FFT)
        
        amp_filtered_FFT, freq_filtered_FFT, _ = FFT_calc(y_Filtrada, Adq_Freq)
        Section_FilterFFT1.data_source.data = dict(data_x_axis=freq_filtered_FFT, daya_y_axis=amp_filtered_FFT)

        # Optional Gaussian Curve Fitting
        x_data = np.linspace(0, len(amp_filtered_FFT), len(amp_filtered_FFT))
        y_data = amp_filtered_FFT - np.mean(amp_filtered_FFT)
        initial_params = [np.max(y_data) * 100, np.argmax(y_data), 1.0]
        try:
            optimized_params, _ = curve_fit(gaussian, x_data, y_data, p0=initial_params)
            amplitude, mean, stddev = optimized_params
            y_curve = gaussian(x_data, amplitude, mean, stddev)
            Section_FilterFFT2.data_source.data = dict(data_x_axis=freq_filtered_FFT, daya_y_axis=y_curve)
        except:
            y_curve=np.zeros(len(freq_filtered_FFT))
            Section_FilterFFT2.data_source.data = dict(data_x_axis=freq_filtered_FFT, daya_y_axis=y_curve)




    # Function to Move to the Next Column
    def next_column():
        nonlocal Counter_Column
        Counter_Column = (Counter_Column + 1) % len(columnas)  # Cycle back to start if at the end
        selected_column = columnas[Counter_Column]
        scroll_menu.value = selected_column  # Update dropdown selection
        update_data_sources(Original_Dataframe[scroll_menu.value])  # Refresh graph data

    # Function to Move to the Previous Column
    def prev_column():
        nonlocal Counter_Column
        Counter_Column = (Counter_Column - 1) % len(columnas)  # Cycle back to start if at the end
        selected_column = columnas[Counter_Column]
        scroll_menu.value = selected_column  # Update dropdown selection
        update_data_sources(Original_Dataframe[scroll_menu.value])  # Refresh graph data


    # Load your files in the initial setup
    def load_files(folder_path):
        return [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    # Example folder where files are stored

    file_list = load_files(path_folder)
    # Create a delete button and add functionality
    button_delete = Button(label="Delete File", button_type="danger")

    # Clear data sources if no files are left
    def clear_data_sources():
        source_original_data.data = {'data_x_axis': [], 'daya_y_axis': []}
        source_filtered_data.data = {'data_x_axis': [], 'daya_y_axis': []}
        source_gaussian_data.data = {'data_x_axis': [], 'daya_y_axis': []}
        source_original_FFT.data = {'data_x_axis': [], 'daya_y_axis': []}
        source_filtered_FFT.data = {'data_x_axis': [], 'daya_y_axis': []}
        source_original_section_FFT.data = {'data_x_axis': [], 'daya_y_axis': []}
        source_filtered_section_FFT1.data = {'data_x_axis': [], 'daya_y_axis': []}
        source_filtered_section_FFT2.data = {'data_x_axis': [], 'daya_y_axis': []}

    # Function to delete the selected file in the real folder
    """
    def delete_current_file():
        selected_file = scroll_menu.value  # Get the currently selected file from the scroll menu
        file_path = os.path.join(path_folder, selected_file)
        try:
            os.remove(file_path)  # Delete the file
            file_list.remove(selected_file)  # Remove from file list
            scroll_menu.options = file_list  # Update scroll menu
            if file_list:  # If files remain, select the next available file
                scroll_menu.value = file_list[0]
                update_data_sources(Original_Dataframe[scroll_menu.value])
            else:  # No files left, clear the data sources
                clear_data_sources()
        except Exception as e:
            print(f"Error deleting file {selected_file}: {e}")

    button_delete.on_click(delete_current_file)
    """
    #############################################################################
    #Function to just delete the file on the table 
    def delete_current_file():
        global Original_Dataframe  # Ensure access to global Original_Dataframe
        selected_indices = source_Table.selected.indices
        if not selected_indices:
            print("No row selected in the table")
            return
        
        df_data = pd.DataFrame(source_Table.data)
        selected_name = df_data['name'][selected_indices[0]]
        
        # Remove the row from the table data
        new_data = df_data.drop(selected_indices[0])
        
        # Reset index without adding duplicate columns
        if 'index' in new_data.columns or 'level_0' in new_data.columns:
            new_data = new_data.drop(columns=['index', 'level_0'], errors='ignore')
        new_data = new_data.reset_index(drop=True)
        
        # Update the ColumnDataSource with the new data
        source_Table.data = {col: new_data[col].values for col in new_data.columns}
        
        # Update the scroll menu and columnas
        if selected_name in columnas:
            columnas.remove(selected_name)
            scroll_menu.options = columnas
            
            # Update the Original_Dataframe by dropping the column
            if selected_name in Original_Dataframe.columns:
                Original_Dataframe = Original_Dataframe.drop(columns=[selected_name])
            
            # Select the next available file or clear if none remain
            if columnas:
                nonlocal Counter_Column
                Counter_Column = min(Counter_Column, len(columnas) - 1)
                scroll_menu.value = columnas[Counter_Column]
                update_data_sources(Original_Dataframe[scroll_menu.value])
            else:
                clear_data_sources()
                scroll_menu.value = ''
                print("No files left to display")

    button_delete.on_click(delete_current_file)




    # Update grpahs based on table selection
    def load_sample_table():
        selected_row_indices = source_Table.selected.indices
        df_data = pd.DataFrame(source_Table.data)
        sel_col=df_data['name'][selected_row_indices]
        sel_col=sel_col.values[0]
        scroll_menu.value = sel_col
        update_data_sources(Original_Dataframe[sel_col])





    #
    # load_comparision
    def load_comparision():

        Histo_Select.renderers = []
        Histo_X_Axis.renderers = []
        Histo_Y_Axis.renderers = []
        Histogram.renderers =[]

        x_label=multi_choice.value[0]
        y_label=multi_choice.value[1]


        #Individual Histogram and CV
        n= int(N_bins.value)
        df_data = pd.DataFrame(source_Table.data)
        data_select=df_data[multi_choice.value[0]].astype(float)
        min_val = data_select.min()
        max_val = data_select.max()
        std_dev = np.std(data_select)
        mean = np.mean(data_select)
        coefficient_of_variation = (std_dev / mean) * 100
        Histogram.title.text = ' CV: '+str(round(coefficient_of_variation,2))+ '% ' 
        bins = np.linspace(min_val, max_val, n+1)
        hist, edges = np.histogram(data_select, bins=bins)
        Histogram.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color='white')
            

        #Correlation Histogram
        df_data = pd.DataFrame(source_Table.data)
        x=df_data[x_label].astype(float)
        y=df_data[y_label].astype(float)

        names = df_data["name"]
        # Create a ColumnDataSource that includes the x, y, and name columns
        source = ColumnDataSource(data=dict(x=x, y=y, name=names))
        #points =Histo_Select.scatter(x, y, size=3, color="#3A5785", alpha=0.6)
        points = Histo_Select.scatter('x', 'y', source=source, size=10, color="#3A5785", alpha=0.8)
        hover = HoverTool(renderers=[points],
                  tooltips=[("File", "@name")])  # "@name" corresponds to the name column in the source
        Histo_Select.add_tools(hover)


        def on_point_click(event):
            # Get the selected index
            selected_index = source.selected.indices
            if selected_index:
                # Get the name of the selected point based on its index
                selected_name = source.data['name'][selected_index[0]]
                # Call the Python function with the selected name
                scroll_menu.value = selected_name
                update_data_sources(Original_Dataframe[selected_name])

        # Connect the tap event with the callback
        source.selected.on_change('indices', lambda attr, old, new: on_point_click(None))





        # create the horizontal histogram
        hhist, hedges = np.histogram(x, bins=n)
        hzeros = np.zeros(len(hedges)-1)
        LINE_ARGS = dict(color="#3A5785", line_color=None)
        Histo_X_Axis.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hhist, color="white", line_color="#3A5785")
        hh1 = Histo_X_Axis.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hzeros, alpha=0.5, **LINE_ARGS)
        hh2 = Histo_X_Axis.quad(bottom=0, left=hedges[:-1], right=hedges[1:], top=hzeros, alpha=0.1, **LINE_ARGS)
        Histo_X_Axis.title.text = str(x_label)


        # create the vertical histogram
        vhist, vedges = np.histogram(y, bins=10)
        vzeros = np.zeros(len(vedges)-1)
        Histo_Y_Axis.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vhist, color="white", line_color="#3A5785")
        vh1 = Histo_Y_Axis.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vzeros, alpha=0.5, **LINE_ARGS)
        vh2 = Histo_Y_Axis.quad(left=0, bottom=vedges[:-1], top=vedges[1:], right=vzeros, alpha=0.1, **LINE_ARGS)
        Histo_Y_Axis.title.text = str(y_label)



        def update_new_points(attr, old, new):
            inds = new
            if len(inds) == 0 or len(inds) == len(x):
                hhist1, hhist2 = hzeros, hzeros
                vhist1, vhist2 = vzeros, vzeros
            else:
                neg_inds = np.ones_like(x, dtype=np.bool_)
                neg_inds[inds] = False
                hhist1, _ = np.histogram(x[inds], bins=hedges)
                vhist1, _ = np.histogram(y[inds], bins=vedges)
                hhist2, _ = np.histogram(x[neg_inds], bins=hedges)
                vhist2, _ = np.histogram(y[neg_inds], bins=vedges)

            hh1.data_source.data["top"]   =  hhist1
            hh2.data_source.data["top"]   = -hhist2
            vh1.data_source.data["right"] =  vhist1
            vh2.data_source.data["right"] = -vhist2

        points.data_source.selected.on_change('indices', update_new_points)




    def save_to_csv():
        # Save the current DataFrame to a CSV file
        table_data=source_Table.data
        table_to_save=pd.DataFrame(table_data)
        table_to_save.to_csv('Particle_Table_Data.csv', index=False)
        print("CSV file saved as 'Particle_Table_Data.csv'.")

    button_save_table.on_click(save_to_csv)




    ###################################################################################
    # Button Actions
    ###################################################################################

    button_refresh.on_click(lambda: update_data_sources(Original_Dataframe[scroll_menu.value]))
    button_section.on_click(update_section_data)
    button_next.on_click(next_column)
    button_prev.on_click(prev_column)
    button_load_sample.on_click(load_sample_table)
    button_comparision.on_click(load_comparision)






    ###################################################################################
    # Layout Setup
    ###################################################################################

    lay2 = layout([
        [scroll_menu, button_refresh,button_prev, button_next,button_section,button_delete, button_save_table],
        [FilterFFT,Section_FilteredFFT], [filterImpact],
        [Section_Particle, Section_Particle_Filtered],
        [button_load_sample],
        [tabla],
        [button_comparision,multi_choice, N_bins],
        [Histogram],
        [Histo_Select, Histo_Y_Axis],
        [Histo_X_Axis]
    ])

    return lay2