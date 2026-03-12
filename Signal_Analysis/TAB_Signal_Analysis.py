"""

Bokeh server for interactive visualization

"""

# bokeh serve --show .\TAB_Signal_Analysis.py

###########################################################################################################################
################################################# Libraries ###############################################################
###########################################################################################################################
from bokeh.plotting import curdoc
from bokeh.models import Button, Div, TextInput, Spacer
from bokeh.layouts import layout, column, row
import tkinter as tk
from tkinter import filedialog
#Server
#from Signal_Analysis.TAB_1_Signal_Analysis import Set_Reader_Data  # Import Set_Reader_Data function
#Local
from TAB_1_Signal_Analysis import Set_Reader_Data  # Import Set_Reader_Data function

# Parameters
Counter_Column = 0

###############################################################################################
# Folder Selection Functions ##################################################################
###############################################################################################

# Description
desc = Div(text=open("..\description.html").read())

# Folder selection elements
folder_label = Div(text="Selected Folder:", width=600)
folder_path_work = Div(text="../Temporal_Files/Particle_Analysis", width=600)
select_button = Button(label="Select Folder", button_type="primary")

# Variable to hold the layout
Tab_Lay_reader = None

def open_folder_dialog():
    global Tab_Lay_reader
    # Create a new instance of tkinter for folder selection
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder_path = filedialog.askdirectory(master=root)
    root.destroy()

    # Update the path and reload data if a folder is selected
    if folder_path:
        folder_path_work.text = f"{folder_path}"
        # Re-create reader layout with the new path
        Tab_Lay_reader.children[0] = Set_Reader_Data(folder_path_work.text, exp_parameters)

# Trigger folder selection dialog on button click
select_button.on_click(lambda: curdoc().add_next_tick_callback(open_folder_dialog))

###############################################################################################
# Experiment Parameters #######################################################################
###############################################################################################

# Function to create experiment parameters input layout
def create_experiment_parameters():
    # Titles and Inputs
    titles = [
        ("Laser wavelength [nm]:", "1550"),
        ("Acquisition frequency [Hz]:", "2000000"),
        ("Angle [deg]:", "80"),
        ("Band pass filter order:", "4"),
        ("Low Frequency [Hz]:", "7000"),
        ("High Frequency [Hz]:", "80000")
    ]

    # Create Div and TextInput for each parameter
    param_divs = [Div(text=title, width=200) for title, _ in titles]
    param_inputs = [TextInput(value=default) for _, default in titles]

    # Link inputs with parameter dictionary
    param_keys = [
        "laser_wavelength", "acquisition_frequency", "laser_angle",
        "bandpass_filter_order", "band_low_filter_frequency", "band_high_filter_frequency"
    ]
    exp_parameters = {key: input.value for key, input in zip(param_keys, param_inputs)}

    # Function to update the dictionary on button click
    def submit_parameters():
        for key, input in zip(param_keys, param_inputs):
            exp_parameters[key] = input.value
        submitted_values_div.text = (
            f"<b>Updated Parameters:</b><br>"
            f"Laser Wavelength: {exp_parameters['laser_wavelength']} nm, "
            f"Acquisition Frequency: {exp_parameters['acquisition_frequency']} Hz, "
            f"Laser Angle: {exp_parameters['laser_angle']} degrees, "
            f"Bandpass Filter Order: {exp_parameters['bandpass_filter_order']}, "
            f"Low Frequency: {exp_parameters['band_low_filter_frequency']} Hz, "
            f"High Frequency: {exp_parameters['band_high_filter_frequency']} Hz"
        )

    
    # Button to submit parameters
    submit_button = Button(label="Update Parameters", button_type="success")
    submit_button.on_click(submit_parameters)

    # Display for submitted values
    submitted_values_div = Div(text="", width=1200)
    submit_parameters()
    # Layout organization for experiment parameters
    param_layout = layout([
        [param_divs[0],param_inputs[0]],
        [param_divs[1],param_inputs[1]],
        [param_divs[2],param_inputs[2]],
        [param_divs[3],param_inputs[3]],
        [param_divs[4],param_inputs[4]],
        [param_divs[5],param_inputs[5]],
        [submit_button],
        [submitted_values_div]]
    )
    return param_layout, exp_parameters

param_layout, exp_parameters = create_experiment_parameters()

###############################################################################################
# Reader Layout ###############################################################################
###############################################################################################

# Load Data and Display info
Tab_Lay_reader = column(Set_Reader_Data(folder_path_work.text, exp_parameters))



###############################################################################################
# Main Layout #################################################################################
###############################################################################################

# Organize the layout using sections
lay = layout([
    [desc],
    param_layout,
    Spacer(height=10),
    [select_button],
    [folder_label],
    [folder_path_work],
    Spacer(height=20),
    Tab_Lay_reader
])

#local
curdoc().add_root(lay)
