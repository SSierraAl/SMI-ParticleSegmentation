
"""

Bokeh server for interactive visualization

"""


###########################################################################################################################
################################################# Libraries ###############################################################
###########################################################################################################################

import sys
from PySide6.QtWebEngineWidgets import QWebEngineView
from threading import Thread
import numpy as np
from Signal_Analysis.TAB_Signal_Analysis import *

#Server
from PySide6.QtWebEngineWidgets import QWebEngineView
from bokeh.server.server import Server
from tornado.ioloop import IOLoop
from threading import Thread

class Server_Init_Bokeh():
    'Class to manage the conection with the GUI and the bokeh server'
    def __init__(self, main_window,shared_queue):

        self.main_window = main_window
        self.shared_queue=shared_queue
        #QWeb widget
        self.web_view = QWebEngineView()
        self.main_window.ui.load_pages.Server_Layout.addWidget(self.web_view)
        #URL where the server is located
        self.web_view.setUrl("http://localhost:5006/bkapp")
        #Thread focused in the server
        self.bokeh_thread = Thread(target=self.start_bokeh_server)
        self.bokeh_thread.start()

        #GUI BTN
        #////////////////////////////////////
        self.main_window.ui.load_pages.Update_but_Server.clicked.connect(self.update_plot_Server)


    # Update queue data 
    def update_plot_Server(self):
        #Clear the figure
        #Graph.renderers = []
        #Update the data
        new_y = np.random.random(len(x)) 
        self.shared_queue.put({'y': new_y, 'x':x },block=True)


    #BOKEH SERVER FUNCTIONS
    #////////////////////////////////////

    #Function to intialize the server
    def start_bokeh_server(self):
        self.ioloop = IOLoop()
        server = Server({'/bkapp': self.bokeh_server_handler}, io_loop=self.ioloop, port=5006)
        server.start()
        self.ioloop.start()
    
    #Handler, and periodic update to modify the column data source
    def bokeh_server_handler(self, doc):
        'Handler associated to the bokeh aplication and document'
        doc.add_root(lay)
        doc.title = "Bokeh Application"
        
        # Periodic callback to update the plot info
        # Share the column data source via the queue
        def update():
            if not self.shared_queue.empty():
                new_data = self.shared_queue.get()
                
                #Graph_line.data_source.data['x_axis'] = new_data['x']
                #Graph_line.data_source.data['y_axis'] = new_data['y']
                #source.stream(new_data)
        #Update every 200ms (to be adjusted)
        doc.add_periodic_callback(update, 200)
  

    #Stop the loop and the thread
    def stop_thread(self):
        self.ioloop.add_callback(self.ioloop.stop)  # Stop the IOLoop
        self.bokeh_thread.join()  # Wait for the Bokeh server thread to finish


