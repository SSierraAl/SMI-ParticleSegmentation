#bokeh serve --show .\DataViewer_Simple.py


from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.layouts import layout


# Generate some data
x = [1,2,3,4,5,6,7,8,9,10]
y = [1,2,3,2,1,2,3,2,1,2]

source_col_data = ColumnDataSource(data=dict(x_axis=x, y_axis=y))

# Create a figure
Graph = figure(title="Sine Wave", x_axis_label='x', y_axis_label='y')
Graph_line=Graph.line('x_axis', 'y_axis', source=source_col_data)

# Define the layout
lay = layout([Graph])


# Add the layout to the current document
#curdoc().add_root(lay)
#curdoc().title = "Bokeh Application"
