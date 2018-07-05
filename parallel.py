import plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline as offline
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


class ParallelCoordinates(object):
    def __init__(self):
        self.traces = []
        self.title = 'pc plot'
        self.filename = 'plot'

    def add_trace(self, df):
        self.traces.append(go.Parcoords(
            line = dict(
                color=df['target'],
                colorscale='Jet',
            ),
            dimensions = list(
            [dict(label = col, values = df[col].values) for col in df.columns]
        )))

    def set_title(self, s):
        self.title = s

    def plot(self):
        layout = go.Layout(title=self.title)
        fig = go.Figure(data=self.traces, layout=layout)
        file_html = self.filename + 'plot.html'
        offline.plot(fig, filename=file_html, image='png', image_filename=self.filename)
        # offline.iplot(fig)


iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

p = ParallelCoordinates()
p.add_trace(df)

p.set_title('iris')
p.plot()
