# ======================================== #
# Some functions for the plotting (called by Main.ipynb)
# ======================================== #

import plotly.graph_objects as go
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

def rescale_size(s, max):
    s /= max
    s = s ** 0.3
    s *= 25
    return s

def triangleplot(XR):
    DF_counts = pd.read_csv("X:/user/dekkerm/Projects/AR6_Variance/variancedecomposition/Data/Counts.csv", index_col=0)
    XR = XR.sel(Time=range(2030,2101))
    varlist = XR.Variable.data
    years = XR.Time.data
    varmax = np.max(XR['Var_total']).data
    cols = ['forestgreen', 'tomato', 'steelblue', 'goldenrod', 'purple', 'grey', 'brown',
            'magenta', 'red', 'darkgrey', 'blue', 'black', 'darkgreen']
    years_str = np.copy(years).astype(str)
    years_str[(years_str != '2050') & (years_str != '2100')] = ''
    fig = go.Figure()

    # Construct percentage-lines (purely layout)
    for i in range(10):
        a = [0+i*0.1, 0+i*0.1]
        b = [0, 1-a[0]]
        c = [1-a[0], 0]
        rgb = mpl.colors.colorConverter.to_rgb(plt.cm.get_cmap('Greys')(0.+i/15))
        col = 'rgb('+str(rgb[0])+','+str(rgb[1])+','+str(rgb[2])+')'
        fig.add_trace(go.Scatterternary(a=a, b=b, c=c, showlegend=False, mode='lines', hoverinfo='skip', line={'width': 0.25+i*0.1, 'color': col}))
        fig.add_trace(go.Scatterternary(a=b, b=a, c=c, showlegend=False, mode='lines', hoverinfo='skip', line={'width': 0.25+i*0.1, 'color': col}))
        fig.add_trace(go.Scatterternary(a=b, b=c, c=a, showlegend=False, mode='lines', hoverinfo='skip', line={'width': 0.25+i*0.1, 'color': col}))

    # Actual lines
    for v in range(len(varlist)):
        # Percentiles
        for member in np.array(XR.Member):
            ds = XR.isel(Variable=v, Member=member)#.quantile(0.99, dim='Member')
            a, b, c, s = np.array(ds[['S_c', 'S_m', 'S_z', 'Var_total']].to_array())
            s = rescale_size(s, varmax)
            fig.add_trace(go.Scatterternary(a=a, b=b, c=c,
                                            mode='lines',
                                            name=varlist[v]+' ('+str(list(DF_counts[DF_counts.Variable==varlist[v]].Count)[0])+')',
                                            showlegend=False, text=years,
                                            #hovertemplate='%{text}<br>Var (climate): %{a}<br>Var (model): %{b} <br>Var (other): %{c}',
                                            marker={'size': 0,
                                                    'color': cols[v],
                                                    'opacity': 1,
                                                    'line' :dict(width=0., color='black')},
                                            line={'width': 0.1},
                                            textfont=dict(size=1,
                                                        color=cols[v]),
                                            hoverinfo='skip'))
            ds = XR.isel(Variable=v, Member=member)#.quantile(0.01, dim='Member')
            a, b, c, s = np.array(ds[['S_c', 'S_m', 'S_z', 'Var_total']].to_array())
            s = rescale_size(s, varmax)
            fig.add_trace(go.Scatterternary(a=a, b=b, c=c,
                                            mode='lines',
                                            name=varlist[v]+' ('+str(list(DF_counts[DF_counts.Variable==varlist[v]].Count)[0])+')',
                                            showlegend=False, text=years,
                                            #hovertemplate='%{text}<br>Var (climate): %{a}<br>Var (model): %{b} <br>Var (other): %{c}',
                                            marker={'size': 0,
                                                    'color': cols[v],
                                                    'opacity': 1,
                                                    'line' :dict(width=0., color='black')},
                                            line={'width': 0.1},
                                            #fill='tonext',
                                            textfont=dict(size=1,
                                                        color=cols[v]),
                                            hoverinfo='skip'))

    for v in range(len(varlist)):
        # Means
        ds = XR.isel(Variable=v).mean(dim='Member')
        a, b, c, s = np.array(ds[['S_c', 'S_m', 'S_z', 'Var_total']].to_array())
        s = rescale_size(s, varmax)
        fig.add_trace(go.Scatterternary(a=a, b=b, c=c,
                                        mode='markers+text+lines',
                                        name=varlist[v]+' ('+str(list(DF_counts[DF_counts.Variable==varlist[v]].Count)[0])+')',
                                        showlegend=True, text=years,
                                        hovertemplate='%{text}<br>Var (climate): %{a}<br>Var (model): %{b} <br>Var (other): %{c}',
                                        marker={'size': s,
                                                'color': cols[v],
                                                'opacity': 1,
                                                'line' :dict(width=0.5, color='black')},
                                        line={'width': 1.5},
                                        textfont=dict(size=1,
                                                      color=cols[v])))
    
    # Stars for the year 2100
    for v in range(len(varlist)):
        ds = XR.isel(Variable=v, Time=-1).mean(dim='Member')
        a, b, c, s = np.array(ds[['S_c', 'S_m', 'S_z', 'Var_total']].to_array())
        s = rescale_size(s, varmax)
        fig.add_trace(go.Scatterternary(a=[a], b=[b], c=[c],
                                        mode='markers', name=varlist[v]+' ('+str(list(DF_counts[DF_counts.Variable==varlist[v]].Count)[0])+')', showlegend=False, text=[years[-1]],
                                        hovertemplate='%{text}<br>Var (climate): %{a}<br>Var (model): %{b} <br>Var (other): %{c}',
                                        marker={'size': s, 'symbol': 'star', 'color': cols[v], 'opacity': 1, 'line' :dict(width=s/10, color='black')}, line={'width': 1.5},
                                        textfont=dict(size=1, color=cols[v])))
    
    # Other layout stuff
    fig.update_layout(height=900, width=1400, ternary={'sum':1,
                                                    'aaxis': {'title': 'Climate target'},
                                                    'baxis': {'title': 'Model'},
                                                    'caxis': {'title': 'Other'}},
                    font=dict(size=18))
    fig.update_layout({
    'ternary':
        {'sum':1,
        'aaxis':{'title': 'Climate target<br>', 'min': 0, 
                'linewidth':0, 'ticks':'outside',
                'tickmode':'array','tickvals':[0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 'ticktext':['50%', '60%', '70%', '80%', '90%', '100%'], 'tickfont':{'size':12}},
        'baxis':{'title': 'Model &nbsp; &nbsp;', 'min': 0, 
                'linewidth':2, 'ticks':'outside',
                'tickmode':'array','tickvals':[0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 'ticktext':['50%', '60%', '70%', '80%', '90%', '100%'],'tickangle':60, 'tickfont':{'size':12}},
        'caxis':{'title': 'Other scenario<br>assumptions', 'min': 0, 
                'linewidth':2, 'ticks':'outside',
                'tickmode':'array','tickvals':[0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 'ticktext':['50%', '60%', '70%', '80%', '90%', '100%'],'tickangle':-60, 'tickfont':{'size':12}}}})
    return fig