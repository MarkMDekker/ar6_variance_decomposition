# ======================================== #
# Some functions for the plotting
# ======================================== #

from re import T
import plotly.graph_objects as go
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from plotly.colors import n_colors
import plotly.express as px
import xarray as xr
import plotly.graph_objects as go
from pathlib import Path
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

def rescale_size(s, max):
    s /= max
    s = s ** 0.3
    s *= 25
    return s

def tableplot(XR, t, srt):
        tot = np.array(XR.sel(Time=2050).S_c)+np.array(XR.sel(Time=2050).S_z)+np.array(XR.sel(Time=2050).S_m)
        ar = np.array([np.array(XR.Variable),
        np.array(XR.sel(Time=t).S_m/tot).round(2),
        np.array(XR.sel(Time=t).S_c/tot).round(2),
        np.array(XR.sel(Time=t).S_z/tot).round(2)])

        colors = n_colors('rgb(255, 200, 200)', 'rgb(200, 0, 0)', 100, colortype='rgb')
        bar = px.colors.diverging.RdBu_r
        colors = []
        for i in range(len(bar)-1):
                colors = colors+n_colors(px.colors.diverging.RdBu_r[i], px.colors.diverging.RdBu_r[i+1], int(100/len(bar)), colortype='rgb')
        colors = colors+n_colors(px.colors.diverging.RdBu_r[i+1], px.colors.diverging.RdBu_r[i+1], int(100/len(bar)), colortype='rgb')
        ar100 = (ar[1:]*100).astype(int)
        fig = go.Figure(data=[go.Table(
        columnwidth = [70,40, 40, 40],
        header = dict(
                values = [['<b>Variable</b><br>in '+str(t)],
                        ['Model'],
                        ['Climate category'],
                        ['Scenario']],

                line_color='darkslategray',
                fill_color='black',
                align=['center','center','center','center'],
                font=dict(color='white', size=12),
                height=40
        ),
        cells=dict(
                values=ar[:, ar[srt].argsort()[::-1]],
                line_color='black',
                align=['left','center','center','center'],
                fill_color=['white',
                        np.array(colors)[ar100[:, ar[srt].argsort()[::-1]][0]],
                        np.array(colors)[ar100[:, ar[srt].argsort()[::-1]][1]],
                        np.array(colors)[ar100[:, ar[srt].argsort()[::-1]][2]]],
                font_size=12,
                font={'family' :["Arial", "Arial Black", "Arial Black", "Arial Black"],
                'color': 'black'},
                height=30)
                )
        ])
        fig.update_layout(height=2000, width=1400)
        return fig

def draw_line(fig, name, x, y1, y2, color, xt, yt):
    fig.add_shape(type="line", xref='paper', yref='paper',
                    x0=x,
                    y0=y1,
                    x1=x,
                    y1=y2,
                    line=dict(color=color, width=3))
    fig.add_shape(type="line", xref='paper', yref='paper',
                    x0=x-0.01,
                    y0=y1,
                    x1=x,
                    y1=y1,
                    line=dict(color=color, width=3))
    fig.add_shape(type="line", xref='paper', yref='paper',
                    x0=x-0.01,
                    y0=y2,
                    x1=x,
                    y1=y2,
                    line=dict(color=color, width=3))
    fig.add_annotation(xref='paper',
                    yref='paper',
                    x=xt,
                    align="center",
                    y=yt,
                    text=name, 
                    textangle=90,
                    font=dict(
                        color=color,
                        size=12
                    ),
                    showarrow=False)

# ======================================== #
# Function to create boxplot traces
# ======================================== #

def boxplots(Var, colory, Year, XRdata, XRmeta, modelorcat, title, remc8):

    # Remove low level entries
    XR_filt = XRdata.sel(Variable=Var)
    XR_filt = XR_filt.dropna('ModelScenario', how='any')
    ModelScenarios = np.array(XR_filt.ModelScenario)
    Models = np.array([XR_filt.ModelScenario.data[i].split('|')[0] for i in range(len(XR_filt.ModelScenario))])
    Ccat = np.array(XRmeta.sel(ModelScenario=XR_filt.ModelScenario).Category.data)
    modscen_new = []
    if remc8 == "yes":
        for i in range(len(Models)):
                m = Models[i]
                c = Ccat[i]
                if len(np.where(Models == m)[0]) >= 10 and c != 'C8':
                        modscen_new.append(ModelScenarios[i])
    elif remc8 == "no":
        for i in range(len(Models)):
                m = Models[i]
                c = Ccat[i]
                if len(np.where(Models == m)[0]) >= 10:
                        modscen_new.append(ModelScenarios[i])
    XR_filt = XR_filt.sel(ModelScenario = modscen_new)

    # Calculate percentages for box plots
    ModelScenarios = np.array(XR_filt.ModelScenario)
    Models = np.array([XR_filt.ModelScenario.data[i].split('|')[0] for i in range(len(XR_filt.ModelScenario))])
    Ccat = np.array(XRmeta.sel(ModelScenario=XR_filt.ModelScenario).Category.data)
    unimodels = np.unique(Models)
    unicats = np.unique(Ccat)
    Perc_models = np.zeros(shape=(len(unimodels)))
    Perc_cats = np.zeros(shape=(len(unicats)))
    for m in range(len(unimodels)):
        ms = ModelScenarios[Models == unimodels[m]]
        #Perc_models[m] = np.percentile(np.array(XR_filt.sel(Time=Year, ModelScenario=ms).Value), 50)
        Perc_models[m] = np.median(np.array(XR_filt.sel(Time=Year, ModelScenario=ms).Value))
    for c in range(len(unicats)):
        ms = ModelScenarios[Ccat == unicats[c]]
        #Perc_cats[c] = np.percentile(np.array(XR_filt.sel(Time=Year, ModelScenario=ms).Value), 50)
        Perc_cats[c] = np.median(np.array(XR_filt.sel(Time=Year, ModelScenario=ms).Value))

    # Colors
    rgb = mpl.colors.colorConverter.to_rgb(colory)
    colly='rgb('+str(rgb[0]*255)+','+str(rgb[1]*255)+','+str(rgb[2]*255)+')'
    modelcolors = n_colors(colly, 'rgb(0,0,0)', len(unimodels)+3, colortype='rgb')
    catcolors = n_colors(colly, 'rgb(0,0,0)', len(unicats)+3, colortype='rgb')
#     for m in range(len(unimodels)):
#         n_colors('rgb(255, 200, 200)', 'rgb(200, 0, 0)', 100, colortype='rgb')
#     mpl.colors.Normalize(vmin=0, vmax=len())
#     cols = ['forestgreen', 'tomato', 'steelblue', 'goldenrod', 'purple', 'grey', 'brown',
#             'magenta', 'red', 'darkgrey', 'blue', 'black', 'darkgreen']
            
    medians = [np.median(np.array(XR_filt.sel(Time=Year).Value)[Models == i]) for i in unimodels]
    ranks_m = [np.argsort(np.argsort(medians))[np.where(unimodels == i)[0][0]] for i in Models]
    ranks_c = [int(i[1]) for i in Ccat]

    r1, p1 = spearmanr(ranks_m, np.array(XR_filt.sel(Time=Year).Value))
    r2, p2 = spearmanr(ranks_c, np.array(XR_filt.sel(Time=Year).Value))
    r1 = r1*np.sign(r1)
    r2 = r2*np.sign(r2)
    traces_m = []
    traces_c = []
    traces_mc = []

    for m in range(len(unimodels)):
        m2 = np.argsort(Perc_models)[m]
        #rgb = mpl.colors.colorConverter.to_rgb(plt.cm.get_cmap('YlOrBr')(0.2+0.8*m/len(unimodels)))
        #col = 'rgb('+str(rgb[0]*255)+','+str(rgb[1]*255)+','+str(rgb[2]*255)+')'
        col = modelcolors[m]
        ms = ModelScenarios[Models == unimodels[m2]]
        y = np.array(XR_filt.sel(Time=Year, ModelScenario=ms).Value)
        #tickvals=unimodels[np.argsort(Perc_models)], ticktext=['<br>-'.join(i.split(' ')[0].split('-')) for i in unimodels[np.argsort(Perc_models)]]
        traces_m.append(go.Box(y=y, name=unimodels[m2][:3], fillcolor=col, line_color='black', marker=dict(color=col), boxpoints='all', jitter=0.5, pointpos=-1.8, whiskerwidth=0.2, marker_size=1.5, line_width=1, showlegend=False))#'<br>-'.join(unimodels[m2].split(' ')[0].split('-'))
    for m in range(len(unicats)):
        #rgb = mpl.colors.colorConverter.to_rgb(plt.cm.get_cmap('RdBu_r')(0.+m/len(unicats)))
        col = catcolors[m]#'rgb('+str(rgb[0])+','+str(rgb[1])+','+str(rgb[2])+')'
        ms = ModelScenarios[Ccat == unicats[m]]
        y = np.array(XR_filt.sel(Time=Year, ModelScenario=ms).Value)
        traces_c.append(go.Box(y=y, name=unicats[m], fillcolor=col, line_color='black', marker=dict(color=col), boxpoints='all', jitter=0.5, pointpos=-1.8, whiskerwidth=0.2, marker_size=1.5, line_width=1, showlegend=False))
    for m in range(len(unimodels)):
        m2 = np.argsort(Perc_models)[m]
        rgb = mpl.colors.colorConverter.to_rgb(plt.cm.get_cmap('YlOrBr')(0.+m/len(unimodels)))
        col = 'rgb('+str(rgb[0])+','+str(rgb[1])+','+str(rgb[2])+')'
        for c in range(len(unicats)):
            ms = ModelScenarios[(Ccat == unicats[c]) & (Models == unimodels[m2])]
            y = np.array(XR_filt.sel(Time=Year, ModelScenario=ms).Value)
            traces_mc.append(go.Box(y=y, name=unicats[c]+' '+unimodels[m2], fillcolor=col, line_color='black', marker=dict(color=col), boxpoints=False, jitter=0.5, pointpos=-1.8, whiskerwidth=0.2, marker_size=1.5, line_width=1, showlegend=False))
#     fig.update_yaxes(title=Var+' in '+str(Year), row=1, col=1)
#     fig.update_xaxes(tickangle=0, tickvals=unimodels[np.argsort(Perc_models)], ticktext=['<br>-'.join(i.split(' ')[0].split('-')) for i in unimodels[np.argsort(Perc_models)]], tickfont=dict(size=11), row=1, col=1)
#     fig.update_layout(
#         title=Var+' in '+str(Year),
#         yaxis=dict(
#             autorange=True,
#             showgrid=True,
#             zeroline=True,
#             gridcolor='rgb(255, 255, 255)',
#             gridwidth=1,
#             zerolinecolor='rgb(255, 255, 255)',
#             zerolinewidth=2,
#         ),
#         #paper_bgcolor='rgb(243, 243, 243)',
#         #plot_bgcolor='rgb(243, 243, 243)',
#         showlegend=False
#     )
#     fig.update_yaxes(ticktext=['']*len(np.arange(0, 270, 25)), row=1, col=2)
#     fig.update_yaxes(title=Var+' in '+str(Year), row=3, col=1)
#    fig.update_layout(height=1000, width = 1500)
    if modelorcat == 'model':
        args = (Var, traces_m, title, colory)
    elif modelorcat == 'cat':
        args = (Var, traces_c, title, colory)
    return args

# ======================================== #
# Function to create triangle plots
# ======================================== #

def triangleplot(XR, xrraw, CCAT, T1, T2, T3, T4):
    y1, t1, title1, col1 = T1
    y2, t2, title2, col2 = T2
    y3, t3, title3, col3 = T3
    y4, t4, title4, col4 = T4
    DF_counts = pd.read_csv("X:/user/dekkerm/Projects/AR6_Variance/variancedecomposition/Data/Counts.csv", index_col=0)
    xrraw = xrraw.sel(Time=range(2030,2101))
    XR = XR.sel(Time=range(2030,2101))
    varlist = XR.Variable.data

    if CCAT == 'Electricity Generation':
        names = []
        for v_i in range(len(varlist)):
                if varlist[v_i] == 'Secondary Energy|Electricity':
                        names.append('Total')
                else:
                        names.append(varlist[v_i][29:])
    elif CCAT == 'Sector Transportation':
        names = []
        for v_i in range(len(varlist)):
                if varlist[v_i] == 'Final Energy|Transportation':
                        names.append('Total')
                else:
                        names.append(varlist[v_i][28:])

    years = XR.Time.data
    varmax = np.max(XR['Var_total']).data

    years_str = np.copy(years).astype(str)
    years_str[(years_str != '2050') & (years_str != '2100')] = ''

    # Colors
    if CCAT == 'Electricity Generation':
        cols = ['white', 'mediumseagreen', 'yellowgreen', 'dimgray', 'saddlebrown', 'rosybrown', 'darkred',
                'tomato', 'goldenrod', 'violet', 'skyblue', 'deeppink']
        scale = 0.7
    elif CCAT == 'Sector Transportation':
        cols = ['white', 'grey', 'saddlebrown', 'teal', 'goldenrod',
                'tomato']
        scale = 2.0


    # Construct percentage-lines (purely layout)
    plines = []
    for i in range(10):
        a = [0+i*0.1, 0+i*0.1]
        b = [0, 1-a[0]]
        c = [1-a[0], 0]
        rgb = mpl.colors.colorConverter.to_rgb(plt.cm.get_cmap('Greys')(0.+i/15))
        col = 'rgb('+str(rgb[0])+','+str(rgb[1])+','+str(rgb[2])+')'
        plines.append(go.Scatterternary(a=a, b=b, c=c, showlegend=False, mode='lines', hoverinfo='skip', line={'width': 0.25+i*0.1, 'color': col}))
        plines.append(go.Scatterternary(a=b, b=a, c=c, showlegend=False, mode='lines', hoverinfo='skip', line={'width': 0.25+i*0.1, 'color': col}))
        plines.append(go.Scatterternary(a=b, b=c, c=a, showlegend=False, mode='lines', hoverinfo='skip', line={'width': 0.25+i*0.1, 'color': col}))

    startraces = []
    linetraces = []
    for v in range(len(varlist)):
        # Means
        ds = XR.sel(Variable=varlist[v])
        a, b, c, s = np.array(ds[['S_c', 'S_m', 'S_z', 'Var_total']].to_array())
        s = scale*rescale_size(s, varmax)
        if varlist[v] != 'Secondary Energy|Electricity|Wind+Solar':
                count = list(DF_counts[DF_counts.Variable==varlist[v]].Count)[0]
        else:
                count = np.min([list(DF_counts[DF_counts.Variable=="Secondary Energy|Electricity|Wind"].Count)[0],
                                list(DF_counts[DF_counts.Variable=="Secondary Energy|Electricity|Solar"].Count)[0]])
        trace = go.Scatterternary(a=a, b=b, c=c,
                                        mode='markers+text+lines',
                                        name=names[v]+' ('+str(count)+')',
                                        showlegend=True, text=years,
                                        hovertemplate='%{text}<br>Var (climate): %{a}<br>Var (model): %{b} <br>Var (other): %{c}',
                                        marker={'size': s,
                                                'color': cols[v],
                                                'opacity': 1,
                                                'line' :dict(width=0.5, color='black')},
                                        line={'width': 1.5},
                                        textfont=dict(size=1,
                                                      color=cols[v]))
        linetraces.append(trace)
        trace = go.Scatterternary(a=[a[-1]], b=[b[-1]], c=[c[-1]],
                                        mode='markers', name=names[v]+' ('+str(count)+')', showlegend=False, text=[years[-1]],
                                        hovertemplate='%{text}<br>Var (climate): %{a}<br>Var (model): %{b} <br>Var (other): %{c}',
                                        marker={'size': float(s[-1]), 'symbol': 'star', 'color': cols[v], 'opacity': 1, 'line' :dict(width=float(s[-1])/10, color='black')}, line={'width': 1.5},
                                        textfont=dict(size=1, color=cols[v]))
        startraces.append(trace)
    
    trace_ternary_border = []
    trace_ternary_border.append(go.Scatterternary(a=[0.004, 0.004], b=[0, 1], c=[1, 0], showlegend=False, mode='lines', hoverinfo='skip', line={'width': 2, 'color': 'black'}))
    trace_ternary_border.append(go.Scatterternary(a=[0, 1], b=[0.004, 0.004], c=[1, 0], showlegend=False, mode='lines', hoverinfo='skip', line={'width': 2, 'color': 'black'}))
    trace_ternary_border.append(go.Scatterternary(a=[0, 1], b=[1, 0], c=[0.004, 0.004], showlegend=False, mode='lines', hoverinfo='skip', line={'width': 2, 'color': 'black'}))
    
    XRdata = xr.open_dataset("X:/user/dekkerm/Projects/AR6_Variance/variancedecomposition/Data/XRdata.nc")
    XRmeta = xr.open_dataset("X:/user/dekkerm/Projects/AR6_Variance/variancedecomposition/Data/XRmeta.nc")
    
    fig = make_subplots(
        rows=6, cols=4,
        horizontal_spacing = 0.02,
        vertical_spacing=0,
        subplot_titles = (title1,
                          "",
                          "",
                          title2,
                          "",
                          "",
                          "",
                          "",
                          "",
                          "",
                          "",
                          "",
                          title3,
                          "",
                          "",
                          title4),
        # column_widths = [0.15, 0.7, 0.15],
        # row_heights = [0.15, 0.7, 0.15],
        specs = [[{"type": "box", "rowspan": 2}, {"type": "scatterternary", "colspan": 2, "rowspan": 6}, {}, {"type": "box", "rowspan": 2}],
                [{}, {}, {}, {}],
                [{}, {}, {}, {}],
                [{"type": "box", "rowspan": 2}, {}, {}, {"type": "box", "rowspan": 2}],
                [{}, {}, {}, {}],
                [{}, {}, {}, {}]
                ]
        )

    # boxplots
    for t_i in range(4):
        T = [t1, t2, t3, t4][t_i]
        for n in range(len(T)):
                fig.add_trace(T[n], [1, 1, 4, 4][t_i], [1, 4, 1, 4][t_i])

    # Triangle plot
    for n in range(len(plines)):
        fig.add_trace(plines[n], 1, 2)
    for v in range(len(varlist)):
        fig.add_trace(linetraces[v], 1, 2)
    for v in range(len(varlist)):
        fig.add_trace(startraces[v], 1, 2)
#     for v in range(len(trace_ternary_border)):
#         fig.add_trace(trace_ternary_border[v], 1, 2)

    drawline_brack(fig, 0.25, 0.66, 1.0, col1)
    drawline_brack(fig, 0.25, 0.16, 0.5, col3)
    drawline_brack(fig, 0.72, 0.66, 1.0, col2, backwards='y')
    drawline_brack(fig, 0.72, 0.16, 0.5, col4, backwards='y')

    # Other layout stuff
    fig.update_layout(height=800, width=1700, ternary={'sum':1,
                                                    'aaxis': {'title': 'Climate target'},
                                                    'baxis': {'title': 'Model'},
                                                    'caxis': {'title': 'Other'}},
                    font=dict(size=18))
    a=0
    for row in [1, 4]:
        for col in [1, 4]:
                fig.update_yaxes(title=[y1, y2, y3, y4][a], tickfont={'size':11}, row=row, col=col, titlefont={'size':11})
                fig.update_xaxes(tickangle=0, tickfont=dict(size=8), row=row, col=col)
                a+=1
    fig.update_layout({#'margin': dict(l=0,r=0,b=0,t=0),
        'plot_bgcolor':'rgb(243, 243, 243)',
        'ternary':{'sum':1,
                   'bgcolor':'whitesmoke',
                   'aaxis':{'title': 'Climate target<br>', 'min': 0, 
                'linewidth':0, 'ticks':'outside',
                'tickmode':'array','tickvals':[0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 'ticktext':['50%', '60%', '70%', '80%', '90%', '100%'], 'tickfont':{'size':12}},
        'baxis':{'title': 'Model &nbsp; &nbsp;', 'min': 0, 
                'linewidth':2, 'ticks':'outside',
                'tickmode':'array','tickvals':[0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 'ticktext':['50%', '60%', '70%', '80%', '90%', '100%'],'tickangle':60, 'tickfont':{'size':12}},
        'caxis':{'title': 'Other scenario<br>assumptions', 'min': 0, 
                'linewidth':2, 'ticks':'outside',
                'tickmode':'array','tickvals':[0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 'ticktext':['50%', '60%', '70%', '80%', '90%', '100%'],'tickangle':-60, 'tickfont':{'size':12}}}})

    # Legend
    fig.update_layout(legend=dict(yanchor="top", y=-0.12, xanchor="left", x=0.04,
                                  orientation='h',
                                  font=dict(#family="Courier",
                                            size=12,
                                            color="black"),
                                  bordercolor="Black",
                                  borderwidth=0))

    return fig