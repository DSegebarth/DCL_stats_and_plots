#!/usr/bin/env python
# coding: utf-8

# Import all dependencies

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pingouin as pg
import seaborn as sns
import os
import itertools
import statistics as stats
import ipywidgets as widgets
from ipywidgets import HBox, VBox, Layout

from IPython.display import display

###################################################################
# Overview:

    # 1 Compute statistics
    # 2 Plotting
    # 3 Annotate stats within the plots
    # 4 Create widget elements
    # 5 Functions that are triggered by clicking the widget buttons
    # 6 Specify widget layout and launch it

###################################################################


###################################################################
# 1 Functions to compute the different statistics
# 1.1 Comparison of independent samples

def independent_samples():        
    global data_col, group_col, d_main, l_groups
    data_col = df.columns[0]
    group_col = df.columns[1]

    d_main = {}
    l_groups = list(df[group_col].unique())
    for group_id in l_groups:
        d_main[group_id] = {'data': df.loc[df[group_col] == group_id, data_col].values,
                            'normality_full': pg.normality(df.loc[df[group_col] == group_id, data_col].values),
                            'normality_bool': pg.normality(df.loc[df[group_col] == group_id, data_col].values)['normal'][0]}

    n_groups = len(l_groups)

    d_main['summary'] = {'normality': all([d_main[elem]['normality_bool'] for elem in l_groups]),
                         'homoscedasticity': pg.homoscedasticity([d_main[elem]['data'] for elem in l_groups])['equal_var'][0]}

    parametric = all([d_main['summary']['normality'], d_main['summary']['homoscedasticity']])

    if len(l_groups) > 2:
        if parametric:
            d_main['summary']['ANOVA'] = pg.anova(data=df, dv=data_col, between=group_col)
        else:
            d_main['summary']['Kruskal_Wallis_ANOVA'] = pg.kruskal(data=df, dv=data_col, between=group_col)

    if len(l_groups) > 1:
        d_main['summary']['pairwise_comparisons'] = pg.pairwise_ttests(data=df, dv=data_col, between=group_col, parametric=parametric, padjust='holm')

    else:
        print('Error: The group_id column has to contain at least two different group_ids for this selection. Did you mean to perform a one-sample test?')


# 1.2 Mixed-model ANOVA (contributed by Konstantin Kobel):

def mixed_model_ANOVA():
    global d_main, data_col, group_col, subject_col, session_col, l_groups, l_sessions
    data_col = df.columns[0]
    group_col = df.columns[1]
    subject_col = df.columns[2]
    session_col = df.columns[3]
    
    d_main = {}
    l_groups = list(df[group_col].unique())
    l_sessions = list(df[session_col].unique())

    # Hier hat der zusätzliche group Filter für den Test auf Normalverteilung gefehlt
    #
    for groups in l_groups:
        for sessions in l_sessions:       
            d_main[groups, sessions] = {'data': df.loc[(df[group_col] == groups) & (df[session_col] == sessions), data_col].values,
                                        'mean': df.loc[(df[group_col] == groups) & (df[session_col] == sessions), data_col].mean(),
                                        'normality_full': pg.normality(df.loc[(df[group_col] == groups) & (df[session_col] == sessions), data_col].values),
                                        'normality_bool': pg.normality(df.loc[(df[group_col] == groups) & (df[session_col] == sessions), data_col].values)['normal'][0]}

    n_groups = len(l_groups)*len(l_sessions)
    d_main['summary'] = {}
        
    d_main['summary'] = {'normality': all([d_main[keys]['normality_bool'] for keys in d_main.keys() if keys != 'summary']),
                     'homoscedasticity': pg.homoscedasticity([d_main[keys]['data'] for keys in d_main.keys() if keys != 'summary'])['equal_var'][0]}    
   
    parametric = all([d_main['summary']['normality'], d_main['summary']['homoscedasticity']])

    d_main['summary']['ANOVA'] = pg.mixed_anova(data=df, dv=data_col, within=session_col, subject=subject_col, between=group_col)
    
    if parametric == False:
        print ("Please be aware that the data require non-parametric testing.\nHowever, this is not implemented yet and a parametric test is computed instead.")
    else:
        nothing_special_here = 'nothing_special_here'
        
    
        
    d_main['summary']['pairwise_comparisons'] = pg.pairwise_ttests(data=df, dv=data_col, 
                                                                   within=session_col, subject=subject_col, 
                                                                   between=group_col, padjust='holm')

###################################################################    

    
###################################################################
# 2 Functions to create the different plots:
# 2.1 Plotting for the comparison of independent samples:

# These functions are currently linked directly to the on_plotting_button_clicked function
# However, actually the choice of plots to make should be updated depending on what test you chose



# 2.2 Plotting for Mixed-Model-ANOVA:
# 2.2.1 Point-plot (contributed by Konstantin Kobel):



###################################################################    

    
###################################################################
# 3 Functions to annotate the results of the statistical tests in the respective plots:
# 3.1 Comparison of independent samples:

def annotate_stats_independent_samples(l_stats_to_annotate):
    if len(l_stats_to_annotate) > 0:
        max_total = df[data_col].max()
        y, h, col = max_total + max_total * 0.05, max_total * 0.05, 'k'

        # Add check whether group level ANOVA / Kruska-Wallis-ANOVA is significant
        df_temp = d_main['summary']['pairwise_comparisons'].copy()

        for group1, group2 in l_stats_to_annotate:

            x1 = l_group_order.index(group1)
            x2 = l_group_order.index(group2)

            if df_temp.loc[(df_temp['A'] == group1) & (df_temp['B'] == group2)].shape[0] > 0:
                if 'p-corr' in df_temp.loc[(df_temp['A'] == group1) & (df_temp['B'] == group2)].columns:
                    pval = df_temp.loc[(df_temp['A'] == group1) & (df_temp['B'] == group2), 'p-corr'].iloc[0]
                else:
                    pval = df_temp.loc[(df_temp['A'] == group1) & (df_temp['B'] == group2), 'p-unc'].iloc[0]

            elif df_temp.loc[(df_temp['B'] == group1) & (df_temp['A'] == group2)].shape[0] > 0:
                if 'p-corr' in df_temp.loc[(df_temp['B'] == group1) & (df_temp['A'] == group2)].columns:
                    pval = df_temp.loc[(df_temp['B'] == group1) & (df_temp['A'] == group2), 'p-corr'].iloc[0]
                else:
                    pval = df_temp.loc[(df_temp['B'] == group1) & (df_temp['A'] == group2), 'p-unc'].iloc[0]
            else:
                print('There was an error with annotating the stats!')

            if pval < 0.001:
                stars = '***'
            elif pval < 0.01:
                stars = '**'
            elif pval < 0.05:
                stars = '*'
            else: 
                stars = 'n.s.'

            plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)    
            plt.text((x1+x2)*.5, y+h, stars, ha='center', va='bottom', color=col)

            y = y + 3*h

# 3.2 Annotate stats for Mixed-model ANOVA (contributed by Konstantin Kobel):
def annotate_stats_mma_pointplot():
    col = 'k'
    
    #Fehlt: Funktion stars
    #Legendenbar
    #annotate mit l_checkboxes
    l_to_annotate=[]
    g=list(itertools.combinations(l_groups, 2))

    for session in l_sessions:
        for combi in g:
            l_to_annotate.append((combi[0], combi[1], session))
    
    #Liste sortieren nach größte Mittelwert-Differenz ganz rechts innerhalb einer Session
    if len(l_groups) > 2:
        last_session=99999999
        for elem in l_to_annotate:
            n=l_to_annotate.index(elem)
            while n > 0 and elem[2] == l_to_annotate[n-1][2] and abs(df.loc[(df[group_id] == elem[0]) & (df[session_id] == elem[2]), data].mean()-df.loc[(df[group_id] == elem[1]) & (df[session_id] == elem[2]), data].mean())> last_session:
                l_to_annotate[n-1], l_to_annotate[n]=l_to_annotate[n], l_to_annotate[n-1]
            last_session=abs(df.loc[(df[group_id] == elem[0]) & (df[session_id] == elem[2]), data].mean()-df.loc[(df[group_id] == elem[1]) & (df[session_id] == elem[2]), data].mean())

    c=0
    prev_session="randomstartsession"
    for elem in l_to_annotate:
        y1=df.loc[(df[group_id] == elem[0]) & (df[session_id] == elem[2]), data].mean()
        y2=df.loc[(df[group_id] == elem[1]) & (df[session_id] == elem[2]), data].mean()
        x=l_sessions.index(elem[2])+0.06
        b=0.05
        plt.text(x+0.1+c, (y1+y2)/2, "***", rotation=90)
        plt.plot([x+c, x+b+c, x+b+c, x+c], [y1, y1, y2, y2], color=col, lw=1.5)
        if elem[2] == prev_session:
            c=c+0.08
        else:
            c=0
        prev_session = elem[2]

###################################################################    

    
###################################################################
# 4 Functions that are triggered by clicking on the widget buttons:
# 4.1 Stats button:        

def on_stats_button_clicked(b):
    uploader = stats_widget.children[0].children[0]
    select_test = stats_widget.children[1].children[0]
    annotate_stats_box = stats_widget.children[3].children[0].children[0].children[0]
    plotting_button = stats_widget.children[2].children[1]
    select_plot = stats_widget.children[2].children[0]
    main_accordion = stats_widget.children[3].children[0]
    select_downloads = stats_widget.children[4].children[0]
    download_button = stats_widget.children[2].children[1]
    
    global df, save_plot, l_checkboxes
    # Open the uploaded file:
    if list(uploader.value.keys())[0].endswith('.csv'): 
        with open("input.csv", "w+b") as i:
            i.write(uploader.value[list(uploader.value.keys())[0]]['content'])
        df = pd.read_csv('input.csv', index_col=0)
        
    elif list(uploader.value.keys())[0].endswith('.xlsx'):
        with open("input.xlsx", "w+b") as i:
            i.write(uploader.value[list(uploader.value.keys())[0]]['content'])
        df = pd.read_excel('input.xlsx', index_col=0)


    save_plot = False
    
    with output:
        output.clear_output()
        
        uploader.layout.visibility = 'hidden'
        plotting_button.layout.visibility = 'visible'
        select_plot.layout.visibility = 'visible'
        main_accordion.layout.visibility = 'visible'
        select_downloads.layout.visibility = 'visible'
        download_button.layout.visibility = 'visible'
        
        # Not at all neccessary if we can really manage to align the plot
        if select_test.value == 0:
            select_plot.options = [('stripplot', 0), ('boxplot', 1), ('boxplot with scatterplot overlay', 2), ('violinplot', 3)]
        elif select_test.value == 2:
            select_plot.options = [('pointplot', 0), ('boxplot', 1), ('boxplot with scatterplot overlay', 2), ('violinplot', 3)]
        else:
            print('Function not implemented. Please go and annoy Dennis to finally do it')
        
        # Check what option was chosen in the select_test dropdown and execute corresponding function
        if select_test.value==0:
            independent_samples()
        elif select_test.value==2:
            mixed_model_ANOVA()
        else:
            nothing_special_here = 'nothing_special_here'
        
        
        # Don't forget to update the annotations accordion // Maybe this could also be placed in the corresponding stats function?
        l_checkboxes = [widgets.Checkbox(value=False,description='{} vs. {}'.format(group1, group2)) for group1, group2 in list(itertools.combinations(l_groups, 2))]

        l_HBoxes = []
        elem = 0
        for i in range(int(len(l_checkboxes)/3)):
            l_HBoxes.append(HBox(l_checkboxes[elem:elem+3]))
            elem = elem + 3
    
        if len(l_checkboxes) % 3 != 0:
            l_HBoxes.append(HBox(l_checkboxes[elem:]))

        checkboxes_to_add = VBox(l_HBoxes).children[:]

        if len(annotate_stats_box.children) == 0:
                annotate_stats_box.children = annotate_stats_box.children + checkboxes_to_add
         
        display(d_main['summary']['pairwise_comparisons'])

# 4.2 Plotting button
def on_plotting_button_clicked(b):
    select_test = stats_widget.children[1].children[0]
    plotting_button = stats_widget.children[2].children[1]
    select_plot = stats_widget.children[2].children[0]
    color_palettes = stats_widget.children[3].children[0].children[0].children[2].children[0]
    marker_size = stats_widget.children[3].children[0].children[0].children[2].children[1]
    yaxis_label_text = stats_widget.children[3].children[0].children[0].children[1].children[0].children[0]
    yaxis_label_fontsize = stats_widget.children[3].children[0].children[0].children[1].children[0].children[1]
    yaxis_label_color = stats_widget.children[3].children[0].children[0].children[1].children[0].children[2]
    xaxis_label_text = stats_widget.children[3].children[0].children[0].children[1].children[1].children[0]
    xaxis_label_fontsize = stats_widget.children[3].children[0].children[0].children[1].children[1].children[1]
    xaxis_label_color = stats_widget.children[3].children[0].children[0].children[1].children[1].children[2]
    with output:
        output.clear_output()
        
        plotting_button.description = 'Refresh the plot'
        
        # Could also be modyfied
        global l_group_order
        l_group_order = l_groups
        
        plt.figure(figsize=(14,8), facecolor='white')
        
        if select_test.value == 0: # independent_samples()
            if select_plot.value == 0:
                sns.stripplot(data=df, x=group_col, y=data_col, order=l_group_order, palette=color_palettes.value, size=marker_size.value)
            elif select_plot.value == 1:
                sns.boxplot(data=df, x=group_col, y=data_col, order=l_group_order, palette=color_palettes.value, size=marker_size.value)
            elif select_plot.value == 2:
                sns.boxplot(data=df, x=group_col, y=data_col, order=l_group_order, palette=color_palettes.value)
                sns.stripplot(data=df, x=group_col, y=data_col, color='k', order=l_group_order, size=marker_size.value)
            else:
                print("Function not implemented. Please go and annoy Dennis to finally do it")
        
        elif select_test.value == 2: # mixed_model_ANOVA()#
            if select_plot.value == 0:
                plot = sns.pointplot(data=df, x=session_col, y=data_col, hue=group_col, dodge=True, err_style='bars', ci='sd', palette=color_palettes.value)
                plot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            elif select_plot.value == 3:
                plot = sns.violinplot(data=df, x=session_col, y=data_col, hue=group_col, palette=color_palettes.value)
                plot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            else:
                print("Function not implemented. Please go and annoy Dennis to finally do it")
        
        else:
            print("Function not implemented. Please go and annoy Dennis to finally do it")
        
        
        # Place inside a function and use if conditions to create l_checkboxes according to the chosen test
        global l_checkboxes
        l_stats_to_annotate = []
        for i in range(len(l_checkboxes)):
            if l_checkboxes[i].value:
                checkbox_description = l_checkboxes[i].description
                group1 = checkbox_description[:checkbox_description.index(' ')]
                group2 = checkbox_description[checkbox_description.index(' vs. ') + 5 :]
                l_stats_to_annotate.append((group1, group2))
        annotate_stats_independent_samples(l_stats_to_annotate)
        
        plt.ylabel(yaxis_label_text.value, fontsize=yaxis_label_fontsize.value, color=yaxis_label_color.value)
        plt.xlabel(xaxis_label_text.value, fontsize=xaxis_label_fontsize.value, color=xaxis_label_color.value)
        
        if save_plot == True:
            plt.savefig('customized_plot.png', dpi=300)
        plt.show()
        
# 4.3 Download button:        
def on_download_button_clicked(b):
    select_downloads = stats_widget.children[4].children[0]
    plotting_button = stats_widget.children[2].children[1]
    global save_plot
    if select_downloads.value == 1:
        d_main['summary']['pairwise_comparisons'].to_csv('pairwise_comparisons.csv')
    elif select_downloads.value == 2:
        save_plot = True
        plotting_button.click()
        save_plot = False
    elif select_downloads.value == 3:
        d_main['summary']['pairwise_comparisons'].to_csv('pairwise_comparisons.csv') 
        save_plot = True
        plotting_button.click()
        save_plot = False
               
###################################################################    

    
###################################################################
# 5 Functions that create the individual widget elements:
# 5.1 Buttons:
def create_buttons():
    # File uploader:
    uploader = widgets.FileUpload(
        accept=('.xlsx,.csv'),  # Accepted file extension e.g. '.txt', '.pdf', 'image/*', 'image/*,.pdf'
        multiple=False)  # True to accept multiple files upload else False

    # Buttons:
    stats_button = widgets.Button(description="Calculate stats", icon='rocket')

    plotting_button = widgets.Button(description='Plot the data', layout={'visibility': 'hidden'})

    download_button = widgets.Button(description='Download', icon='file-download', layout={'visibility': 'hidden'})
    
    return uploader, stats_button, plotting_button, download_button

# 5.2 Dropdown menus:
def create_dropdowns():
    select_test = widgets.Dropdown(options=[('pairwise comparison of two or more independent samples', 0), ('compare one sample vs. fixed value', 1), ('Mixed_model_ANOVA', 2)],
                                   value=1,
                                   description='Please select which test you want to perform:',
                                   layout={'width': '700px'},
                                   style={'description_width': 'initial'})

    select_plot = widgets.Dropdown(options=[('something initial', 0)],
                                   value=0,
                                   description='Please select which type of plot you want to create:',
                                   layout={'width': '700px', 
                                          'visibility': 'hidden'},
                                   style={'description_width': 'initial'})

    select_downloads = widgets.Dropdown(options=[('statistical results only', 1), ('plot only', 2), ('both', 3)],
                                   value=1,
                                   description='Please select what you would like to write to disk:',
                                   layout={'width': '700px', 
                                          'visibility': 'hidden'},
                                   style={'description_width': 'initial'})

    return select_test, select_plot, select_downloads


def create_hbox_y_axis():
    yaxis_label_text = widgets.Text(value='data', placeholder='data', description='y-axis label:', layout={'width': '300px'})
    yaxis_label_fontsize = widgets.IntSlider(value=12, min=8, max=40, step=1, description='y-axis label fontsize:')
    yaxis_label_color = widgets.ColorPicker(concise=False, description='y-axisl label color', value='#000000')
    return HBox([yaxis_label_text, yaxis_label_fontsize, yaxis_label_color])

def create_hbox_x_axis():
    xaxis_label_text = widgets.Text(value='group_IDs', placeholder='group_IDs', description='x-axis label:', layout={'width': '300px'})
    xaxis_label_fontsize = widgets.IntSlider(value=12, min=8, max=40, step=1, description='x-axis label fontsize:')
    xaxis_label_color = widgets.ColorPicker(concise=False, description='x-axisl label color', value='#000000')
    return HBox([xaxis_label_text, xaxis_label_fontsize, xaxis_label_color])

def create_hbox_plot_style_features():
    color_palettes = widgets.Dropdown(options=['colorblind', 'Spectral', 'viridis', 'rocket', 'cubehelix'],
                             value='colorblind',
                             description='Please select a color palette', 
                             layout={'width': '400px'},
                             style={'description_width': 'initial'})
    
    marker_size = widgets.FloatText(value=5,description='marker size:')
    
    return HBox([color_palettes, marker_size])

def create_accordion_to_customize_the_plot():
    # Accordion to customize the plotting:
    # We will create another Accordion inside the main accordion
    # First accordion will contain checkboxes to select which stats shall be annotated
    # Will be filled as soon as stats_button is clicked and tests are run
    annotate_stats_box = VBox([])

    # Second accordion will contain widgets to customize the axes
    yaxis_hbox = create_hbox_y_axis()
    xaxis_hbox = create_hbox_x_axis()

    customize_axes_box = VBox([yaxis_hbox, xaxis_hbox])
    
    # Third accordion will contain widgets to customize the style of the plot (colorpalette, markersizes)
    customize_features_box = create_hbox_plot_style_features()


    # Create the accordion that actually contains all widget-containing accordions and will become the only child of the main accordion
    accordion = widgets.Accordion(children=[annotate_stats_box, customize_axes_box, customize_features_box], selected_index=None)

    # Give the individual accordions titles that are displayed before dropdown is clicked
    accordion.set_title(0, 'Annotate stats')
    accordion.set_title(1, 'Customize axes')
    accordion.set_title(2, 'Customize features')

    # Create the main accordion that contains all widgets to customize the plot and use selected_index=None to avoid dropdown by default
    main_accordion = widgets.Accordion(children=[accordion], selected_index=None, continous_update=False, layout={'visibility': 'hidden'})
    main_accordion.set_title(0, 'Expand me to customize your plot!')
    
    return main_accordion 
    
    
def top_level_layout():
    main_accordion = create_accordion_to_customize_the_plot()
    
    select_test, select_plot, select_downloads = create_dropdowns()
    
    uploader, stats_button, plotting_button, download_button = create_buttons()
    
    # Bind the on_button_clicked functions to the respective buttons:
    stats_button.on_click(on_stats_button_clicked)
    plotting_button.on_click(on_plotting_button_clicked) 
    download_button.on_click(on_download_button_clicked)
    # Layout of the remaining elements
    first_row = HBox([uploader])
    second_row = HBox([select_test, stats_button])
    third_row = HBox([select_plot, plotting_button])
    third_row_extension = HBox([main_accordion])
    fourth_row = HBox([select_downloads, download_button])

    stats_widget = VBox([first_row, second_row, third_row, third_row_extension, fourth_row])
    
    return stats_widget
    
    
def launch():
    # Is there a more convenient way than making everything a global variable???
    #global uploader, stats_button, plotting_button, download_button, select_test, select_plot, select_downloads, color_palettes
    #global yaxis_label_text, xaxis_label_text, marker_size, yaxis_label_color, xaxis_label_color, yaxis_label_fontsize, xaxis_label_fontsize
    #global annotate_stats_box, yaxis_hbox, xaxis_hbox, customize_axes_box, customize_features_box, accordion, main_accordion
    #global first_row, second_row, third_row, third_row_extension, fourth_row, box, output    
    
    global stats_widget
    # Configure the layout:
    stats_widget = top_level_layout()

    global output
    # Define the output
    output = widgets.Output()

    # Display the widget:
    display(stats_widget, output)