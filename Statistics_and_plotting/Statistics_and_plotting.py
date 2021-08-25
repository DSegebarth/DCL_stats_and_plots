import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pingouin as pg
import seaborn as sns
import os
import itertools
import statistics as stats
import math
import ipywidgets as widgets
from ipywidgets import HBox, VBox, Layout

from IPython.display import display

###################################################################
#Overview:

    # 1 Compute statistics
    # 2 Annotate stats within the plots
    # 3 Functions that are triggered by clicking the widget buttons
    # 4 Create all widget elements
    # 5 Specify widget layout and launch it
    # 6 Process statistical results for download

###################################################################


###################################################################
# 1 Functions to compute the different statistics
# 1.1 Comparison of independent samples
def independent_samples():
    global data_col, group_col, d_main, l_groups, performed_test
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
            d_main['summary']['group_level_statistic'] = pg.anova(data=df, dv=data_col, between=group_col)
            performed_test = 'One-way ANOVA'
        else:
            d_main['summary']['group_level_statistic'] = pg.kruskal(data=df, dv=data_col, between=group_col)
            performed_test = 'Kruskal-Wallis-ANOVA'

    if len(l_groups) > 1:
        d_main['summary']['pairwise_comparisons'] = pg.pairwise_ttests(data=df, dv=data_col, between=group_col, parametric=parametric, padjust='holm')

    else:
        print('Error: The group_id column has to contain at least two different group_ids for this selection.\
        \nDid you mean to perform a one-sample test?')   
   
        
# 1.2 Mixed-model ANOVA (contributed by Konstantin Kobel):
def mixed_model_ANOVA():
    global d_main, data_col, group_col, subject_col, session_col, l_groups, l_sessions, performed_test
    data_col = df.columns[0]
    group_col = df.columns[1]
    subject_col = df.columns[2]
    session_col = df.columns[3]
    
    d_main = {}
    l_groups = list(df[group_col].unique())
    l_sessions = list(df[session_col].unique())

    for group_id in l_groups:
        for session_id in l_sessions:       
            d_main[group_id, session_id] = {'data': df.loc[(df[group_col] == group_id) & (df[session_col] == session_id), data_col].values,
                                            'mean': df.loc[(df[group_col] == group_id) & (df[session_col] == session_id), data_col].mean(),
                                            'normality_full': pg.normality(df.loc[(df[group_col] == group_id) 
                                                                                  & (df[session_col] == session_id), data_col].values),
                                            'normality_bool': pg.normality(df.loc[(df[group_col] == group_id) 
                                                                                  & (df[session_col] == session_id), data_col].values)['normal'][0]}

    n_groups = len(l_groups)*len(l_sessions)
    d_main['summary'] = {}
        
    d_main['summary'] = {'normality': all([d_main[key]['normality_bool'] for key in d_main.keys() if key != 'summary']),
                         'homoscedasticity': pg.homoscedasticity([d_main[key]['data'] for key in d_main.keys() if key != 'summary'])['equal_var'][0]}    
   
    parametric = all([d_main['summary']['normality'], d_main['summary']['homoscedasticity']])

    d_main['summary']['group_level_statistic'] = pg.mixed_anova(data=df, dv=data_col, within=session_col, subject=subject_col, between=group_col)
    performed_test = 'Mixed-model ANOVA'
    # If we found some non-parametric alternative this could be implemented here
    if parametric == False:
        print ("Please be aware that the data require non-parametric testing.\n\
        However, this is not implemented yet and a parametric test is computed instead.")
        
    d_main['summary']['pairwise_comparisons'] = pg.pairwise_ttests(data=df, dv=data_col, 
                                                                   within=session_col, subject=subject_col, 
                                                                   between=group_col, padjust='holm')

###################################################################    

    
###################################################################
# 2 Functions to annotate the results of the statistical tests in the respective plots:
# 2.1 Get and update all customization values that were set by the user:
def get_customization_values():
    global distance_stars_to_brackets, distance_brackets_to_data, fontsize_stars_bold
    global linewidth_annotations, fontsize_stars, annotation_brackets_factor
    global l_xlabel_order, l_hue_order
    
    distance_stars_to_brackets = set_distance_stars_to_brackets.value
    distance_brackets_to_data = set_distance_brackets_to_data.value
    fontsize_stars = set_fontsize_stars.value
    linewidth_annotations = set_linewidth_annotations.value
    
    if set_stars_fontweight_bold.value == True:
        fontsize_stars_bold = 'bold'
    else:
        fontsize_stars_bold = 'normal'
    
    if select_bracket_no_bracket.value == 'Brackets':
        annotation_brackets_factor = 1
    else:
        annotation_brackets_factor = 0

    l_xlabel_order = []
    l_xlabel_string = set_xlabel_order.value
    
    while ', ' in l_xlabel_string:
        l_xlabel_order.append(l_xlabel_string[:l_xlabel_string.index(', ')])
        l_xlabel_string = l_xlabel_string[l_xlabel_string.index(', ')+2:]

    l_xlabel_order.append(l_xlabel_string)
    
    l_hue_order = []
    l_hue_string = set_hue_order.value
    
    while ', ' in l_hue_string:
        l_hue_order.append(l_hue_string[:l_hue_string.index(', ')])
        l_hue_string = l_hue_string[l_hue_string.index(', ')+2:]

    l_hue_order.append(l_hue_string)
    
        
# 2.2 Get l_stats_to_annotate:
# 2.2.1 For independent samples:
def get_l_stats_to_annotate_independent_samples():
    l_stats_to_annotate = []
    if set_annotate_all.value==True:
        for i in range(len(l_checkboxes)):
            l_checkboxes[i].value = True
    for i in range(len(l_checkboxes)):
        if l_checkboxes[i].value:
            checkbox_description = l_checkboxes[i].description
            group1 = checkbox_description[:checkbox_description.index(' ')]
            group2 = checkbox_description[checkbox_description.index(' vs. ') + 5 :]
            l_stats_to_annotate.append((group1, group2))
    return l_stats_to_annotate


# 2.2.2 For Mixed-Model-ANOVA:
def get_l_stats_to_annotate_mma():
    l_stats_to_annotate = []
    if set_annotate_all.value==True:
        for i in range(len(l_checkboxes)):
            l_checkboxes[i][1].value = True
    for i in range(len(l_checkboxes)):
        if l_checkboxes[i][1].value:
            checkbox_description = l_checkboxes[i][1].description
            group1 = checkbox_description[:checkbox_description.index(' ')]
            group2 = checkbox_description[checkbox_description.index(' vs. ') + 5 :]
            session_id = l_checkboxes[i][0]
            l_stats_to_annotate.append((group1, group2, session_id))
    return l_stats_to_annotate


# 2.3 Get the 'stars' string for the respective pairwise comparison:
def get_stars_str(df_tmp, group1, group2):
    if df_tmp.loc[(df_tmp['A'] == group1) & (df_tmp['B'] == group2)].shape[0] > 0:
        if 'p-corr' in df_tmp.loc[(df_tmp['A'] == group1) & (df_tmp['B'] == group2)].columns:
            pval = df_tmp.loc[(df_tmp['A'] == group1) & (df_tmp['B'] == group2), 'p-corr'].iloc[0]
        else:
            pval = df_tmp.loc[(df_tmp['A'] == group1) & (df_tmp['B'] == group2), 'p-unc'].iloc[0]

    elif df_tmp.loc[(df_tmp['B'] == group1) & (df_tmp['A'] == group2)].shape[0] > 0:
        if 'p-corr' in df_tmp.loc[(df_tmp['B'] == group1) & (df_tmp['A'] == group2)].columns:
            pval = df_tmp.loc[(df_tmp['B'] == group1) & (df_tmp['A'] == group2), 'p-corr'].iloc[0]
        else:
            pval = df_tmp.loc[(df_tmp['B'] == group1) & (df_tmp['A'] == group2), 'p-unc'].iloc[0]
    else:
        print('There was an error with annotating the stats!')
    if pval <= 0.001:
        stars = '***'
    elif pval <= 0.01:
        stars = '**'
    elif pval <= 0.05:
        stars = '*'
    else: 
        stars = 'n.s.'
    return stars


# 2.4 Annotate the stats in the respective plots
# 2.4.1 Annotate stats in independent sample plots:
def annotate_stats_independent_samples(l_stats_to_annotate):
    if len(l_stats_to_annotate) > 0:
        max_total = df[data_col].max()
        y_shift_annotation_line = max_total * distance_brackets_to_data
        brackets_height = y_shift_annotation_line*0.5*annotation_brackets_factor
        y_shift_annotation_text = brackets_height + y_shift_annotation_line*0.5*distance_stars_to_brackets
        
        # Set initial y
        y = max_total + y_shift_annotation_line

        # Add check whether group level ANOVA / Kruska-Wallis-ANOVA is significant
        df_temp = d_main['summary']['pairwise_comparisons'].copy()

        for group1, group2 in l_stats_to_annotate:

            x1 = l_xlabel_order.index(group1)
            x2 = l_xlabel_order.index(group2)
            
            stars = get_stars_str(df_temp, group1, group2)

            plt.plot([x1, x1, x2, x2], [y, y+brackets_height, y+brackets_height, y], c='k', lw=linewidth_annotations)    
            plt.text((x1+x2)*.5, y+y_shift_annotation_text, stars, ha='center', va='bottom', color='k', 
                     fontsize=fontsize_stars, fontweight=fontsize_stars_bold)
            
            # With set_distance_stars_to_brackets being limited to 5, stars will always be closer than next annotation line
            y = y+3*y_shift_annotation_line

            
# 2.4.2 Annotate stats in Mixed-model ANOVA plots:
# 2.4.2.1 Annotate stats in Mixed-model ANOVA point plot:
def annotate_stats_mma_pointplot(l_stats_to_annotate):
    if len(l_stats_to_annotate) > 0:
        l_to_annotate_ordered = []
        for session_id in l_sessions:
            l_temp = [elem for elem in l_stats_to_annotate if elem[2]==session_id]
            for elem in l_temp:
                abs_mean_difference = abs(df.loc[(df[group_col]==elem[0]) & (df[session_col]==elem[2]), data_col].mean()-
                                          df.loc[(df[group_col]==elem[1]) & (df[session_col]==elem[2]), data_col].mean())
                l_temp[l_temp.index(elem)] = elem+(abs_mean_difference,)
            l_temp.sort(key=sort_by_third)
            l_to_annotate_ordered = l_to_annotate_ordered+l_temp

        df_temp = d_main['summary']['pairwise_comparisons'].copy()
        
        for elem in l_to_annotate_ordered:
            group1, group2, session_id, abs_mean_difference = elem
            
            if l_to_annotate_ordered.index(elem) == 0:
                n_previous_annotations_in_this_session_id = 0
            elif session_id == prev_session:
                n_previous_annotations_in_this_session_id = n_previous_annotations_in_this_session_id + 1
            else:
                n_previous_annotations_in_this_session_id = 0
            
            x_shift_annotation_line = distance_brackets_to_data + distance_brackets_to_data * n_previous_annotations_in_this_session_id * 1.5
            brackets_height = distance_brackets_to_data*0.5*annotation_brackets_factor
            x_shift_annotation_text = brackets_height + distance_brackets_to_data*0.5*distance_stars_to_brackets            
            
            x = l_xlabel_order.index(session_id) + x_shift_annotation_line
            y1=df.loc[(df[group_col] == group1) & (df[session_col] == session_id), data_col].mean()
            y2=df.loc[(df[group_col] == group2) & (df[session_col] == session_id), data_col].mean()            
            
            stars = get_stars_str(df_temp.loc[df_temp[session_col] == session_id], group1, group2)
            
            plt.plot([x, x+brackets_height, x+brackets_height, x], [y1, y1, y2, y2], color='k', lw=linewidth_annotations)
            plt.text(x+x_shift_annotation_text, (y1+y2)/2, stars, rotation=-90, ha='center', va='center', 
                     fontsize=fontsize_stars, fontweight=fontsize_stars_bold)             

            prev_session = session_id

            
# Helper function to make sorting based on 3rd element in tuple possible
def sort_by_third(e):
    return e[3]


# 2.4.2.2 Annotate stats in Mixed-model ANOVA violin plot:
def annotate_stats_mma_violinplot(l_stats_to_annotate):
    if len(l_stats_to_annotate) > 0:
        l_to_annotate_ordered = []
        for session_id in l_sessions:
            l_temp = [elem for elem in l_stats_to_annotate if elem[2]==session_id]
            for elem in l_temp:
                abs_mean_difference = abs(df.loc[(df[group_col]==elem[0]) & (df[session_col]==elem[2]), data_col].mean()-
                                          df.loc[(df[group_col]==elem[1]) & (df[session_col]==elem[2]), data_col].mean())
                l_temp[l_temp.index(elem)] = elem+(abs_mean_difference,)
            l_temp.sort(key=sort_by_third)
            l_to_annotate_ordered = l_to_annotate_ordered+l_temp

        df_temp = d_main['summary']['pairwise_comparisons'].copy()

        max_total = df[data_col].max()
        y_shift_annotation_line = max_total * distance_brackets_to_data
        brackets_height = y_shift_annotation_line*0.5*annotation_brackets_factor
        y_shift_annotation_text = brackets_height + y_shift_annotation_line*0.5*distance_stars_to_brackets
        
        for elem in l_to_annotate_ordered:
            group1, group2, session_id, abs_mean_difference = elem

            if l_to_annotate_ordered.index(elem) == 0:
                n_previous_annotations_in_this_session_id = 0
            elif session_id == prev_session:
                n_previous_annotations_in_this_session_id = n_previous_annotations_in_this_session_id + 1
            else:
                n_previous_annotations_in_this_session_id = 0

            y = max_total + y_shift_annotation_line + y_shift_annotation_line*n_previous_annotations_in_this_session_id*3
            
            width = 0.8
            x_base = l_xlabel_order.index(session_id) - width/2 + width/(2*len(l_hue_order))
            x1 = x_base + width/len(l_hue_order)*l_hue_order.index(group1)
            x2 = x_base + width/len(l_hue_order)*l_hue_order.index(group2)
            
            stars = get_stars_str(df_temp.loc[df_temp[session_col] == session_id], group1, group2)

            plt.plot([x1, x1, x2, x2], [y, y+brackets_height, y+brackets_height, y], color='k', lw=linewidth_annotations)
            plt.text((x1+x2)/2, y+y_shift_annotation_text, stars, ha='center', va='bottom', 
                     fontsize=fontsize_stars, fontweight=fontsize_stars_bold)
            
            prev_session = session_id

            
###################################################################    

    
###################################################################
# 3 Functions that are triggered by clicking on the widget buttons:
# 3.1 Stats button:        
def on_stats_button_clicked(b):
    global df, save_plot, l_checkboxes
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
        expand_me_accordion.layout.visibility = 'visible'
        select_downloads.layout.visibility = 'visible'
        download_button.layout.visibility = 'visible'
        
        if select_test.value == 0: # comparison of independent samples
            select_plot.options = [('stripplot', 0), ('boxplot', 1), ('boxplot with scatterplot overlay', 2), ('violinplot', 3)]
        elif select_test.value == 2: # mixed-model ANOVA
            select_plot.options = [('pointplot', 0), ('boxplot', 1), ('boxplot with scatterplot overlay', 2), ('violinplot', 3)]
        else:
            print('Function not implemented. Please go and annoy Dennis to finally do it')
        
        if select_test.value==0:
            independent_samples()
            checkboxes_to_add, l_checkboxes = create_checkboxes_pairwise_comparisons()
        elif select_test.value==2:
            mixed_model_ANOVA()
            checkboxes_to_add, l_checkboxes = create_checkboxes_pairwise_comparisons_mma()

        if len(select_annotations_vbox.children) == 0:
                select_annotations_vbox.children = select_annotations_vbox.children + checkboxes_to_add
        
        create_group_order_text()
        create_ylims()
        
        create_group_color_pickers()
        
        
        
        display(d_main['summary']['pairwise_comparisons'])   

        
# 3.2 Plotting button
def on_plotting_button_clicked(b):
    global l_xlabel_order 
    # Update all variables according to the customization input of the user
    get_customization_values()
    
    with output:
        output.clear_output()
        
        plotting_button.description = 'Refresh the plot'
                   
        if select_palette_or_individual_color.value == 0:
            color_palette = select_color_palettes.value
        else:
            color_palette = {}
            for group_id in l_groups:
                color_palette[group_id] = group_colors_vbox.children[l_groups.index(group_id)].value
             
        fig = plt.figure(figsize=(set_fig_width.value/2.54 , set_fig_height.value/2.54), facecolor='white')
        ax = fig.add_subplot()
        
        for axis in ['top', 'right']:
            ax.spines[axis].set_visible(False)
        
        for axis in ['bottom','left']:
            ax.spines[axis].set_linewidth(set_axes_linewidth.value)
            ax.spines[axis].set_color(set_axes_color.value)      

        plt.tick_params(labelsize=set_axes_tick_size.value, colors=set_axes_color.value)

        if select_test.value == 0: # independent_samples()
            if select_plot.value == 0:
                sns.stripplot(data=df, x=group_col, y=data_col, order=l_xlabel_order, palette=color_palette, size=set_marker_size.value)
            elif select_plot.value == 1:
                sns.boxplot(data=df, x=group_col, y=data_col, order=l_xlabel_order, palette=color_palette)
            elif select_plot.value == 2:
                sns.boxplot(data=df, x=group_col, y=data_col, order=l_xlabel_order, palette=color_palette, showfliers=False)
                sns.stripplot(data=df, x=group_col, y=data_col, color='k', order=l_xlabel_order, size=set_marker_size.value)
            elif select_plot.value == 3:
                sns.violinplot(data=df, x=group_col, y=data_col, order=l_xlabel_order, palette=color_palette, cut=0)
                sns.stripplot(data=df, x=group_col, y=data_col, color='k', order=l_xlabel_order, size=set_marker_size.value)                
            else:
                print("Function not implemented. Please go and annoy Dennis to finally do it")
                  
        elif select_test.value == 2: # mixed_model_ANOVA()
            if select_plot.value == 0:
                sns.pointplot(data=df, x=session_col, y=data_col, order=l_xlabel_order, hue=group_col, hue_order=l_hue_order,
                              palette=color_palette, dodge=True, ci='sd', err_style='bars', capsize=0)  
            elif select_plot.value == 1:
                sns.boxplot(data=df, x=session_col, y=data_col, order=l_xlabel_order, hue=group_col, hue_order=l_hue_order,
                            palette=color_palette)
            elif select_plot.value == 2:
                sns.boxplot(data=df, x=session_col, y=data_col, order=l_xlabel_order, hue=group_col, hue_order=l_hue_order,
                            palette=color_palette, showfliers=False)
                sns.stripplot(data=df, x=session_col, y=data_col, order=l_xlabel_order, hue=group_col, hue_order=l_hue_order,
                              dodge=True, color='k', size=set_marker_size.value)
            elif select_plot.value == 3:
                sns.violinplot(data=df, x=session_col, y=data_col, order=l_xlabel_order, hue=group_col, hue_order=l_hue_order,
                               width=0.8, cut=0, palette=color_palette)
                sns.stripplot(data=df, x=session_col, y=data_col, order=l_xlabel_order, hue=group_col, hue_order=l_hue_order,
                              dodge=True, color='k', size=set_marker_size.value)
            else:
                print("Function not implemented. Please go and annoy Dennis to finally do it")
                          
            if set_show_legend.value == True:
                if select_plot.value == 0:
                    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
                
                elif select_plot.value in [1, 2, 3]:
                    handles, labels = ax.get_legend_handles_labels()
                    new_handles = handles[:len(l_hue_order)]
                    new_labels = labels[:len(l_hue_order)]
                    ax.legend(new_handles, new_labels, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
            else:
                ax.get_legend().remove()

        else:
            print("Function not implemented. Please go and annoy Dennis to finally do it")
        
        if select_test.value == 0: # independent_samples()
            l_stats_to_annotate = get_l_stats_to_annotate_independent_samples()
            annotate_stats_independent_samples(l_stats_to_annotate)
        
        elif select_test.value == 2: # mixed_model_ANOVA()#
            l_stats_to_annotate = get_l_stats_to_annotate_mma()
            if select_plot.value == 0:
                annotate_stats_mma_pointplot(l_stats_to_annotate)
            elif select_plot.value in [1, 2, 3]:
                annotate_stats_mma_violinplot(l_stats_to_annotate)
            else:
                print("Function not implemented. Please go and annoy Dennis to finally do it")
            
        plt.ylabel(set_yaxis_label_text.value, fontsize=set_yaxis_label_fontsize.value, color=set_yaxis_label_color.value)
        plt.xlabel(set_xaxis_label_text.value, fontsize=set_xaxis_label_fontsize.value, color=set_xaxis_label_color.value)        
               
        if set_yaxis_scaling_mode.value == 1:
            plt.ylim(set_yaxis_lower_lim.value, set_yaxis_upper_lim.value)
        
        plt.tight_layout()
        
        if save_plot == True:
            plt.savefig('customized_plot.png', dpi=300)
        
        plt.show()
        
        
# 3.3 Download button:        
def on_download_button_clicked(b):
    global save_plot
    if select_downloads.value == 0 or select_downloads.value == 2:
        if select_test.value == 0:
            df_individual_group_stats = get_individual_group_stats_for_download(False)
            df_group_level_overview = get_group_level_stats_for_download()
            df_pairwise_comparisons = d_main['summary']['pairwise_comparisons'].copy()
            
        elif select_test.value == 2:
            df_individual_group_stats = get_individual_group_stats_for_download(True)
            df_group_level_overview = get_group_level_stats_for_download()
            df_pairwise_comparisons = d_main['summary']['pairwise_comparisons'].copy()
        
        with pd.ExcelWriter('statistic_results.xlsx') as writer:  
            df_individual_group_stats.to_excel(writer, sheet_name='Individual group statistics')
            df_group_level_overview.to_excel(writer, sheet_name='Whole-group statistics')
            df_pairwise_comparisons.to_excel(writer, sheet_name='Pairwise comparisons')                
                
    if select_downloads.value == 1 or select_downloads.value == 2:
        save_plot = True
        plotting_button.click()
        save_plot = False

        
###################################################################    

    
###################################################################
# 4 Functions that create the individual widget elements:
# 4.1 Buttons:
def create_buttons():
    global uploader, stats_button, plotting_button, download_button
    uploader = widgets.FileUpload(accept=('.xlsx,.csv'), multiple=False)
    stats_button = widgets.Button(description="Calculate stats", icon='rocket')
    plotting_button = widgets.Button(description='Plot the data', layout={'visibility': 'hidden'})
    download_button = widgets.Button(description='Download', icon='file-download', layout={'visibility': 'hidden'})

# 4.2 Dropdown menus:
def create_dropdowns():
    global select_test, select_plot, select_downloads
    select_test = widgets.Dropdown(options=[('pairwise comparison of two or more independent samples', 0), ('Mixed_model_ANOVA', 2)], 
                                   value=0, description='Please select which test you want to perform:',
                                   layout={'width': '700px'}, style={'description_width': 'initial'})

    select_plot = widgets.Dropdown(options=[('something initial', 0)], value=0,
                                   description='Please select which type of plot you want to create:',
                                   layout={'width': '700px', 'visibility': 'hidden'}, style={'description_width': 'initial'})

    select_downloads = widgets.Dropdown(options=[('statistical results only', 0), ('plot only', 1), ('both', 2)], value=1,
                                   description='Please select what you would like to write to disk:',
                                   layout={'width': '700px', 'visibility': 'hidden'}, style={'description_width': 'initial'})

# 4.3 Create all default widgets that allow customization of the stats annotations
#     and that don´t require any information about the data (e.g. how many groups)
def create_default_stats_annotation_widgets():
    global set_distance_stars_to_brackets, set_distance_brackets_to_data, set_fontsize_stars
    global set_linewidth_annotations, set_stars_fontweight_bold, set_annotate_all, select_bracket_no_bracket    
    
    # How far will the annotation lines be shifted from the data? Calculates as: 
    # y_shift_annotation_line = max(data) * set_distance_brackets_to_data.value
    set_distance_brackets_to_data = widgets.BoundedFloatText(description='Distance of the annotation bars to the graph',
                                                             style={'description_width': 'initial'}, value=0.1, min=0, max=1, 
                                                             step=0.005, layout={'width':'initial'})

    # Determines annotation_brackets_factor: 0 for 'No brackets', 1 for 'brackets'
    # brackets_height = y_shift_annotation_line*0.5*annotation_brackets_factor
    select_bracket_no_bracket = widgets.RadioButtons(options=['Brackets', 'No brackets'], 
                                                     value=('Brackets'), style={'description_width': 'initial'}, 
                                                     layout={'width': '300px', 'height': '50px'}, description='Annotation bar style:')
    
    # How far will the annotation stars be shifted from the annotation lines? Calculates as:
    # y_shift_annotation_text = y_shift_annotation_line + brackets_height + y_shift_annotation_line*0.5*set_distance_stars_to_brackets.value
    set_distance_stars_to_brackets = widgets.BoundedFloatText(description='Distance of the stars to the annotation bars', 
                                                              value=0.5, style={'description_width': 'initial'}, min=0, max=3, 
                                                              step=0.05, layout={'width':'initial'})
    
    set_fontsize_stars = widgets.BoundedFloatText(description='Fontsize of the stars', value=10, min=1, max=50, 
                                           style={'description_width': 'initial'}, layout={'width':'initial'})
    
    set_linewidth_annotations = widgets.BoundedFloatText(description='Linewidth of the annotation bars', 
                                                    value=1.5, min=0, max=10, step=0.1, layout={'width':'initial'}, 
                                                    style={'description_width': 'initial'})
    
    set_stars_fontweight_bold = widgets.Checkbox(description='Stars bold', value=False)
    
    customize_stats_annotation_vbox = VBox([HBox([set_stars_fontweight_bold, select_bracket_no_bracket]),
                                            set_distance_stars_to_brackets, set_distance_brackets_to_data, 
                                            set_fontsize_stars, set_linewidth_annotations]) 

    set_annotate_all = widgets.Checkbox(value=False, description='Annotate all', indent=False)

    return customize_stats_annotation_vbox


# 4.4 Create elements that allow the customization of the plot
# 4.4.1 Create and arrange the main accordion that has to be expanded by the user to access customization elements.
#       Triggers several functions that in turn create and/or arrange the respective elements.
def create_accordion_to_customize_the_plot():
    global expand_me_accordion, customization_accordion, select_annotations_vbox, customize_annotations_accordion

    # Still missing: 
        # Optional annotation of within and between statistics for mma
    customize_stats_annotation_vbox = create_default_stats_annotation_widgets()
    
    # Create empty VBox that will be filled with checkboxes to select individual pairwise 
    # comparisons that shall be annotated, as soon as the data is specified (stats_button.click())
    select_annotations_vbox = VBox([])
    
    select_annotations_accordion = widgets.Accordion(children=[select_annotations_vbox])
    select_annotations_accordion.set_title(0, 'Select individual comparisons for annotation')
    
    customize_annotations_accordion = widgets.Accordion(children=[VBox([select_annotations_accordion, set_annotate_all]), 
                                                                  customize_stats_annotation_vbox],
                                                       selected_index=None)
    
    customize_annotations_accordion.set_title(0, 'Select which stats shall be annotated')
    customize_annotations_accordion.set_title(1, 'Customize annotation features')
    
    # Second accordion will contain widgets to customize the axes
    customize_yaxis_vbox = create_vbox_y_axis()
    customize_xaxis_vbox = create_vbox_x_axis()
    customize_both_axes_hbox = create_hbox_both_axes()

    customize_axes_accordion = widgets.Accordion(children=[customize_yaxis_vbox, customize_xaxis_vbox, customize_both_axes_hbox])
    customize_axes_accordion.set_title(0, 'y-axis') 
    customize_axes_accordion.set_title(1, 'x-axis')
    customize_axes_accordion.set_title(2, 'common features')
    
    # Third accordion will contain widgets to customize the style of the plot (colorpalette, markersizes)
        # Still missing:
            # Plot size (2 sliders, x & y) to change fig_size [make sure violinplot annotation is still working for mma()]
                # e.g.: y_size=widgets.FloatSlider(description='Change the size of your plot.', value=1, min=0, max=10)
            # Make sure set_marker_size only shows up if possible to change
            # Plot title (+ size & color)
            # Option to remove upper and right spines
            # Set dpi
            # Select (.png, .tif, .pdf)
    customize_plot_features_hbox = create_customize_plot_features_hbox()
    
    # Create the accordion that actually contains all widget-containing accordions and will become the only child of the main accordion
    customization_accordion = widgets.Accordion(children=[customize_annotations_accordion, 
                                                          customize_axes_accordion, 
                                                          customize_plot_features_hbox], selected_index=None)

    # Give the individual accordions titles that are displayed before dropdown is clicked
    customization_accordion.set_title(0, 'Customize how statistics are annotated in the plot')
    customization_accordion.set_title(1, 'Customize axes')
    customization_accordion.set_title(2, 'Customize other features of the plot')

    # Create the main accordion that contains all widgets to customize the plot and use selected_index=None to avoid dropdown by default
    expand_me_accordion = widgets.Accordion(children=[customization_accordion], selected_index=None, continous_update=False, layout={'visibility': 'hidden'})
    expand_me_accordion.set_title(0, 'Expand me to customize your plot!')


# 4.4.2 Customization axes:
# 4.4.2.1 Create an HBox that allows customization of the y-axis
def create_vbox_y_axis():
    global set_yaxis_label_text, set_yaxis_label_fontsize, set_yaxis_label_color, set_yaxis_scaling_mode, set_yaxis_lower_lim, set_yaxis_upper_lim
    
    set_yaxis_label_text = widgets.Text(value='data', placeholder='data', description='y-axis title:', layout={'width': 'auto'})
    set_yaxis_label_fontsize = widgets.IntSlider(value=12, min=8, max=40, step=1, description='fontsize:')
    set_yaxis_label_color = widgets.ColorPicker(concise=False, description='font color', value='#000000')
    yaxis_hbox1 = HBox([set_yaxis_label_text, set_yaxis_label_fontsize, set_yaxis_label_color])
    
    set_yaxis_scaling_mode = widgets.RadioButtons(description = 'Please select whether you want to use automatic or manual scaling of the yaxis:', 
                                                              options=[('Use automatic scaling', 0), ('Use manual scaling', 1)],
                                                              value=0, layout={'width': '700px', 'height': '75px'}, style={'description_width': 'initial'})
    
    set_yaxis_lower_lim = widgets.FloatText(value=0.0, description='lower limit:', style={'description_width': 'initial'})
    set_yaxis_upper_lim = widgets.FloatText(value=0.0, description='upper limit:', style={'description_width': 'initial'})
    yaxis_hbox2 = HBox([set_yaxis_lower_lim, set_yaxis_upper_lim])

    return VBox([yaxis_hbox1, set_yaxis_scaling_mode, yaxis_hbox2])


# 4.4.2.2 Create an HBox that allows customization of the x-axis
def create_vbox_x_axis():
    global set_xaxis_label_text, set_xaxis_label_fontsize, set_xaxis_label_color, set_xlabel_order, set_hue_order
    set_xaxis_label_text = widgets.Text(value='group_IDs', placeholder='group_IDs', description='x-axis title:', layout={'width': 'auto'})
    set_xaxis_label_fontsize = widgets.IntSlider(value=12, min=8, max=40, step=1, description='fontsize:')
    set_xaxis_label_color = widgets.ColorPicker(concise=False, description='font color', value='#000000')
    xaxis_hbox = HBox([set_xaxis_label_text, set_xaxis_label_fontsize, set_xaxis_label_color])
    
    set_xlabel_order = widgets.Text(value='x label order', 
                                    placeholder='Specify the desired order of the x-axis labels with individual labels separated by a comma',
                                    description='x-axis label order (separated by comma):', 
                                    layout={'width': '800px', 'visibility': 'hidden'},
                                    style={'description_width': 'initial'})
    
    set_hue_order = widgets.Text(value='hue order',
                                 placeholder='Specify the desired group order with individual groups separated by a comma',
                                 description='group order (separated by comma):',
                                 layout={'width': '800px', 'visibility': 'hidden'},
                                 style={'description_width': 'initial'})
    
    
    
    return VBox([xaxis_hbox, set_xlabel_order, set_hue_order])


# 4.4.2.3 Create an HBox that allows customization of general axis features
def create_hbox_both_axes():
    global set_axes_linewidth, set_axes_color, set_axes_tick_size
    set_axes_linewidth = widgets.BoundedFloatText(value=1, min=0, max=40, description='Axes linewidth', 
                                           style={'description_width': 'initial'}, layout={'width': 'auto'})
    set_axes_color = widgets.ColorPicker(concise=False, description='Axes and tick label color', 
                                         value='#000000', style={'description_width': 'initial'}, layout={'width': 'auto'})
    set_axes_tick_size = widgets.BoundedFloatText(value=10, min=1, max=40, description='Tick label size', 
                                            style={'description_width': 'initial'}, layout={'width': 'auto'})
    return HBox([set_axes_linewidth, set_axes_color, set_axes_tick_size])


# 4.4.3 Customize general features of the plot (like colors, size, ...)
def create_customize_plot_features_hbox():
    global select_color_palettes, set_marker_size, select_palette_or_individual_color, group_colors_vbox
    global plot_style_features_hbox, set_fig_width, set_fig_height, set_show_legend
    select_palette_or_individual_color = widgets.RadioButtons(description = 'Please select a color code option and chose from the respective options below:', 
                                                              options=[('Use a pre-defined palette', 0), ('Define colors individually', 1)],
                                                              value=0, layout={'width': '700px', 'height': '75px'}, style={'description_width': 'initial'})
    
    select_color_palettes = widgets.Dropdown(options=['colorblind', 'Spectral', 'viridis', 'rocket', 'cubehelix'],
                             value='colorblind',
                             description='Select a color palette', 
                             layout={'width': '350'},
                             style={'description_width': 'initial'})
    
    set_show_legend = widgets.Checkbox(value=True, description='Show legend (if applicable):', style={'description_width': 'initial'})
    set_marker_size = widgets.FloatText(value=5,description='marker size (if applicable):', style={'description_width': 'initial'})
    
    optional_features_hbox = HBox([set_show_legend, set_marker_size])
    
    # Empty VBox which will be filled as soon as groups are determined (stats_button.click())
    group_colors_vbox = VBox([])
    
    set_fig_width = widgets.FloatSlider(value=28, min=3, max=30, description='Figure width:', style={'description_width': 'inital'})
    set_fig_height = widgets.FloatSlider(value=16, min=3, max=30, description='Figure height:', style={'description_width': 'inital'})
    fig_size_hbox = HBox([set_fig_width, set_fig_height])
    
    plot_style_features_vbox = VBox([select_palette_or_individual_color, HBox([select_color_palettes, group_colors_vbox]), fig_size_hbox, optional_features_hbox])
    return plot_style_features_vbox


# 4.5 Create elements that are dependent on group information:
# 4.5.1 Create checkboxes to select individual comparisons that shall be annotated
# 4.5.1.1 Base-function: create and arrange checkboxes of all possible pairwise comparisons
def create_checkboxes_pairwise_comparisons():
    # Create a checkbox for each pairwise comparison
    l_checkboxes_temp = [widgets.Checkbox(value=False,description='{} vs. {}'.format(group1, group2)) 
                         for group1, group2 in list(itertools.combinations(l_groups, 2))]
    # Arrange checkboxes in a HBoxes with up to 3 checkboxes per HBox
    l_HBoxes = []
    elem = 0
    for i in range(int(len(l_checkboxes_temp)/3)):
        l_HBoxes.append(HBox(l_checkboxes_temp[elem:elem+3]))
        elem = elem + 3

    if len(l_checkboxes_temp) % 3 != 0:
        l_HBoxes.append(HBox(l_checkboxes_temp[elem:]))

    # Arrange HBoxes in a VBox and select all as tuple to later place in empty placeholder (select_annotations_vbox)
    checkboxes_to_add_temp = VBox(l_HBoxes).children[:]

    return checkboxes_to_add_temp, l_checkboxes_temp 


# 4.5.1.2 Create checkboxes taking session_id into account (for mixed-model ANOVA):
def create_checkboxes_pairwise_comparisons_mma():
    annotate_session_stats_accordion = widgets.Accordion(children=[], selected_index=None)
    l_all_checkboxes = []

    for session_id in l_sessions:
        checkboxes_to_add_temp, l_checkboxes_temp = create_checkboxes_pairwise_comparisons()
        # Little complicated, but neccessary since the output of create_checkboxes_pairwise_comparisons() is a tuple
        checkboxes_to_add_temp_vbox = VBox([])
        checkboxes_to_add_temp_vbox.children = checkboxes_to_add_temp_vbox.children + checkboxes_to_add_temp
        annotate_session_stats_accordion.children = annotate_session_stats_accordion.children + (checkboxes_to_add_temp_vbox, )
        l_all_checkboxes = l_all_checkboxes + [(session_id, elem) for elem in l_checkboxes_temp]

    for i in range(len(list(annotate_session_stats_accordion.children))):
        annotate_session_stats_accordion.set_title(i, l_sessions[i])
    
    return VBox([annotate_session_stats_accordion]).children[:], l_all_checkboxes


# 4.5.2 Create color pickers that allow the user to specify a color for each group
def create_group_color_pickers():
    for group_id in l_groups:
        set_group_color = widgets.ColorPicker(concise=False, description = group_id, style={'description_width': 'initial'})
        group_colors_vbox.children = group_colors_vbox.children + (set_group_color, )


# 4.5.3 Specify the group order string:
def create_group_order_text():
    if select_test.value == 0:
        for group_id in l_groups:
            if l_groups.index(group_id) == 0:
                l_xlabel_string = group_id
            else:
                l_xlabel_string = l_xlabel_string + ', {}'.format(group_id)
        set_xlabel_order.value = l_xlabel_string
        set_xlabel_order.layout.visibility = 'visible'
        
    elif select_test.value == 2:
        for session_id in l_sessions:
            if l_sessions.index(session_id) == 0:
                l_xlabel_string = session_id
            else:
                l_xlabel_string = l_xlabel_string + ', {}'.format(session_id)
        set_xlabel_order.value = l_xlabel_string
        set_xlabel_order.layout.visibility = 'visible'
        
        for group_id in l_groups:
            if l_groups.index(group_id) == 0:
                l_hue_string = group_id
            else:
                l_hue_string = l_hue_string + ', {}'.format(group_id)
        set_hue_order.value = l_hue_string
        set_hue_order.layout.visibility = 'visible'        
        
def create_ylims():
    if df[data_col].min() < 0:
        set_yaxis_lower_lim.value = round(df[data_col].min() + df[data_col].min()*0.1, 2)
    else:
        set_yaxis_lower_lim.value = round(df[data_col].min() - df[data_col].min()*0.1, 2)
        
    if df[data_col].max() < 0:
        set_yaxis_upper_lim.value = round(df[data_col].max() - df[data_col].max()*0.1, 2)
    else:
        set_yaxis_upper_lim.value = round(df[data_col].max() + df[data_col].max()*0.1, 2)
    
###################################################################    

    
###################################################################
# 5 Specify the layout of the widget and define the launch function
# 5.1 Top level widget layout
def top_level_layout():
    global stats_widget

    create_accordion_to_customize_the_plot()
    create_dropdowns()
    create_buttons()

    # Bind the on_button_clicked functions to the respective buttons:
    stats_button.on_click(on_stats_button_clicked)
    plotting_button.on_click(on_plotting_button_clicked) 
    download_button.on_click(on_download_button_clicked)
    
    # Layout of the remaining elements
    first_row = HBox([uploader])
    second_row = HBox([select_test, stats_button])
    third_row = HBox([select_plot, plotting_button])
    third_row_extension = HBox([expand_me_accordion])
    fourth_row = HBox([select_downloads, download_button])

    stats_widget = VBox([first_row, second_row, third_row, third_row_extension, fourth_row])


# 5.2 Launch function
def launch():
    global output
    # Configure the layout:
    top_level_layout()
    # Define the output
    output = widgets.Output()
    # Display the widget:
    display(stats_widget, output)
    
###################################################################    

    
###################################################################
# 6 Functions to process the statistical data for download:
# 6.1 Calculate individual group statistics:
def calculate_individual_group_stats(d, key):
    group_data = d_main[key]['data']
    d['means'].append(np.mean(group_data))
    d['medians'].append(np.median(group_data))
    d['stddevs'].append(np.std(group_data))
    d['stderrs'].append(np.std(group_data) / math.sqrt(group_data.shape[0]))
    d['tests'].append('Shapiro-Wilk')
    d['test_stats'].append(d_main[key]['normality_full'].iloc[0,0])
    d['pvals'].append(d_main[key]['normality_full'].iloc[0,1])
    d['bools'].append(d_main[key]['normality_full'].iloc[0,2])
    return d


# 6.2 Create the DataFrame:
def get_individual_group_stats_for_download(include_sessions):
    d_individual_group_stats = {'means': [],
                                'medians': [],
                                'stddevs': [],
                                'stderrs': [],
                                'tests': [],
                                'test_stats': [], 
                                'pvals': [], 
                                'bools': []}

    l_for_index = []
    
    if include_sessions == False:
        # for independent samples:
        for group_id in l_groups:
            d_individual_group_stats = calculate_individual_group_stats(d_individual_group_stats, group_id)
            l_for_index.append(group_id)
        l_index = l_for_index
    else:
        # for mma:
        for group_id in l_groups:
            for session_id in l_sessions:
                d_individual_group_stats = calculate_individual_group_stats(d_individual_group_stats, (group_id, session_id))
                l_for_index.append((group_id, session_id))
            l_index = pd.MultiIndex.from_tuples(l_for_index)
                
    df_individual_group_stats = pd.DataFrame(data=d_individual_group_stats)

    multi_index_columns = pd.MultiIndex.from_tuples([('Group statistics', 'Mean'), ('Group statistics', 'Median'), ('Group statistics', 'Standard deviation'), ('Group statistics', 'Standard error'),
                                             ('Test for normal distribution', 'Test'), ('Test for normal distribution', 'Test statistic'), ('Test for normal distribution', 'p-value'),
                                             ('Test for normal distribution', 'Normally distributed?')])

    df_individual_group_stats.columns = multi_index_columns
    df_individual_group_stats.index = l_index

    return df_individual_group_stats


# 6.3 Group-level statistics:
def get_group_level_stats_for_download():
    df_group_level_overview = pg.homoscedasticity([d_main[key]['data'] for key in d_main.keys() if key != 'summary'])
    df_group_level_overview.index = [0]
    df_group_level_overview.columns = pd.MultiIndex.from_tuples([('Levene', 'W statistic'), ('Levene', 'p value'), ('Levene', 'Equal variances?')])

    df_group_level_overview[('', 'all normally distributed?')] = False
    df_group_level_overview[('', 'critera for parametric test fulfilled?')] = False
    df_group_level_overview[('', 'performed test')] = performed_test
    df_group_level_overview[' '] = ''

    df_group_statistics = d_main['summary']['group_level_statistic'].copy()
    
    df_group_statistics.index = list(range(df_group_statistics.shape[0]))
    df_group_statistics.columns = pd.MultiIndex.from_tuples([(performed_test, elem) for elem in df_group_statistics.columns])

    df_group_level_overview = pd.concat([df_group_level_overview, df_group_statistics], axis=1)
    
    return df_group_level_overview