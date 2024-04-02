

from main import DataRepresentationBuilder


# TODO: move relevant initial data processing methods here (train/paramopt/test splits, threshold efficacy plots, etc.)

def plot_thresholds(self, df_, figure_label_, output_dir__='', savefig=True):
    fig, ax = plt.subplots()
    fig.set_size_inches(w=5, h=4)

    colors_ls = [x.replace('inefficient', '#3AA6E2').replace('efficient', '#F7B531').replace('undefined', '#B6B6B7') for
                 x in list(df_.sort_values(by=self.expr_key)['class'])]
    ax.bar(
        x=list(range(len(df_))),
        height=df_.sort_values(by=self.expr_key)[self.expr_key],
        color=colors_ls,
        # width=(1.0),
    )

    container2 = ax.errorbar(
        list(range(len(df_))),
        df_.sort_values(by=self.expr_key)[self.expr_key],
        yerr=df_.sort_values(by=self.expr_key)['standard_deviation'],
        lolims=True,
        color='black',
    )

    connector, (caplines,), (vertical_lines,) = container2.lines
    connector.set_alpha(0)
    caplines.set_solid_capstyle('butt')
    try:
        caplines.set_marker(None)
    except:
        pass
    vertical_lines.set_linewidth(1.0)  # 0.5)

    ax.set_ylim(0, max(df_[self.expr_key]) + 0.2 * max(df_[self.expr_key]))
    ax.set_xlim(0, len(df_))
    ax.set_ylabel('Target Gene Expression (%)\nNormalized Per Assay (Gene)')
    ax.set_xlabel('siRNAs (' + str(len(df_)) + ' total)')
    ax.tick_params(axis='x', bottom=False, labelbottom=False)  # remove x-axis ticks and labels

    # Legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor='#F7B531', edgecolor=None,
              label=('< ' + str(self.effco_) + '% : Efficient (' + str(len(df_[df_['class'] == 'efficient'])) + ' siRNAs)')),
        Patch(facecolor='#3AA6E2', edgecolor=None, label=('≥ ' + str(self.ineffco_) + '% : Inefficient (' + str(
            len(df_[df_['class'] == 'inefficient'])) + ' siRNAs)')),
        Patch(facecolor='#B6B6B7', edgecolor=None,
              label=('Undefined (' + str(len(df_[df_['class'] == 'undefined'])) + ' siRNAs)')),
    ]
    ax.legend(handles=legend_elements, loc='upper left', frameon=False, fontsize=9)

    plt.title(figure_label_)
    # plt.title(output_run_file_info_string_.split('effco')[0].replace('_',' ').replace('-',' ')+'\n'+'Thresholds <'+str(effco_)+'% | ≥'+str(ineffco_)+'%',fontsize=12)

    fig.tight_layout()

    if savefig:
        # ** SAVE FIGURE **
        plt.rcParams['svg.fonttype'] = 'none'  # exports text as strings rather than vector paths (images)
        fnm_ = (all_output_dir + output_dir__ + figure_label_.split('\n')[0].replace(' ', '_').lower().replace('%',
                                                                                                               'pcnt') + '_partition')
        fnm_svg_ = (all_output_dir + output_dir__ + 'svg_figs/' + figure_label_.split('\n')[0].replace(' ',
                                                                                                       '_').lower().replace(
            '%', 'pcnt') + '_partition')

        fig.savefig(fnm_svg_.split('.')[0] + '.svg', format='svg', transparent=True)

        fig.savefig(fnm_.split('.')[0] + '.png', format='png', dpi=300, transparent=False)
        print('Figure saved to:', fnm_ + '.png')


def plot_proportions_pie(self, figure_label_, output_dir__, df_, undefined_df_, train_split_df_, split_initial_df_,
                         df_train_kfld_, df_test_kfld_, df_paramopt_kfld_, round_ct_, savefig=True):
    # Checking porportions of partitioned data roughly match input parameters

    fig, ax = plt.subplots(1)  # 1,3)
    fig.set_size_inches(w=5, h=5)
    # For Plotting
    import matplotlib.pylab as pylab
    params = {'legend.fontsize': 12,
              'font.size': 12,
              }
    pylab.rcParams.update(params)

    ax.set_title('Round ' + str(round_ct_ + 1) + ' Partition\nTotal siRNAs: ' + str(len(train_split_df_)))
    ax.pie(
        [len(df_train_kfld_), len(df_test_kfld_), len(df_paramopt_kfld_)],  # len(split_initial_df_) ],
        autopct='%1.f%%',
        startangle=90,
        colors=['#DB7AC8', '#FECC0A', '#28A18B'],  # '#28A18B'],
    )

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor='#DB7AC8', edgecolor=None, label=(
                str(100 - self.split_set_size_pcnt_) + '% Round ' + str(round_ct_ + 1) + ' Training Dataset (' + str(
            len(df_train_kfld_)) + ')')),
        Patch(facecolor='#FECC0A', edgecolor=None, label=(
                str(self.test_set_size_pcnt_) + '% Round ' + str(round_ct_ + 1) + ' Testing Dataset (' + str(
            len(df_test_kfld_)) + ')')),
        Patch(facecolor='#28A18B', edgecolor=None, label=(str(self.paramopt_set_size_pcnt_) + '% Round ' + str(
            round_ct_ + 1) + ' Parameter Optimization Dataset (' + str(len(df_paramopt_kfld_)) + ')')),
    ]

    ax.legend(handles=legend_elements, frameon=False, loc='upper right', bbox_to_anchor=(1.5, 0.1))
    fig.tight_layout()  # NOTE: h and w (above in fig.set_size... MUST be large enough to accomodate legends or will be cut off/squished in output)

    if savefig:
        # ** SAVE FIGURE **
        plt.rcParams['svg.fonttype'] = 'none'  # exports text as strings rather than vector paths (images)
        fnm_ = (all_output_dir + output_dir__ + figure_label_.split('\n')[0].replace(' ', '_').lower().replace('%',
                                                                                                               'pcnt') + '_partition')
        fnm_svg_ = (all_output_dir + output_dir__ + 'svg_figs/' + figure_label_.split('\n')[0].replace(' ',
                                                                                                       '_').lower().replace(
            '%', 'pcnt') + '_partition')

        fig.savefig(fnm_svg_.split('.')[0] + '.svg', format='svg', transparent=True)

        fig.savefig(fnm_.split('.')[0] + '.png', format='png', dpi=300, transparent=False)
        print('Figure saved to:', fnm_ + '.png')



