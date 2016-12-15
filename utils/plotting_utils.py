import numpy as np
import matplotlib.pyplot as plt
import re


def create_bar_plot(labels, x_values, y_values, y_label, x_label, title):
    """
    Creating a bar plot from the given values and saving it in a file
    :param labels:
    :param x_values:
    :param y_values:
    :param y_label:
    :param x_label:
    :param title:
    :return:
    """
    # Create values and labels for bar chart
    if len(labels) != len(x_values) or len(x_values) != len(y_values):
        print "Inconsistent dimensions of lables and values" \
              "!!!"
        return
    # Plot a bar chart
    plt.close('all')
    #plt.figure(1, figsize=(5, 4))  # 6x4 is the aspect ratio for the plot
    plt.bar(x_values, y_values, align='center')  # This plots the data
    plt.grid(True)  # Turn the grid on
    plt.ylabel(y_label)  # Y-axis label
    plt.xlabel(x_label)  # X-axis label
    plt.title(title)  # Plot title
    plt.xlim(np.min(x_values)-0.5,np.max(x_values)+0.5) #set x axis range
    plt.ylim(np.min(y_values)-0.002, np.max(y_values)+0.001)  # Set yaxis range
    # Set the bar labels
    plt.gca().set_xticks(x_values)  # label locations
    plt.gca().set_xticklabels(labels)  # label values
    # Save the chart
    plt.savefig("../Figures/bar_plot_" + re.sub(" ", "_", title.lower()) + ".jpg")


def create_line_plot(labels, x_values, y_values, y_label, x_label, title):
    """
    Creating a line plot from the given values and saving it in a file
    :param labels:
    :param x_values:
    :param y_values:
    :param y_label:
    :param x_label:
    :param title:
    :return:
    """
    if len(labels) != len(x_values):
        print "Inconsistent dimensions of lables and values!!!"
        return
    plt.close('all')
    plt.figure(2, figsize=(6, 4))  # 6x4 is the aspect ratio for the plot
    plt.plot(x_values, y_values, 'or-', linewidth=3)  # Plot the first series in red with circle marker
    # This plots the data
    plt.grid(True)  # Turn the grid on
    plt.ylabel(y_label)  # Y-axis label
    plt.xlabel(x_label)  # X-axis label
    plt.title(title)  # Plot title
    plt.xlim(np.min(x_values), np.max(x_values))  # set x axis range
    plt.ylim(np.min(y_values), np.max(y_values))  # Set yaxis range
    plt.gca().set_xticks(x_values)  # label locations
    plt.gca().set_xticklabels(labels)  # label values
    # Save the chart
    plt.savefig("../Figures/line_plot_" + re.sub(" ", "_", title.lower()) + ".jpg")


def create_bar_plot_clustered(min_range,max_range,labels,x_label,y_label,title):
    plt.close('all')
    f, axarr = plt.subplots()
    index = np.arange(len(min_range))
    bar_width = 0.6
    opacity = 0.4
    index = np.arange(len(max_range))
    plt.gca().set_xticks(index + bar_width / 2)  # label locations
    plt.gca().set_xticklabels(labels)
    plt1= axarr.bar(index, min_range, alpha=opacity, color='b', width=bar_width / 2)
    plt2 = axarr.bar(index + bar_width / 2, max_range, alpha=opacity, color='y', width=bar_width / 2)
    plt.ylabel(y_label)  # Y-axis label
    plt.xlabel(x_label)  # X-axis label
    plt.title(title)  # Plot title
    plt.legend((plt1[0], plt2[1]), ("Min Value", "Max Value"))
    plt.tight_layout()
    plt.savefig("../Figures/bar_plot_" + re.sub(" ", "_", title.lower()) + ".jpg")


def create_pv_revenue_lr_plots(df):
    plt.close('all')
    f, axarr = plt.subplots(1,8,sharex=True)
    column_names=map(lambda x: "Mean_PV_For_PrevWeek_%d"%x,range(1,9))
    for row in range(1):
        for column in range(8):
            x = df[column_names[row*8 + column]]
            y = np.log10(df['Weekend_Revenue'])
            fit = np.polyfit(x,y, deg=1)
            print len(axarr)
            print len(fit)
            axarr[row,column].plot(x, fit[0] * x + fit[1], color='blue')
            axarr[row, column].scatter(x, y)
            #axarr[row, column].set_title("Revenue vs "+ column_names[row*4 + column])
    plt.savefig("../Figures/revenue_pv_comparison.jpg")
