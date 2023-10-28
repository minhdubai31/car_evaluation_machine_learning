import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np



# Function to create a heatmap chart displaying correlation matrix
# and save it to an image file
def export_heatmap(data, save_path, round_decimals=3, title='Biểu đồ tương quan'):
    # Set size for the chart
    plt.figure(figsize=(18, 15))

    # Create the heatmap
    sns.set(font_scale=2)
    sns.heatmap(data.corr().round(round_decimals),
                annot=True, cmap='crest', linewidths=1)
    
    # Set title for the chart
    plt.title(title, fontsize=30, fontweight='bold', pad=30)
    
    # Export the chart to file
    plt.savefig(save_path)


# Function to create a pie chart displaying the ratio between the values 
# of the feature and save it to image files
def export_values_distribution(column, save_path, title='Phân bố giá trị'):
    labels = np.unique(column)
    values = column.value_counts()
    
    # Set size for the chart
    plt.figure(figsize=(15, 15))
    plt.rc('font', size=24)

    # Create the pie chart
    plt.pie(x=values, labels=labels, colors=sns.color_palette(), autopct='%1.1f%%')

    # Set title for the chart
    plt.title(title, fontsize=30, fontweight='bold')

    # Export the chart to file
    plt.savefig(save_path)


# Function to create a line chart displaying each model's test scores
# and save it to an image file
def export_line_chart(x_arr, y_arr, labels_arr, colors_arr, save_path):
    # Set size for the chart
    plt.figure(figsize=(17, 11))
    plt.rc('font', size=18)

    for index, arr in enumerate(y_arr):
        plt.plot(x_arr, arr, label=labels_arr[index], color=colors_arr[index], linewidth=3) 
    
    plt.xlabel('Lần test', labelpad=30, fontstyle='italic') 
    plt.ylabel('F1 score', labelpad=30, fontstyle='italic') 
    plt.ylim(0, 100)
    plt.yticks(np.arange(0, 101, 10))
    plt.xlim(1, max(x_arr))
    plt.title('Biểu đồ độ chính xác của các model', fontweight='bold', pad=30)
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))
    plt.tight_layout()

    plt.savefig(save_path)


# Function to create a horizontal columns chart displaying the importance level
# of each feature and save it to an image file
def export_feature_importances(feature_importances, index, savepath):
    feature_imp = pd.Series(feature_importances, index=index).sort_values(ascending=False)

    plt.figure(figsize=(15,10))
    plt.rc('font', size=18)
    # Creating a bar plot
    sns.barplot(x=feature_imp, y=feature_imp.index, hue=feature_imp.index, palette='tab10')
    # Add labels to the graph
    plt.xlabel('Điểm quan trọng của thuộc tính', labelpad=30, fontstyle='italic')
    plt.ylabel('Thuộc tính', labelpad=30, fontstyle='italic')
    plt.title("Biểu đồ sự quan trọng của các thuộc tính", fontweight='bold', pad='30')
    plt.tight_layout()

    plt.savefig(savepath)


# Function to convert data's string values to numeric values
def car_data_to_numeric(data):
    data_processed = data.copy()

    data_processed['buying'] = data_processed['buying'].replace({
        'low': 0,
        'med': 1,
        'high': 2,
        'vhigh': 3
    })

    data_processed['maint'] = data_processed['maint'].replace({
        'low': 0,
        'med': 1,
        'high': 2,
        'vhigh': 3
    })

    data_processed['doors'] = data_processed['doors'].replace({
        '2': 0,
        '3': 1,
        '4': 2,
        '5more': 3
    })

    data_processed['persons'] = data_processed['persons'].replace({
        '2': 0,
        '4': 1,
        'more': 2
    })

    data_processed['lug_boot'] = data_processed['lug_boot'].replace({
        'small': 0,
        'med': 1,
        'big': 2
    })

    data_processed['safety'] = data_processed['safety'].replace({
        'low': 0,
        'med': 1,
        'high': 2
    })

    data_processed['class'] = data_processed['class'].replace({
        'unacc': 0,
        'acc': 1,
        'good': 2,
        'vgood': 3
    })

    return data_processed
