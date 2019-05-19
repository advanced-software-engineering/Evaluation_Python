import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.pyplot import figure
from evaluator.statistic import Statistic

class Evaluator:
    statistics = []
    baseline_statistic = {}

    def __init__(self, args):
        if not args[1]:
            print('No csv folder path provided')
            return
        csv_folder_path = args[1]
        files = os.listdir(csv_folder_path)
        for file in files:
            if file.__contains__('receiverType') or file.__contains__('baseline_new'):
                continue
            dataframe = pd.read_csv(os.path.join(csv_folder_path, file), engine='python', error_bad_lines=False)
            statistic = Statistic(file, dataframe)
            statistic.calculate_statistics()
            if statistic.name == 'baseline':
                self.baseline_statistic = statistic
            else:
                self.statistics.append(statistic)

    def create_charts(self):

        # Plot accuracy
        if len(self.statistics) != 0:
            self.baseline_statistic.plot_accuracy()

        # Plot top 5
        self.baseline_statistic.plot_top5()
        for statistic in self.statistics:
            statistic.plot_top5()

        # Plot weight changes
        self.plot_top5_changes()

        # Plot similarity correlation
        self.plot_similarity_correlation()

    def print_general_statistics(self):
        statistic = self.baseline_statistic

        # Header
        print(statistic.name + ' (', end='')
        for key, value in statistic.weights.items():
            print(key + ': ' + str(value) + ', ', end='')
        print(')')
        print('')

        print(str(statistic.total_evaluated) + '/' + str(statistic.total_recommendations) + ' recommendations evaluated (' + str(statistic.evaluated_ratio) + ')')
        print(str(statistic.total_not_evaluated) + '/' + str(statistic.total_recommendations) + ' recommendations not evaluated (' + str(statistic.not_evaluated_ratio) + ')')
        print('')
        print('Top1: ' + str(statistic.correct_top1) + ' (' + str(statistic.correct_top1_ratio) + ')')
        print('Top2: ' + str(statistic.correct_top2) + ' (' + str(statistic.correct_top2_ratio) + ')')
        print('Top3: ' + str(statistic.correct_top3) + ' (' + str(statistic.correct_top3_ratio) + ')')
        print('Top4: ' + str(statistic.correct_top4) + ' (' + str(statistic.correct_top4_ratio) + ')')
        print('Top5: ' + str(statistic.correct_top5) + ' (' + str(statistic.correct_top5_ratio) + ')')
        print('----------------------------------------')

    def plot_top5_changes(self):
        data = {
            'requiredType': [],
            'objectOrigin': [],
            'surroundingExpression': [],
            'enclosingMethodReturnType': [],
            'enclosingMethodParameterSize': [],
            'enclosingMethodParameters': [],
            'enclosingMethodSuper': []
        }

        fig = plt.figure(figsize=(25, 15))
        axarr = [
            fig.add_subplot(2, 4, 1),
            fig.add_subplot(2, 4, 2),
            fig.add_subplot(2, 4, 3),
            fig.add_subplot(2, 4, 4),
            fig.add_subplot(2, 4, 5),
            fig.add_subplot(2, 4, 6),
            fig.add_subplot(2, 4, 7)
        ]
        fig.suptitle('Recommendation quality change for different weights')
        for statistic in self.statistics:
            data[statistic.name].append(statistic)
        count = 0
        for key, value in data.items():
            value.append(self.baseline_statistic)
            value.sort(key=lambda x: float(x.weights[key]))

            top1 = []
            top3 = []
            top5 = []
            x_values = np.arange(0, 2.25, 0.25)
            y_tick_labels = ['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%']
            y_ticks = np.arange(0, 1.1, 0.1)
            for statistic in value:
                top1.append(statistic.correct_top1_ratio)
                top3.append(statistic.correct_top3_ratio)
                top5.append(statistic.correct_top5_ratio)
            l1,= axarr[count].plot(x_values, top1, color='blue', marker='x', linestyle='-.')
            l3,= axarr[count].plot(x_values, top3, color='green', marker='v', linestyle='--')
            l5,= axarr[count].plot(x_values, top5, color='orange', marker='o', linestyle='-')
            axarr[count].set_title(statistic.name)
            axarr[count].set_xlabel('Weight')
            axarr[count].set_ylabel('Accuracy')
            axarr[count].set_yticks(y_ticks)
            axarr[count].set_yticklabels(y_tick_labels)
            axarr[count].set_ylim(0.2, 0.6)
            axarr[count].legend([l1, l3, l5], ['Top 1', 'Top 3', 'Top 5'], loc=4)

            count += 1
        plt.savefig('plots\\weight_change.png')
        plt.clf()

    def plot_similarity_correlation(self):
            dataframe = self.baseline_statistic.dataframe
            dataframe = dataframe.loc[dataframe['evaluated'] == True]

            correct = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            incorrect = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            for index, row in dataframe.iterrows():
                bin_index = 9
                similarity = row['similarity_1']

                if similarity < 0.1:
                    bin_index = 0
                elif 0.1 <= similarity < 0.2:
                    bin_index = 1
                elif 0.2 <= similarity < 0.3:
                    bin_index = 2
                elif 0.3 <= similarity < 0.4:
                    bin_index = 3
                elif 0.4 <= similarity < 0.5:
                    bin_index = 4
                elif 0.5 <= similarity < 0.6:
                    bin_index = 5
                elif 0.6 <= similarity < 0.7:
                    bin_index = 6
                elif 0.7 <= similarity < 0.8:
                    bin_index = 7
                elif 0.8 <= similarity < 0.9:
                    bin_index = 8
                elif similarity >= 0.9:
                    bin_index = 9

                if row['selectedMethod'] == row['recommendedMethod_1']:
                    correct[bin_index] += 1
                else:
                    incorrect[bin_index] += 1

            data = []
            for i in range(10):
                ratio = 0.0
                if correct[i] != 0 or incorrect[i] != 0:
                    ratio = correct[i] / (correct[i] + incorrect[i])
                data.append(ratio)

            x_coordinates = np.arange(len(data))
            x_ticks = ['0 - 0.1', '0.1 - 0.2', '0.2 - 0.3', '0.3 - 0.4', '0.4 - 0.5', '0.5 - 0.6', '0.6 - 0.7', '0.7 - 0.8', '0.8 - 0.9', '0.9 - 1']
            y_coordinates = np.arange(0, 1.1, 0.1)
            y_labels = ['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%']

            plt.rcParams.update({'font.size': 22})
            plt.bar(x=x_coordinates, height=data, align='center')
            plt.xticks(x_coordinates, x_ticks)
            plt.xlabel('Similarity')
            plt.yticks(y_coordinates, y_labels)
            plt.ylabel('Accuracy')
            plt.savefig('plots\\similarity_correlation.png')
            plt.clf()
