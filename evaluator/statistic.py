import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class Statistic:
    def __init__(self, file_name, dataframe):
        self.name = ''
        self.weights = {
            'requiredType': 1.0,
            'objectOrigin': 1.0,
            'surroundingExpression': 1.0,
            'enclosingMethodReturnType': 1.0,
            'enclosingMethodParameterSize': 1.0,
            'enclosingMethodParameters': 1.0,
            'enclosingMethodSuper': 1.0
        }
        self.set_name_and_weights(file_name)
        self.dataframe = dataframe

        self.total_recommendations = self.dataframe.shape[0]
        self.total_evaluated = 0
        self.total_not_evaluated = 0
        self.evaluated_ratio = 0.0
        self.not_evaluated_ratio = 0.0

        self.correct_top1 = 0
        self.correct_top1_ratio = 0.0
        self.correct_top2 = 0
        self.correct_top2_ratio = 0.0
        self.correct_top3 = 0
        self.correct_top3_ratio = 0.0
        self.correct_top4 = 0
        self.correct_top4_ratio = 0.0
        self.correct_top5 = 0
        self.correct_top5_ratio = 0.0

    def set_name_and_weights(self, file_name):
        name = file_name.replace('ASE_Evaluation_', '').replace('.csv', '')

        if name == 'baseline':
            self.name = name
            return

        split = name.split('_', 1)
        self.name = split[0]
        weight = split[1]

        if self.name == 'requiredType':
            self.weights['requiredType'] = weight
        elif self.name == 'objectOrigin':
            self.weights['objectOrigin'] = weight
        elif self.name == 'surroundingExpression':
            self.weights['surroundingExpression'] = weight
        elif self.name == 'enclosingMethodReturnType':
            self.weights['enclosingMethodReturnType'] = weight
        elif self.name == 'enclosingMethodParameterSize':
            self.weights['enclosingMethodParameterSize'] = weight
        elif self.name == 'enclosingMethodParameters':
            self.weights['enclosingMethodParameters'] = weight
        elif self.name == 'enclosingMethodSuper':
            self.weights['enclosingMethodSuper'] = weight

    def calculate_statistics(self):
        self.calculate_accuracy()
        self.calculate_top_5()

    def calculate_accuracy(self):
        series = self.dataframe['evaluated']
        self.total_evaluated = sum(series)
        self.total_not_evaluated = self.total_recommendations - self.total_evaluated
        self.evaluated_ratio = self.total_evaluated / self.total_recommendations
        self.not_evaluated_ratio = 1 - self.evaluated_ratio

    def calculate_top_5(self):
        for index, row in self.dataframe.iterrows():
            if not row['evaluated']:
                continue

            selected_method = row['selectedMethod']
            rec_method1 = row['recommendedMethod_1']
            rec_method2 = row['recommendedMethod_2']
            rec_method3 = row['recommendedMethod_3']
            rec_method4 = row['recommendedMethod_4']
            rec_method5 = row['recommendedMethod_5']

            self.total_recommendations += 1

            condition = selected_method == rec_method1
            if condition:
                self.correct_top1 += 1
            condition = condition or selected_method == rec_method2
            if condition:
                self.correct_top2 += 1
            condition = condition or selected_method == rec_method3
            if condition:
                self.correct_top3 += 1
            condition = condition or selected_method == rec_method4
            if condition:
                self.correct_top4 += 1
            condition = condition or selected_method == rec_method5
            if condition:
                self.correct_top5 += 1

            self.correct_top1_ratio = self.correct_top1 / self.total_evaluated
            self.correct_top2_ratio = self.correct_top2 / self.total_evaluated
            self.correct_top3_ratio = self.correct_top3 / self.total_evaluated
            self.correct_top4_ratio = self.correct_top4 / self.total_evaluated
            self.correct_top5_ratio = self.correct_top5 / self.total_evaluated

    def plot_accuracy(self):
        data = [self.total_evaluated, self.total_not_evaluated]
        labels = ['evaluated', 'not evaluated']
        plt.pie(x=data, labels=labels, autopct='%1.1f%%')
        plt.title('Evaluation accuracy')
        plt.savefig(fname='plots\\evaluation_accuracy.png')
        plt.clf()

    def plot_top5(self):
        name = str(self.name)
        if name != 'baseline':
            name += '_' + str(self.weights.get(self.name))
        file_path = 'plots\\top5_plots\\' + name + '.png'

        names = ['Top 1', 'Top 3', 'Top 5']
        values = [
            self.correct_top1,
            self.correct_top3,
            self.correct_top5,
        ]
        heights = []
        for value in values:
            heights.append(value / self.total_evaluated)
        x_coordinates = np.arange(len(names))
        y_coordinates = np.arange(0, 1.1, 0.1)
        y_labels = ['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%']

        bar = plt.bar(x=x_coordinates, height=heights, align='center')
        for rect in bar:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d%%' % round(height*100, 1), ha='center', va='bottom')
        plt.xticks(ticks=x_coordinates, labels=names)
        plt.yticks(ticks=y_coordinates, labels=y_labels)
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.title(name)
        plt.savefig(file_path)
        plt.clf()
