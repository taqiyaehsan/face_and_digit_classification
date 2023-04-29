import os 

classifier = "perceptron"
dataSet = ('digits', 5000)
# dataSet = ('faces', 450)
outputFile = 'PerceptronDigits.txt'

open(outputFile, 'w').close()

for percent in (x * 0.1 for x in range(1, 11)):
    for i in range(10): 
        os.system('python dataClassifier.py -c ' + classifier + ' -d ' + dataSet[0]+' -t ' + str(int(dataSet[1]*percent)) + ' >> ' + outputFile)
print("DONE")
