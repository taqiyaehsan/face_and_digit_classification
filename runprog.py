import os 

for i in range(10): 
    print(i+1)
    os.system('python dataClassifier.py -c naiveBayes -d faces -t 451')