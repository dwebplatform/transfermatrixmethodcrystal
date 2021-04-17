def readExperimentalDataFromThreeCols(fileName):
    with open(fileName) as f:
    experimentalData = [line.split() for line in f]
    finalRes = {}
    for threeCol in experimentalData:
        [lamda, RealN, ImN] = threeCol
        lamda = int(int(float(lamda)*10000)/100)
        finalRes[lamda] = complex(float(RealN), float(ImN)/1000000000)
    return finalRes
