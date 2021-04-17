# with open('CA.txt') as f:
#     CA_data = [line.split() for line in f]

# for value in CA_data:
#     value[0] = int(float(value[0])*100)
#     if "0.*1j" in value[1]:
#         value[1] = value[1].replace("0.*1j", "0j")
#         value[1] = complex(value[1])
# print(CA_data)

def readExperimentalData(fileName):
    with open(fileName) as f:
        data = [line.split() for line in f]
    objectData = {}
    for value in data:
        value[0] = int(int(float(value[0]) * 1000)/10)
        if "0.*1j" in value[1]:
            value[1] = value[1].replace("0.*1j", "0j")
        if "*1j" in value[1]:
            value[1] = value[1].replace("*1j", "j")
        objectData[value[0]] = value[1]

    return objectData


# with open('Стекло.txt') as f:
#     GlassData = [line.split() for line in f]
# finalRes = {}
# for threeCol in GlassData:
#     [lamda, RealN, ImN] = threeCol
#     lamda = int(int(float(lamda)*10000)/100)
#     finalRes[lamda] = complex(float(RealN), float(ImN)/1000000000)


def readExperimentalDataFromThreeCols(fileName):
    with open(fileName) as f:
        experimentalData = [line.split() for line in f]
        finalRes = {}
        for threeCol in experimentalData:
            [lamda, RealN, ImN] = threeCol
            lamda = int(int(float(lamda)*10000)/100)
            finalRes[lamda] = complex(float(RealN), float(ImN)/1000000000)
    return finalRes
