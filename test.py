import csv
filename = 'freqrnc2011.csv'
filename_2 = 'freqdict.csv'
freqdict = []

with open(filename, newline= '', encoding= 'utf-8') as freqfile:
    freqreader = csv.reader(freqfile, delimiter= '	')
    for row in freqreader:
        word, pos, ipm = row[0], row[1], float(row[2])
        freqdict.append((ipm, word, pos))

freqdict.sort()

with open(filename, 'w', newline='', encoding= 'utf-8') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter='	',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for ipm, word, pos in reversed(freqdict):
        spamwriter.writerow([word, pos, ipm])

