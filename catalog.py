import numpy as np
from astropy.io import fits
import numexpr as ne

class Catalog():
    '''Catalog class to hold input parameters for sources'''
    def __init__(self, filename, columnNames=None):
        self.readFitsFile(filename)
            
    def __str__(self):
        str(self.tableColumnNames)
        
    def readFitsFile(self, filename, dataHDU=1):
        '''Read in catalog from fits file'''
        hdulist = fits.open(filename)
        self.table = hdulist[dataHDU].data
        self.tableColumnNames = hdulist[dataHDU].columns.names
        hdulist.close()

    def selectColumns(self, selectColumns, maxPts=None):
        '''Given a set of columns return a recarray with data

        read only the first maxPts
        '''
        #formats = ['f8' for name in selectColumns]
        #dtype = {'names' : selectColumns, 'formats': formats}

        self.dataColumnNames = [''.join(command) for command in selectColumns]
        if (maxPts == None):
            self.data = np.rec.fromarrays([self.evaluateMath(command) for command in selectColumns],
                                          names=self.dataColumnNames)
        else:
            self.data = np.rec.fromarrays([self.evaluateMath(command)[:maxPts] for command in selectColumns])


    def evaluateMath(self, expression):
        '''Evaluate a mathematical expressions on a set of columns'''
        if (len(expression) == 1):
            return self.table[expression[0]]

        _x = self.table[expression[0]]
        _y = self.table[expression[2]]
        command = "_x %s _y"%expression[1]
        return ne.evaluate(command)
