#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 12:16:22 2019

@author: henskyconsulting
"""

import unittest
import numpy as np
import pandas as pd
import logging


logging.basicConfig(filename='matmult.log',filemode='w')

class KripaDef():
    """
    Storage Object for Calculations of Krippandorf Alpha
    """
    def __init__(self):
        self.useaccints = True
        self.maxdim = 10000
        self.usewide = True
        #: maximum number of cells in matrix 
        self.N = -1
        #: cells not available
        self.NA = -1
        # if negative will be set by a scan of the input matrix
        self.minval = -1
        self.maxval = -1
        #: Coincidence matrix
        self.coinmat = None
        #: Numver of coders
        self.codeno = -1
        #: Number of cases to code
        self.caseno = -1
        #: Number of pairs that can be coded
        self.pairno =  -1
    def __repr__(self):
        retval = "KripaDef Storarge Object for Defaults /n"
        retval = "minval = {} maxval = {} /n ".format(self.minval,self.maxval)
        return retval

class MatMult():
    """
    Provide customized control of output
    and error checking for matrix multiplication
    """
    def __init__(self,mata="NA",matb="NA"):
        #: left hand side of matrix multiplication
        self.diag = logging.StreamHandler()
        self.diag.setLevel(logging.INFO)
        self.mata = mata
        #: right hand side of matrix multiplication
        self.matb = matb
        #: number of calcs
        self.calcno = 0
        self.matdim = 1000
        self.retcodes = {}
        self.krip_alpha = np.NaN
        self.kripadef = KripaDef()
        self.loadcodes()
        
    def __del__(self):
        print('object destroyed')
        
    def __repr__(self):
        print('%i operations have been performed',self.calcno )
        ret = 'check mats %s'.format(self.checkmats())
        print('It is ',ret,' that the matrices are ready' )
        
    def loadmata(self,matadata,na_extra='',wide=True,skip=0,by_observation=False,qualval=None):
        """ Loads data in self.mata
            args:
              matadata: path to csv file
              wide: each column is an record
              skipn: drop records where index has less than skipn count
              Qualval: list of legal values
            returns:
              boolean if sucessful
        """
        if by_observation:
            print('*** 17.1')
            print('*** 17.11  self.mata before load type= ',type(self.mata))
            self.mata = self.loadmatbyob(matadata,na_extra,qualval)
            print('*** 17.12 self.mata after load type= ',type(self.mata))
        elif wide and not by_observation:
            self.mata = self.loadmat(matadata,na_extra)
            print('*** 17.13 self.mata after load type=',type(self.mata))
        else:
            self.mata = self.loadmatnarrow(matadata,na_extra)
            print('*** 17.14 self.mata after load type=',type(self.mata))
            
        print('*** 17.2')
        print('coders with less than ',skip,' will be removed')
        print('in loadmata mata has dimensionsion ',self.mata.shape,' before')
        print('   coders with less than ',skip,' observations are removed')
        rowcount = self.mata.notnull().sum(axis=1)


        print('*** 17.3')
        rowkeep = rowcount > skip
        self.mata = self.mata[rowkeep]
        print('after process of skips dimension of self.matat',self.mata.shape)
        return True
    
    def cellvalidatebyob(self,matfile,tag1):
        """ replicates calculation of a single cell in a coincidence matrix
            used for replication purposes only
            args:
                matfile path to input csv file
                first value to validated
                second value to be validated
            returns:
                expected cell value in the coincidence matrix
        """
        testdata_raw = pd.read_csv(matfile,keep_default_na=True,na_values=True)
        print('*** 100 in test shape of test data is ',testdata_raw.shape)
        # subset for taga
        # total should be row total in concidence matrix
        testdata_raw2 = testdata_raw[testdata_raw.tag==tag1]
        print('*** 101 in testdata_raw2 shape',testdata_raw2.shape)
        testdata_raw2.sort_values(by=['tweet'],inplace=True)
        print('*** 100 in cellvalidatebyob should see seven pairs')
        print(testdata_raw2.head(30))
        
        
         
        
    
    def loadmatbyob(self,matfile,na_extra,qualval):
        """ 
        loads data where one record is observation
        should compile to a reliability matrix
        args:
          qualval = None if value data is integer
          qualval = dictionary of valid nonminal values
        """
        print('*** 50 in load matbyob')
        print('*** 50.05 in loadmatbyob qualval=',qualval)
        databyob_raw = pd.read_csv(matfile,keep_default_na=True,na_values=na_extra)
        databyob_raw.sort_values(["tweet","owner"],inplace=True)
        # databyob_nodup3 = databyob_raw.drop_duplicates()
        # databyob_nodup2 = databyob_raw.drop_duplicates(subset=['owner','tweet'])
        databyob3 = databyob_raw.drop_duplicates(inplace=False)
        databyob2 = databyob_raw.drop_duplicates(subset=["tweet","owner"],inplace=False)
        print('*** 50.1 in load matbyob')
        print('*** shape of databyob before duplicates dropped',databyob_raw.shape)
        print('*** shape of databyob after duplicates dropped',databyob3.shape)
        print('*** shape of databyob after duplicates 2 dropped',databyob2.shape)
        databyob = databyob2
            
        print(databyob.head())
        print(databyob.describe())
        coders = databyob.values[:,0]
        units = databyob.values[:,2]
        valuevec = databyob.values[:,1]  
        databyob = databyob_raw 
        print(coders[0:4],units[0:4],valuevec[0:4])
        values = pd.Series(valuevec,dtype="category")
        print(values[0:4])
        unit_lst= list(np.unique(units))
        coder_lst = list(np.unique(coders))
        print('*** 50.2 shape of reliability matrix (',len(coder_lst),',',len(unit_lst),')')
        datamat = np.empty((len(coder_lst),len(unit_lst)))
        datamat[:] = np.nan
        datamat = pd.DataFrame(datamat,index=coder_lst,columns=unit_lst)
        print('*** 50.3 just iniitialized self.mata')
        print(datamat.shape)
        print('*** 50.4 start long loop')
        print('*** there are ',databyob.shape[0],' rows to be processed')
        for row in range(databyob.shape[0]):
            rowdata = databyob.iloc[row,:]
            if qualval is None:
                theval = rowdata.tag
            else:
                try:
                    theval = qualval[rowdata.tag]
                except KeyError:
                    print('at row',row,'could not process tag=',rowdata.tag)
            thecoder = rowdata.owner
            theunit = rowdata.tweet
            datamat.loc[thecoder,theunit] = theval    
        print('*** 50.5')
        print(datamat.shape)
        NumNA = datamat.isna().sum().sum()
        goodvals = datamat.shape[0] * datamat.shape[1] - NumNA
        print('*** 50.6')
        print('number of good values in self.mata=',goodvals)
        print('number of vals in databyob=',databyob.shape[0])
        print('*** 50.7')
        print('return reliability matrix of type',type(datamat))
        print(datamat.shape)
        
        return datamat
   
    def loadmatnarrow(self,matfile,na_extra):
        datamat = pd.read_csv(matfile,keep_default_na=True,na_values=na_extra,index_col=0)
        datamat = datamat.T
        
        return datamat

    
    def loadmat(self,matfile,na_extra):
        """ Use Pandas to read spreadsheet and create reliability matrix
        assume for now it is a csv file
        returns:
            pandas datafrom (not numpy matrix)
        """
        datamat = pd.read_csv(matfile,keep_default_na=True,na_values=na_extra,index_col=0 ) 
        print('*** 1.0 in loadmat')
        print(datamat.describe())
        return datamat
        
        
    def loadcodes(self,langdef = 'en'):
        """ load return codes in dictionary
        args:
            langdef : ISO 639 langauge code
          ::returns::
              true if langdef is valid
       """
        langfnd = False
        if langdef == 'fr':
            self.retcodes['AB compatible'] = True
            self.reccodes['dimension trop grande'] = False
            self.reccodes['langused'] = ('code de langue', langdef)
            langfnd = True
            return langfnd
        elif langdef == 'es':
            self.retcodes['AB compatible'] = True
            self.reccodes['Dimension tambien amplio'] = False
            self.reccodes['Código utilizado'] = langdef
            langfnd = True
            return langfnd
        else:
            if langdef == 'en':
                self.retcodes['langused'] = ('Language Used', 'en')
                langfnd = True
            # if language is not known english will be used
            self.retcodes['AB compatible'] = True
            self.retcodes['Dimension too Large'] = False
            self.retcodes['matna'] = ('Matrix Defined',False)
            self.retcodes['langused'] = ('Language Code Used', 'en')
            return langfnd
    def checkmats(self):
        if type(self.mata) == str or type(self.matb) == str:
            print('matrices not loaded')
            return False
        if self.mata.shape[1] != self.matb.shape[0]:
            print('incompatible dimensions')
            print('*** 1')
            print(self.mata.shape)
            print(self.matb.shape)
            return False
        return True
    
    def summat(self,themat):
        """ brief summary of the mat
        args:
            themat
        results:
            str_summary
        """
        retstr = ""
        retstr += "The shape is " + str(themat.shape[0]) 
        retstr += " rows and " + str(themat.shape[1]) + " columns \n"
        
        N_tot = themat.count().sum()
        N_na = themat.isna().sum().sum()
        M_min = themat.min().min()
        M_max = themat.max().max()
        
        retstr += "for a sample of " + str(N_tot) + " of which " 
        retstr += str(N_na) + " are not available \n" 
        retstr += "The values range from " + str(M_min) + " to " + str(M_max)
    
        return retstr
    
    def numpairs(self,ellist):
        """
        returns and integer representing the level of agreement given 
        in a column and returns a dictionary of potential
        args:
            list of coder responses
        returns:
            dictionary of additions to coincidence matrix
        """
        # strip out nans
        ellist2 = [x for x in ellist if not np.isnan(x)]
        count = len(ellist2) - 1
        
        return max(count,1)
      
    def calcmult(self,mata="NA",matb="NA"):
        """ Performn matrix multiplication
        """
        if(not self.checkmats()):
            return False
        return np.matmul(self.mata,self.matb)
    
    
    
    def kripa_from_wide(self,themat,coder,valrng=np.nan):
        """ calculate krippendorf alpha statistic
        args:
            themat pandas dataframe indexed on coder
            coder value in themat that represents coding unit
            value value generated by coding process
        results:
            float the test statistic
        """
        # See usage note in Docs on kripa
        self.kripadef.N = themat.count().sum()
        self.kripadef.NA = themat.isna().sum().sum()
        print(themat.iloc[1:5])
        if np.isnan(valrng): 
            self.kripadef.minval = themat.min().min()
            self.kripadef.maxval = themat.max().max()
        else:
            self.kripadef.minval = valrng[0]
            self.kripadef.maxval = valrng[1]
            
        self.kripadef.codeno = themat.shape[0]
        #: Number of cases to code
        self.kripadef.caseno = themat.shape[1]

        
        
        # assuming integer ratings for now
        try:
            coindef = int(self.kripadef.maxval - self.kripadef.minval + 1)
        except ValueError:
            print('invalid minimum and maximum values')
            print('self.kripadef.maxval=',self.kripadef.maxval)
            print('self.kripadef.maxval=',self.kripadef.minval)
            return np.NaN
        if(coindef < 0):
            print('error coindef=',coindef)
            print('valrng =',valrng)
        # will calculate kripadef by then
        self.kripadef.coinmat = np.zeros((coindef,coindef),dtype=int)
        # need local matrix for by question calculations
        coinquest = np.zeros((coindef,coindef),dtype=int)
        # in wide format the coders are the number of rows
        # the cols are the questions are being coded
        # agreement and disagreement is defined as equality
        # of values in the same column
        for colval in range(themat.shape[1]):
            # zero array for the current quesiion
            coinquest[:,:] = 0
            # will providing check for estimates of total pairs
            paircount = 0
            for coder1  in range(themat.shape[0]):
                # calculate over the questions  
                curval1 = themat.iloc[coder1,colval]
                if  pd.isna(curval1) :
                    continue
                for coder2 in range(coder1+1,themat.shape[0]):      
                    curval2 = themat.iloc[coder2,colval]
                    if pd.isna(curval2) :
                        continue
                    paircount += 1
                    curval2 = int(curval2)
                    coin_row = int(curval1-1)
                    coin_col = int(curval2-1)
                    # note that diagnonals should be incremented twice
                    coinquest[coin_row,coin_col] += 1
                    coinquest[coin_col,coin_row] += 1 
                    
                    # regardless of agreement or disagreement
            # use coinvec to augment coinmat
            # need to devide coinquest for the approbriate value
            for coincol in range(coindef):
                paircount = self.numpairs(themat.iloc[:,colval])
                if paircount == 0:
                    continue
                coinquest[:,coincol] = coinquest[:,coincol]/paircount
            coinquestint = coinquest.astype(int)
            self.kripadef.coinmat += coinquestint
                        
        self.kripadef.pairno = self.kripadef.coinmat.sum().sum()         

                
    def krip_cal(self):
        """ calculate Krippendoffs alpha from the conincende matrix
        
            returns bool indicating success
        """

        print('*** 7.0 krip_cal start coinmat \n',self.kripadef.coinmat)
        try:
            coindef = int(self.kripadef.maxval - self.kripadef.minval + 1)
        except ValueError:
            print('error in kripcal kripadef= \n')
            print(self.kripadef.coinmat)
            return False
            
        totdisagg = 0
        # sum up the elements of the off-diagonal 
        for rowel in range(0,coindef-1):
            for colel in range(rowel+1,coindef):
                totdisagg += self.kripadef.coinmat[rowel,colel]
        totpos = 0
        for colel1 in range(0,coindef):
            coltot1 = self.kripadef.coinmat[:,colel1].sum()
            for colel2 in range(colel1+1,coindef):
                coltot2 = self.kripadef.coinmat[:,colel2].sum()
                val = coltot1 * coltot2
                totpos += val

        # error here
        N_df = self.kripadef.pairno - 1
        
        self.krip_alpha = 1 - totdisagg * N_df /totpos
        
        return True
        
    
        
    def kripa_from_long(self,themat,coder,value):
        pass
        
        
    
class Test(unittest.TestCase):
    
    """
    def test_creation_matmult(self):
        print('\n',40*'*')
        print('test_creation_matmult')
        mult1 = MatMult()
        self.assertEqual(type(mult1.retcodes),dict)
        # default should be english
        self.assertEqual(mult1.retcodes['langused'],('Language Code Used','en'))
    def test_checkmat(self):
        print('\n',40*'*')
        print('test_creation_checkmat')
        matI = np.identity(100)
        mat2 = np.random.rand(3,100)
        matout1 = MatMult(matI,mat2)
        self.assertEqual(matout1.checkmats(),False)
    def test_calc(self):
        print('\n',40*'*')
        print('test_calc')
        matI = np.identity(100)
        mat2 = np.random.rand(3,100)
        # should fail as the dimensions are wrong
        matobj1 = MatMult(matI,mat2)
        matout1 = matobj1.calcmult()
        numtest1 = np.allclose(mat2,matout1)
        self.assertEqual(numtest1,False)
        # should succed with corection as the dimensions are wrong
        matobj2 = MatMult(mat2,matI)
        print('*** 1 in test_calc')
        matout2 = matobj2.calcmult()
        print('*** 2 in test_calc')
        numtest1 = np.allclose(mat2,matout2)
        self.assertEqual(numtest1,True)
        print(40*'*')
    """
    """
    def test_kripa(self):
        print('\n',40*'*')
        print('test_kripa')
        kripcalc = MatMult()
        kripcalc.loadmata('./testdata/kripa.csv',na_extra='*')
        thesum = kripcalc.summat(kripcalc.mata)
        print(kripcalc.mata)
        print(thesum)
        kripcalc.kripa_from_wide(kripcalc.mata,coder='Units u')
        
        retval = kripcalc.krip_cal()
        self.assertEqual(retval,True)
        self.assertEqual(np.around(kripcalc.krip_alpha,3),0.691)
    """  
    """
    def test_kripa_wide(self):
        print('\n',40*'*')
        print('test_kripa')
        kripcalc = MatMult()
        kripcalc.loadmata('./testdata/kripa_T.csv',na_extra='*',wide=False)
        thesum = kripcalc.summat(kripcalc.mata)
        print(kripcalc.mata)
        print(thesum)
        kripcalc.kripa_from_wide(kripcalc.mata,coder='Units u')
        
        retval = kripcalc.krip_cal()
        self.assertEqual(retval,True)
        self.assertEqual(np.around(kripcalc.krip_alpha,3),0.691)   
    
    """
    """
    def test_kripa_wide_plus1(self):
        print('\n',40*'*')
        print('test_kripa_wide_plus1')
        kripcalc = MatMult()
        kripcalc.loadmata('./testdata/kripa_T_plus1.csv',na_extra='*',wide=False)
        thesum = kripcalc.summat(kripcalc.mata)
        print(kripcalc.mata)
        print(thesum)
        kripcalc.kripa_from_wide(kripcalc.mata,coder='Units u')
        
        retval = kripcalc.krip_cal()
        self.assertEqual(retval,True)
        self.assertEqual(np.around(kripcalc.krip_alpha,3),0.691)     
    """
    def test_kripa_wide_plus1_withskip(self):
        print('\n',40*'*')
        print('test_kripa_wide_plus1')
        kripcalc = MatMult()
        kripcalc.loadmata('./testdata/kripa_T_plus1.csv',na_extra='*',wide=False,skip=1)
        thesum = kripcalc.summat(kripcalc.mata)
        print(kripcalc.mata)
        print(thesum)
        kripcalc.kripa_from_wide(kripcalc.mata,coder='Units u')
        
        retval = kripcalc.krip_cal()
        self.assertEqual(retval,True)
        self.assertEqual(np.around(kripcalc.krip_alpha,3),0.691)       
    
    """
    def test_small_tot_dis_agg(self):
        print('\n',40*'*')
        print('test_kripa')
        kripcalc = MatMult()
        kripcalc.loadmata('./testdata/small_tot_dis_agg.csv',na_extra='*')
        thesum = kripcalc.summat(kripcalc.mata)
        print(kripcalc.mata)
        print(thesum)
        kripcalc.kripa_from_wide(kripcalc.mata,coder='Coder')

        retval = kripcalc.krip_cal()
        self.assertEqual(retval,True)
        self.assertEqual(kripcalc.krip_alpha,0.0)
        
    def test_small_tot_agg(self):
        print('\n',40*'*')
        print('test_kripa')
        kripcalc = MatMult()
        kripcalc.loadmata('./testdata/small_tot_agg.csv',na_extra='*')
        thesum = kripcalc.summat(kripcalc.mata)
        print(kripcalc.mata)
        print(thesum)
        kripcalc.kripa_from_wide(kripcalc.mata,coder='Units u')

        retval = kripcalc.krip_cal()
        self.assertEqual(retval,True)
        self.assertEqual(kripcalc.krip_alpha,1.0)        
    """
    """
    def test_small_tot_agg(self):
        print('\n',40*'*')
        print('test_small_tot_agg')
        kripcalc = MatMult()
        kripcalc.loadmata('./testdata/small_cell34_det.csv',na_extra='*',)
        thesum = kripcalc.summat(kripcalc.mata)
        print(kripcalc.mata)
        print(thesum)
        kripcalc.kripa_from_wide(kripcalc.mata,coder='Units u',valrng=(1,4))

        retval = kripcalc.krip_cal()
        self.assertEqual(retval,True)
        self.assertEqual(kripcalc.krip_alpha,1.0)
    """
    """
    def test_numpairs(self):
        print('\n',40*'*')
        print('test_numpairs')
        kripcalc = MatMult()
        kripcalc.loadmata('./testdata/kripa.csv',na_extra='*',)
        print(kripcalc.mata)
        
        # test come from wiki artilce 
        # note that column numbers offset by one
        ret0 = kripcalc.numpairs(kripcalc.mata.iloc[:,0])
        self.assertEqual(ret0,1)
        
        ret1 = kripcalc.numpairs(kripcalc.mata.iloc[:,1])
        self.assertEqual(ret1,1)

        ret2 = kripcalc.numpairs(kripcalc.mata.iloc[:,2])
        self.assertEqual(ret2,1) 
        
        ret5 = kripcalc.numpairs(kripcalc.mata.iloc[:,5])
        self.assertEqual(ret5,2)  
        
        ret6 = kripcalc.numpairs(kripcalc.mata.iloc[:,6])
        self.assertEqual(ret6,2)
     """   
if __name__ == '__main__':
    unittest.main()               
        
        
        