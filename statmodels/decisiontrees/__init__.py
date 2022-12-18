## imports ##
import os
import sys

## set path to package ##
sys.path.append(os.getcwd()+'/decisiontrees')

## import relevant functionality ##
from statmodels.decisiontrees.treeclassifier import *
from statmodels.decisiontrees.treeregressor import *
from statmodels.decisiontrees.decisiontreeregressor import *