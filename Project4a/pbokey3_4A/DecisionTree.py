from math import log
import sys
from scipy.stats import chisqprob

class Node:
  """
  A simple node class to build our tree with. It has the following:
  
  children (dictionary<str,Node>): A mapping from attribute value to a child node
  attr (str): The name of the attribute this node classifies by. 
  islead (boolean): whether this is a leaf. False.
  """
  
  def __init__(self,attr):
    self.children = {}
    self.attr = attr
    self.isleaf = False

class LeafNode(Node):
    """
    A basic extension of the Node class with just a value.
    
    value (str): Since this is a leaf node, a final value for the label.
    islead (boolean): whether this is a leaf. True.
    """
    def __init__(self,value):
        self.value = value
        self.isleaf = True
    
class Tree:
  """
  A generic tree implementation with which to implement decision tree learning.
  Stores the root Node and nothing more. A nice printing method is provided, and
  the function to classify values is left to fill in.
  """
  def __init__(self, root=None):
    self.root = root

  def prettyPrint(self):
    print str(self)
    
  def preorder(self,depth,node):
    if node is None:
      return '|---'*depth+str(None)+'\n'
    if node.isleaf:
      return '|---'*depth+str(node.value)+'\n'
    string = ''
    for val in node.children.keys():
      childStr = '|---'*depth
      childStr += '%s = %s'%(str(node.attr),str(val))
      string+=str(childStr)+"\n"+self.preorder(depth+1, node.children[val])
    return string    

  def count(self,node=None):
    if node is None:
      node = self.root
    if node.isleaf:
      return 1
    count = 1
    for child in node.children.values():
      if child is not None:
        count+= self.count(child)
    return count  

  def __str__(self):
    return self.preorder(0, self.root)
  
  def classify(self, classificationData):
    """
    Uses the classification tree with the passed in classificationData.`
    
    Args:
        classificationData (dictionary<string,string>): dictionary of attribute values
    Returns:
        str
        The classification made with this tree.
    """
    node = self.root
    if node.isleaf != True:
        while node.isleaf == False:
            if node.attr != None:
                node = node.children[classificationData[node.attr]]
                if node == None:
                    return None
        return node.value
    else:
        return node.value
  
def getPertinentExamples(examples,attrName,attrValue):
    """
    Helper function to get a subset of a set of examples for a particular assignment 
    of a single attribute. That is, this gets the list of examples that have the value 
    attrValue for the attribute with the name attrName.
    
    Args:
        examples (list<dictionary<str,str>>): list of examples
        attrName (str): the name of the attribute to get counts for
        attrValue (str): a value of the attribute
    Returns:
        list<dictionary<str,str>>
        The new list of examples.
    """
    ex = []

    for e in examples:
        if e[attrName] != attrValue:
            continue
        else:
            ex.append(e)

    return ex
  
def getClassCounts(examples,className):
    """
    Helper function to get a dictionary of counts of different class values
    in a set of examples. That is, this returns a dictionary where each key 
    in the list corresponds to a possible value of the class and the value
    at that key corresponds to how many times that value of the class 
    occurs.
    
    Args:
        examples (list<dictionary<str,str>>): list of examples
        className (str): the name of the class
    Returns:
        dictionary<string,int>
        This is a dictionary that for each value of the class has the count
        of that class value in the examples. That is, it maps the class value
        to its count.
    """
    counts_class = {}
    
    for e in examples:
        class_val = e[className]
        if class_val not in counts_class:
            counts_class[class_val] = 0
        counts_class[class_val] += 1

    return counts_class

def getMostCommonClass(examples,className):
    """
    A freebie function useful later in makeSubtrees. Gets the most common class
    in the examples. See parameters in getClassCounts.
    """
    counts = getClassCounts(examples,className)
    if len(examples) > 0:
        return max(counts, key=counts.get)
    else:
        return None

def getAttributeCounts(examples,attrName,attrValues,className):
    """
    Helper function to get a dictionary of counts of different class values
    corresponding to every possible assignment of the passed in attribute. 
	  That is, this returns a dictionary of dictionaries, where each key  
	  corresponds to a possible value of the attribute named attrName and holds
 	  the counts of different class values for the subset of the examples
 	  that have that assignment of that attribute.
    
    Args:
        examples (list<dictionary<str,str>>): list of examples
        attrName (str): the name of the attribute to get counts for
        attrValues (list<str>): list of possible values for the attribute
        className (str): the name of the class
    Returns:
        dictionary<str,dictionary<str,int>>
        This is a dictionary that for each value of the attribute has a
        dictionary from class values to class counts, as in getClassCounts
    """
    attr_count = {}
    for val in attrValues:
        pert_ex = getPertinentExamples(examples, attrName, val)
        attr_count[val] = getClassCounts(pert_ex, className)
    return attr_count
        

def setEntropy(classCounts):
    """
    Calculates the set entropy value for the given list of class counts.
    This is called H in the book. Note that our labels are not binary,
    so the equations in the book need to be modified accordingly. Note
    that H is written in terms of B, and B is written with the assumption 
    of a binary value. B can easily be modified for a non binary class
    by writing it as a summation over a list of ratios, which is what
    you need to implement.
    
    Args:
        classCounts (list<int>): list of counts of each class value
    Returns:
        float
        The set entropy score of this list of class value counts.
    """
    
    e = 0
    for count in classCounts:
        count = float(count)
        total_class_counts = sum(classCounts)
        count = count / total_class_counts
        tot = -(count * log(count, 2))
        e += tot
    return e

   

def remainder(examples,attrName,attrValues,className):
    """
    Calculates the remainder value for given attribute and set of examples.
    See the book for the meaning of the remainder in the context of info 
    gain.
    
    Args:
        examples (list<dictionary<str,str>>): list of examples
        attrName (str): the name of the attribute to get remainder for
        attrValues (list<string>): list of possible values for attribute
        className (str): the name of the class
    Returns:
        float
        The remainder score of this value assignment of the attribute.
    """
    class_counts = getClassCounts(examples, className).values()
    total_class_counts = sum(class_counts)
    result = 0
    for value in attrValues:
        pert_ex = getPertinentExamples(examples,attrName,value)
        kclasscounts = getClassCounts(pert_ex, className)
        tots = float(sum(kclasscounts.values()))
        interm_tots = tots / total_class_counts 
        result += interm_tots * setEntropy(kclasscounts.values())
    return result
          
def infoGain(examples,attrName,attrValues,className):
    """
    Calculates the info gain value for given attribute and set of examples.
    See the book for the equation - it's a combination of setEntropy and
    remainder (setEntropy replaces B as it is used in the book).
    
    Args:
        examples (list<dictionary<str,str>>): list of examples
        attrName (str): the name of the attribute to get remainder for
        attrValues (list<string>): list of possible values for attribute
        className (str): the name of the class
    Returns:
        float
        The gain score of this value assignment of the attribute.
    """
    rem = remainder(examples,attrName,attrValues,className)
    ent = setEntropy(list(getClassCounts(examples, className).values()))
    return ent - rem
  
def giniIndex(classCounts):
    """
    Calculates the gini value for the given list of class counts.
    See equation in instructions.
    
    Args:
        classCounts (list<int>): list of counts of each class value
    Returns:
        float
        The gini score of this list of class value counts.
    """
    res = 1.0
    s = sum(classCounts)
    for cl in classCounts:
        quout = (float(cl) / s)
        neg = -pow(quout, 2)
        res += neg
    return res
  
def giniGain(examples,attrName,attrValues,className):
    """
    Return the inverse of the giniD function described in the instructions.
    The inverse is returned so as to have the highest value correspond 
    to the highest information gain as in entropyGain. If the sum is 0,
    return sys.maxint.
    
    Args:
        examples (list<dictionary<str,str>>): list of examples
        attrName (str): the name of the attribute to get counts for
        attrValues (list<string>): list of possible values for attribute
        className (str): the name of the class
    Returns:
        float
        The summed gini index score of this list of class value counts.
    """
    res = 0
    for attrValue in attrValues:
        ex = getPertinentExamples(examples, attrName, attrValue)
        classCounts = getClassCounts(ex, className)
        res += float(len(ex)) / len(examples) * giniIndex(classCounts.values())

    if res == 0:
        return sys.maxint
    else:
        return 1.0 / res
    
def makeTree(examples, attrValues,className,setScoreFunc,gainFunc):
    """
    Creates the classification tree for the given examples. Note that this is implemented - you
    just need to imeplement makeSubtrees.
    
    Args:
        examples (list<dictionary<str,str>>): list of examples
        attrValues (dictionary<string,list<string>>): list of possible values for attribute
        className (str): the name of the class
        classScoreFunc (func): the function to score classes (ie setEntropy or giniIndex)
        gainFunc (func): the function to score gain of attributes (ie infoGain or giniGain)
    Returns:
        Tree
        The classification tree for this set of examples
    """
    rem_attr = attrValues.keys()
    return Tree(makeSubtrees(rem_attr,examples,attrValues,className,getMostCommonClass(examples,className),setScoreFunc,gainFunc))
    
def makeSubtrees(remainingAttributes,examples,attributeValues,className,defaultLabel,setScoreFunc,gainFunc):
    """
    Creates a classification tree Node and all its children. This returns a Node, which is the root
    Node of the tree constructed from the passed in parameters. This should be implemented recursively,
    and handle base cases for zero examples or remainingAttributes as covered in the book.    

    Args:
        remainingAttributes (list<string>): the names of attributes still not used
        examples (list<dictionary<str,str>>): list of examples
        attrValues (dictionary<string,list<string>>): list of possible values for attribute
        className (str): the name of the class
        defaultLabel (string): the default label
        setScoreFunc (func): the function to score classes (ie setEntropy or giniIndex)
        gainFunc (func): the function to score gain of attributes (ie infoGain or giniGain)
    Returns:
        Node or LeafNode
        The classification tree node optimal for the remaining set of attributes.
    """
    if len(examples) == 0:
        return LeafNode(defaultLabel)
    first_val = examples[0][className]
    if len(examples) == 1:
        return LeafNode(first_val)

    boo = True
    for e in examples:
        if not e[className] == first_val:
            boo = False
            break
    if boo:
        return LeafNode(first_val)
    if len(remainingAttributes) == 0:
        return LeafNode(getMostCommonClass(examples, className))

    argmax = remainingAttributes[0]
    max_func = gainFunc(examples, argmax, attributeValues[argmax], className)
    for attr in remainingAttributes:
        func = gainFunc(examples, attr, attributeValues[attr], className)
        if func <= max_func:
            continue
        else:
            argmax = attr
            max_func = func

    curr_attr = list(remainingAttributes)
    curr_attr.remove(argmax)
    root = Node(argmax)
    for v in attributeValues[argmax]:
        ex = getPertinentExamples(examples, argmax, v)
        defaultLabel = getMostCommonClass(examples, className)
        root.children[v] = makeSubtrees(curr_attr, ex, attributeValues, className, defaultLabel, setScoreFunc, gainFunc)
    return root



def makePrunedTree(examples, attrValues,className,setScoreFunc,gainFunc,q):
    """
    Creates the classification tree for the given examples. Note that this is implemented - you
    just need to imeplement makeSubtrees.
    
    Args:
        examples (list<dictionary<str,str>>): list of examples
        attrValues (dictionary<string,list<string>>): list of possible values for attribute
        className (str): the name of the class
        classScoreFunc (func): the function to score classes (ie setEntropy or giniIndex)
        gainFunc (func): the function to score gain of attributes (ie infoGain or giniGain)
        q (float): the Chi-Squared pruning parameter
    Returns:
        Tree
        The classification tree for this set of examples
    """
    remainingAttributes=attrValues.keys()
    return Tree(makePrunedSubtrees(remainingAttributes,examples,attrValues,className,getMostCommonClass(examples,className),setScoreFunc,gainFunc,q))
    
def makePrunedSubtrees(remainingAttributes,examples,attributeValues,className,defaultLabel,setScoreFunc,gainFunc,q):
    """
    Creates a classification tree Node and all its children. This returns a Node, which is the root
    Node of the tree constructed from the passed in parameters. This should be implemented recursively,
    and handle base cases for zero examples or remainingAttributes as covered in the book.    

    Args:
        remainingAttributes (list<string>): the names of attributes still not used
        examples (list<dictionary<str,str>>): list of examples
        attrValues (dictionary<string,list<string>>): list of possible values for attribute
        className (str): the name of the class
        defaultLabel (string): the default label
        setScoreFunc (func): the function to score classes (ie classEntropy or gini)
        gainFunc (func): the function to score gain of attributes (ie entropyGain or giniGain)
        q (float): the Chi-Squared pruning parameter
    Returns:
        Node or LeafNode
        The classification tree node optimal for the remaining set of attributes.
    """
    if len(examples) == 0:
        node = LeafNode(defaultLabel)
        return node

    #all examples have the same classification
    all_same = True
    chck = examples[0][className]
    for e in examples:
        if e[className] != chck:
            all_same = False
    if all_same == True:
        node = LeafNode(chck)
        return node

    if len(remainingAttributes) == 0:
        most_common = getMostCommonClass(examples, className)
        return LeafNode(most_common)

    maxA = None
    maxGain = -9999999
    for atr in remainingAttributes:
        if gainFunc(examples, atr, attributeValues[atr], className) > maxGain:
            maxGain = gainFunc(examples, atr, attributeValues[atr], className)
            maxA = atr

    # chi-square check
    dic = getAttributeCounts(examples, maxA, attributeValues[maxA], className)
    tracking_dict = {}
    for key in dic.keys():
        sub = 0
        for item in dic[key].keys():
            sub += dic[key][item]
        tracking_dict[key] = sub #class count

    class_count = getClassCounts(examples, className)
    dev = 0
    for key in dic.keys():
        chi_test = 0
        for item in dic[key].keys():
            pi_count = dic[key][item] * 1.0
            phi_median_calc = (class_count[item] / (len(examples) * 1.0))
            phi = phi_median_calc * tracking_dict[key]
            chi_test = chi_test + (pi_count - phi)**2 / phi
        dev = dev + chi_test
    
    v = len(attributeValues[maxA]) - 1

    if chisqprob(dev, v) > q:
        return LeafNode(getMostCommonClass(examples, className))
    
    # add subtree
    n = Node(maxA)
    rem = []
    for atr in remainingAttributes:
        if atr == maxA:
            continue
        else:
            rem.append(atr)

    m_comm = getMostCommonClass(examples,className)
    dic = {}
    for val in attributeValues[maxA]:
        new_ex = getPertinentExamples(examples, maxA, val);
        child_node = makePrunedSubtrees(rem, new_ex, attributeValues, className, m_comm, setScoreFunc, gainFunc, q)
        dic[val] = child_node
    n.children = dic
        
    return n