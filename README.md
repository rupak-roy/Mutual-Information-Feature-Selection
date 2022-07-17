#Mutual Information Feature Selection

Feature selection using mutual information mi scores from information theory 

Mutual Information Feature Selection
Mutual information from the field of information theory is the application of information gain (typically used in the construction of decision trees) to feature selection.
Mutual information is calculated between two variables and measures the reduction in uncertainty for one variable given a known value of the other variable.
Mutual information is straightforward when considering the distribution of two discrete (categorical or ordinal) variables, such as categorical input and categorical output data. Nevertheless, it can be adapted for use with numerical input and output data. Mutual Information measures the entropy drops under the condition of the target value. 
Simple explanation to this concept is this formula:
MI(feature;target) = Entropy(feature) - Entropy(feature|target)
The MI score will fall in the range from 0 to ∞. 
The high value of Mi means a closer connection between the feature and the target indicating feature importance for training the model. However, the lower the MI score like 0 indicates a weak connection between the feature and the target.

Tutorial: https://bobrupakroy.medium.com/8eb19071664b?source=friends_link&sk=acda45060f35ec47f74c84945fcb6b66
