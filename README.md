# Rumor Detection
Preprocessing.py- We add below user properties to the input:
1) follower count of user
2) friend count of user
3) ratio of followers and friends
4) whether a verify account or not
5) registration time (year)
6) Number of history tweets


*** /nfold_new directory has the new source files with additional features

Model folder in master branchhas the code for pruning the trees based on lenght of the tweets.
Changes include: construct tree function in the file Function_from_original_author.py

## Reference
We develop our code based on the previous work of following github repo:
1. https://github.com/majingCUHK/Rumor_RvNN
2. https://github.com/ShuHwaiTeoh/AI_rumor_detection
