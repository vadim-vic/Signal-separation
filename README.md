# # Signal reconstruction

During inventory in a densely packed stock, multiple radio-frequency data transmitters often interfere with each other, leading to signal collisions. This reduces the efficiency of inventory. We present a method to resolve these collisions. The good news: despite these collisions, the items can still be identified, and their signals can be reconstructed. This advancement greatly enhances the performance of radio-frequency identification (RFID) systems.

\# RFID, I/Q data,  Aloha collision, Signal separation, Self-modeling regression



The Aloha protocol resolves a mixture of replies: the inventory time-segment splits into time-slots.  At request, each tag waits a random number of time slots and replies. 
Due to this randomness, some  time slots are left  unoccupied, some time slots keep a single tag ID, but some are still occupied by several tags.  Can we avoid collisions in one inventory cycle? The problem of estimation of probability that two tags hit one slot is called [the birthday paradox](https://dialnet.unirioja.es/descarga/articulo/5997063.pdf). What is the probability that two people have their birthdays on the same day? One tag hits any of $`D`$ slots with the probability of $`\frac{1}{D}`$. Two tags do not hit the same slot with the probability $`1-\frac{1}{D}`$. The third tag cannot hit both occupied slots, so the probability is\
```math
\frac{D-1}{D} \frac{D-2}{D} = \left(1-\frac{1}{D}\right) \left(1- \frac{2}{D}\right).
```
So for given~$`D`$ slots,  the probability that none of~$`N`$ tags do not collide is\
```math
\frac{D!}{D^N(D-N)!}.
```
Figure~\ref{fig:pr_collision-free} shows that the probability of successful inventory is small for any reasonable number of tags. So if the shopping cart has over 100 items with tags, most likely there is a collision even for a long inventory cycle. See the green and red lines. 







The Independent Component Analysis is used for signal separation, the challenge is the signal signal receiver2

1. [The Aloha RFID collision detection classifier model description](latex/CollisionDetector.pdf), Feb 7
2. [Two-class Aloha collision detection with RBF and Logistic Regression](ipynb/AlohaCollisionDetector2class_Feb7.ipynb), Feb 7
3. [Plot the probability of birthdays' collision](ipynb/1_Plot_Birthday_Probability_NQ.ipynb)<!-- for no birthday, one, two, and three or more birthdays on the same day-->, Feb 5+
4. [Find the clusters and their centroids in the signal collection](/ipynb/9_Distance_to_6bit.ipynb), Feb 2
5. [Analyze the dimensionality of the span of basis signals](/ipynb/10_SingularValuesDecomposition.ipynb), Feb 9
6. [New data generation procedure to reconstruct the mixed signals](/ipynb/11_GetData_FindTheBasis.ipynb), Feb 19
   
## Examples
1. Import functions from files in the Goole Disc to the Google Colab: [example_utility.ipynb](examples/example_utility.ipynb), [example_utility.py](examples/example_utility.py)
2. [Collect indices of the cartesian product of 1, ..., C sets](examples/16_Example_Cartesian_UpToC.ipynb)
