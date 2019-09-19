Maya Zoe Shanmugam
—————————
I have adhered to the Honor Code in this assignment 

I didn’t really get the chance to finish this. Overall, this was challenging because I haven’t programmed in a while and am essentially learning python. Otherwise, I think this would’ve been fairly straightforward!That being said, this assignment was fun and I enjoyed learning about knn. 

Since I ran out of time, I was unable to create the confusion matrix output file, so I calculate accuracy using the getAccuracy function. 

I recently ONLY updated my answers to question 2, which was previously blank. All other files untouched. 

P.S. I had to upload files manually, as my git ssh key failed. I will come to your office soon to figure this out. 

—————QUESTIONS————————

1. Random Seed: 23352
   Percent Training: 0.64

Accuracies:
- iris.csv: 98.13% 
- mnist_100.csv: 86.67%
- mnist_1000.csv: 93.97%
- monks1.csv: 100%

2. 

- iris.csv: [.955,1.0076]
- mnist_100.csv: [.7951,.9382]
- mnist_1000.csv: [.8930,.9864]
- monks1.csv: 1

3. My accuracies differed by six percent, with the mnist_1000.csv having a higher accuracy of 93.97%. I think this is a result of the mnist_1000.csv having a significantly larger data set.

4.
file:  iris.csv
k = 15: 98.15%
K = 45: 90.74%
K = 90: 48.15%
I think the accuracy drastically changed because of the size of the data set. Remember that iris has only 150 rows. So the first k is 10% of the set, the second is 30%, and the last is 60%. Thus, we are drastically affecting the accuracy values as we are allowing more and more neighbors to be included. If the majority of the dataset is included as neighbors, then naturally we will end up with a lower accuracy. 
 
