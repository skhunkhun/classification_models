Sunveer Khunkhun
March 21 2023

<!-- HOW TO RUN CODE -->

    -To run the code type in the command line 'python3 main.py' and everything should run at once.

    * Every time the program is run, it randomizes the Iris data set each time with 120 train samples and 30 test samples. This means that the accuracy will most likely be different each time the program is run.

<!-- K-MEANS IMPLEMENTATION -->

    Sample test run:

        K-Means:
        Iris Setosa accuracy: 100.0%
        Iris Versicolour accuracy: 73.0%
        Iris Virginica accuracy: 60.0%

    I chose N = 20, as I believe it is enough runs to present an accurate model, while keeping the computational resources to a minimum.

<!-- DECISION TREE IMPLEMENTATION -->

    Sample test run:

        Decision Tree:
        Iris Setosa accuracy: 100.0%
        Iris Versicolour accuracy: 91.0%
        Iris Virginica accuracy: 70.0%
    

<!-- DISCUSSION -->

    For both models, the Iris Setosa tree classification was 100% accurate. I believe on most runs, it will be very accurate 
    due to its smaller dimensions, especially in the pedal width. With the K-Means algorithm, this makes it easier to cluster 
    each of the Setosa flowers, and thus predict when a flower is a Setosa. The same ideology can be used for the decision
    tree model, as it will find the most favourable feature (pedal width) for the Sentosa flowers, and classify any flower that meets that criteria as a Sentosa flower, making the accuracy very high.

    The classification of the Iris Versicolor was 73% for the K-Means algorithm and 91% for the decision tree and the 
    classification of the Iris Virginica was 60% for the K-Means algorithm and 70% for the decision tree. 

    I believe this is due to it being harder to distinguish between a Iris Versicolour and a Iris Virginica, as they have similar dimensions, with the Iris Virginica being slightly bigger on average. So this test does heavily rely on the selecting training data, and the test data. On average, the decision tree did perform better with higher accuracies. I believe this is due to the
    nature of an unsupervised clustering algorithm and a supervisied model based on rules and labels. It is hard to cluster data
    that is very similar, so that is why the K-Means algorithm on average was worse at predicting the accuracies of the Iris
    Versicolour and the Iris Virginica. The decision tree takes real data and labels them to the correct species of flower, and
    learns the rules based on the data given, instead of guessing where data should go. 

    Overall, the clustering algorithm is great for data exploration and data with consistent patterns, which is why it does so well
    when trying to classify the Iris Setosa, but it struggles with classifying data that is similar. The decision tree is great for
    data that is labeled, as it can continously learn given more rules, which is why it did better than the K-means model with
    the other 2 flower classifications.