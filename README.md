Running PPPCA on the promoter task of the GUE dataset (Genome Understanding Evaluation).

The task consists in fitting the PCA on train sequences (unsupervised) and then projecting test sequences into the learned PCA space.

The scores of the train sequences are used to train simple classifiers (e.g. Random Forest, Small MLP) on the reduced feature space.

The performance of these classifiers is then evaluated on the test sequences projected into the same PCA space.