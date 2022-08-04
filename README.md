# CS89_VariantEffectPrediction

Final Project for CSCI-S89, Introduction to Deep Learning

Harvard Summer School 2022

Kelly Brock

Please note: these toy models were built for training purposes only, and DEFINITELY should not be used for clinical application! Please refer to <https://evemodel.org/> for an example of a more rigorous computational predictor.

Link to Youtube presentation: <https://youtu.be/qcD5TAge05U>

For a detailed set of installation instructions and details about implementation, please refer to the write-up document in the top level of this repository:

GeneticVariantEffectPrediction_KellyBrock_CS89_Writeup.pdf

Abstract: Genetic disorders can involve changes to the coding region of our DNA - or in other words, the part of our genome that can be transcribed and translated to form proteins, which are the building blocks of cells. In particular, missense mutations (where a change in the DNA nucleotide(s) leads to a change in amino acids in the corresponding protein sequence) are of special interest, because they can have a more ambiguous effect on the encoded protein than other types of mutations like premature truncations. Although the scientific community has identified thousands of these single-amino acid changes that are thought to be pathogenic (contributing to disease), genetics studies can be time-consuming and statistically underpowered, particularly in the case of rare variants. Most genetic variants found in humans remain of unknown significance. To fill this gap, a growing wealth of computational methods aim to predict whether certain missense variants are either pathogenic or benign in the context of human disease. For this case study in biology, I built 3 separate neural net architectures that take as input increasingly more information about protein sequence and the missense mutation of interest, and output a classification for that missense mutant as either pathogenic or benign. A particular challenge was how to split our available data for training and testing, so I also experimented with different split methods as well. Across the three methods, I obtained a validation accuracy of up to 80%. I then used these architectures to generate predictions for all possible missense mutations for the KCNQ1 gene, which encodes a potassium channel implicated in congenital heart arrhythmia disorder called long QT syndrome.
