# Feature-Selection-Methods

This repository contains implementations of two feature selection methods: DRF0 and MRMD.

## DRF0
DRF0 is a distributed feature selection algorithm proposed by V. Bolón-Canedo, N. Sánchez-Maroño, and A. Alonso-Betanzos in their paper "Distributed feature selection: An application to microarray data classification" (Applied Soft Computing, 2015).

In traditional feature selection, a centralized approach is used where a single learning model is employed to solve the problem. However, in today's distributed data environments, it is advantageous to utilize the processing power of multiple subgroups simultaneously. DRF0 allows feature selection in a distributed manner, leveraging the benefits of parallel computing. By dividing the data into subgroups vertically (i.e., according to features), relevant features are selected using a filter-type feature selection algorithm. The selected features are then merged by updating the final feature subgroups based on improvements in classification accuracy.

## MRMD
MRMD is a feature selection method presented in the paper "Feature redundancy term variation for mutual information-based feature selection" by Gao, Wanfu, Liang Hu, and Ping Zhang (Applied Intelligence, 2020).

The conventional approach of minimizing redundancy and maximizing dependency in feature selection may lead to the rejection of features with high redundancy values, even though they could provide valuable information for record classification. MRMD addresses this issue by reducing the redundancy range of a feature while simultaneously maximizing its dependency range using information theory. This method aims to select the most informative features by optimizing the trade-off between redundancy and dependency.

Please refer to the respective papers for detailed explanations of the algorithms and their implementations.

Note: This README provides a brief overview of the feature selection methods in this repository. For instructions on how to use the code and apply these methods to your own data, please refer to the documentation within the repository.
