# Group Convolutional Neural Networks
Group Convolutional Neural Networks (G-CNNs) are a generalization of CNNs that reduces the data complexity by exploiting symmetries using Group Theory (you can learn more on the [original paper](https://arxiv.org/abs/1602.07576)).
This type of architecture falls under the category of **Geometric Deep Learning**, which is a subfield of Deep Learning that focuses on the development of algorithms that can learn from data with a known geometric structure.

This method provides **an alternative to data augmentation**, it increases the expressive capacity of the network without increasing the number of parameters by using weight sharing.

In this repository you will see the result of applying G-CNNs on the [Fashion-MNIST](https://arxiv.org/abs/1708.07747) dataset with rotations of multiples of 90°.

#### Example

![Example input](./images/input.png)

![Group CNN final representation](./images/group_cnn_final_representation.png)

![Vanilla CNN final representation](./images/vanilla_cnn_final_representation.png)xw