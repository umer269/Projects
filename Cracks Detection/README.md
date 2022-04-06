1) resnet.py: Resnet architecture to detect cracks in solar cells using TensorFlow.

Explanation:

Different kinds of cracks. The size of cracks may range from very small cracks (a few pixels in our case) to large cracks that cover the whole cell. In most cases, the performance of the cell is unaffected by this type of defect, since connectivity of the cracked area is preserved.
Inactive regions are mainly caused by cracks, where a part of the cell to becomes disconnected. This disconnected area does not contribute to the power production. Hence, the cell performance is decreased.
Data Set: Images of solar cells with cracks and inactive regions.
