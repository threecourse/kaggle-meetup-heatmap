## Heatmap with change-cell-size feature

### Motivation
Heatmap is a great chart to catch information in matrix.
However, I sometimes want to add more information in cases like this:
* matrix is like [product A-F] x [area a-f]
* want to see [profitability] and [sales volume] of each cell.

In these cases, by seeing their volumes, we can know their importance and statistical reliability.
I tried to add an feature to change size of each cell in heatmap for that purpose.

### Implementation

I started with copying codes from heatmap in seaborn.matrix.py.
After that, I modified points below:
* draw each rectangle by add_patch instead of pcolormesh
* added and changed some arguments and their usage
    * I tried to keep interfaces and structure as much as possible, but had to make some changes on them.

Below is the code that contains modified heatmap class and function.   (It's too long so I hide it)

### Kaggle Kernel Example
https://www.kaggle.com/threecourse/heatmap-with-change-cell-size-feature/notebook