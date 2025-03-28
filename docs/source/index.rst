
#####################
EarlyStopping
#####################



EarlyStopping is a Python library implementing computationally efficient model selection methods.
For iterative estimation procedures applied to statistical learning problems, it is necessary to choose a suitable iteration index to avoid under- and overfitting.
Classical model selection criteria can be prohibitively expensive in high dimensions.
Recently, it has been shown for several regularisation methods that sequential early stopping can achieve statistical and computational efficiency by halting at a data-driven index depending on previous iterates only.




.. tab-set::
    :class: sd-width-content-min

    .. tab-item:: pip

        .. code-block:: bash

            pip install EarlyStoppingPy

    .. tab-item:: development

        .. code-block:: bash

            python3 -m pip install build virtualenv               # Install build tools
            git clone https://github.com/ESFIEP/EarlyStopping.git # Clone git repository
            python3 -m build                                      # Build package
            python3 -m venv myenv                                 # Create virtual environment
            source myenv/bin/activate                             # Activate virtual environment
            python3 -m pip install numpy ipykernel                # Install python packages to the environment
            python3 -m pip install -e .                           # Install the EarlyStopping package in editable mode
            python3 -m ipykernel install --user --name=myenv      # Create Jupyter kernel from the environment




.. image:: tree_heatmaps.gif
   :alt: EarlyStopping Animation
   :align: center
   :width: 100%



.. grid:: 1 1 2 2

    .. grid-item-card::
        :padding: 2
        :columns: 12

        **References**
        ^^^

        - `Early stopping for statistical inverse problems via truncated SVD estimation <https://projecteuclid.org/journals/electronic-journal-of-statistics/volume-12/issue-2/Early-stopping-for-statistical-inverse-problems-via-truncated-SVD-estimation/10.1214/18-EJS1482.full>`_.

          - G. Blanchard, M. Hoffmann, M. Reiß. In *Electronic Journal of Statistics* 12(2): 3204-3231 (2018).

        - `Optimal adaptation for early stopping in statistical inverse problems <https://arxiv.org/abs/1606.07702>`_.

          - G. Blanchard, M. Hoffmann, M. Reiß. In *SIAM/ASA Journal of Uncertainty Quantification* 6(3), 1043–1075 (2018).

        - `Early stopping for L2-boosting in high-dimensional linear models <https://arxiv.org/abs/2210.07850v1>`_.

          - B. Stankewitz. arXiv:2210.07850 [math.ST] (2022).

        - `Estimation and inference of treatment effects with L2-boosting in high-dimensional settings <https://www.sciencedirect.com/science/article/abs/pii/S0304407622000471>`_.

          - J. Kueck, Y. Luo, M. Spindler, Z. Wang. In *Journal of Econometrics* 234(2), 714-731 (2023).

        - `Early stopping for conjugate gradients in statistical inverse problems <https://arxiv.org/pdf/2406.15001>`_.

          - L. Hucker, M. Reiß. arXiv:2406.15001 [math.ST] (2024).


    .. grid-item-card::
        :padding: 2
        :columns: 6


        **Supported Methods**
        ^^^

        - Truncated SVD
        - Landweber Algorithm
        - Conjugate Gradient Descent
        - L2 Boosting
        - Regression Tree (CART)

    .. grid-item-card::
        :padding: 2
        :columns: 6


        .. toctree::
           classes/documentation
           auto_examples/index
           :maxdepth: 1
           :caption: Contents


This research has been partially funded by the Deutsche Forschungsgemeinschaft
(DFG) – Project-ID 318763901 - SFB1294, Project-ID 460867398 - Research Unit
5381 and the German Academic Scholarship Foundation.


