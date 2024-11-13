from __future__ import annotations
import numpy as np
import pandas as pd
from queue import Queue
import warnings

class RegressionTree:
    """
    A class for constructing a regression tree and computing theoretical quantities.

    **Parameters**

    *design*: ``np.array``. The design matrix for the regression tree.

    *response*: ``np.array``. The response variable values.

    *min_samples_split*: ``int``. The number of samples that the terminal node can have at maximum.

    *true_signal*: ``np.array or None``. Used only in simulation contexts for computing theoretical quantities.

    *true_noise_vector*: ``np.array or None``. Used only in simulation contexts for theoretical quantities.

    **Attributes**

    *sample_size*: ``int``. The number of samples in the data.

    *dimension*: ``int``. The number of variables in the design matrix.

    *regression_tree*: ``Node``. The root node of the regression tree.

    *residuals*: ``np.array``. Stores the mean squared error residuals at each level of the tree.

    *bias2*: ``np.array``. Stores the squared bias values at each level of the tree.

    *variance*: ``np.array``. Stores the variance values at each level of the tree.

    *risk*: ``np.array``. Theoretical risk values based on bias and variance.

    **Methods**

    +-------------------------------------------+--------------------------------------------------------------------+
    | iterate( ``max_depth=None`` )             | Grows a regression tree up to the specified depth.                 |
    +-------------------------------------------+--------------------------------------------------------------------+
    | predict( ``design, depth`` )              | Predicts target values using the regression tree at a given depth. |
    +-------------------------------------------+--------------------------------------------------------------------+
    | get_discrepancy_stop( ``crit, max_depth``)| Finds the first generation satisfying the discrepancy principle.   |
    +-------------------------------------------+--------------------------------------------------------------------+
    | get_balanced_oracle( ``max_depth`` )      | Computes the balanced oracle generation.                           |
    +-------------------------------------------+--------------------------------------------------------------------+
    """

    class Node:
        """
        A class representing a node of the regression tree.
        """
        def __init__(self):
            self.split_threshold = None
            self.variable = None
            self.left_child = None
            self.right_child = None
            self.is_terminal = False
            self.node_prediction = None

        def set_params(self, split_threshold: float, variable: int) -> None:
            """ Set split and variable parameters for this node. """
            self.split_threshold = split_threshold
            self.variable = variable

    def __init__(self,
                 design: np.array,
                 response: np.array,
                 min_samples_split: int,
                 true_signal: np.array = None,
                 true_noise_vector: np.array = None):

        self.true_signal = true_signal
        self.minimal_samples_split = min_samples_split
        self.true_noise_vector = true_noise_vector
        self.design = design
        self.response = response
        self.sample_size = self.design.shape[0]
        self.dimension = self.design.shape[1]



    def iterate(self, max_depth: int = None):
        """
        Grows the regression tree up to the specified depth.

        **Parameters**

        *max_depth*: ``int or None``. Maximum depth to which the tree should grow. If None, the tree grows fully.
        """

        self.residuals = np.array([])
        # For theoretical quantities:
        self.block_matrix = {}
        self.indices_processed = {}
        self.block_matrices_full = {}
        self.bias2 = np.array([])
        self.variance = np.array([])
        self.risk = np.array([])

        # Initialize root node of the tree
        self.regression_tree = self.Node()
        self.queue = Queue()
        self.queue.put((self.regression_tree, np.arange(self.sample_size), 1)) # start with level 1
        self.terminal_indices = {}

        # Process all nodes at the current level (= perform 'one iteration'):
        self.maximal_depth = max_depth
        while not self.queue.empty():
            self._grow_one_iteration()

    def _grow_one_iteration(self):
        """
        Grows one iteration of the regression tree using a breadth-first approach.
        """

        #Initialize:
        level_mse_sum = 0
        level_node_count = 0
        next_level_queue = Queue()
        self.level_indices = []
        current_level_observations = {}

        # Process all nodes at the current level
        for _ in range(self.queue.qsize()):
            self.node, parent_indices, level = self.queue.get()
            number_observations_node = len(self.response[parent_indices])
            self.node.node_prediction = np.mean(self.response[parent_indices])

            # Calculate MSE for the current node and update level MSE
            level_mse_sum += self._impurity(parent_indices) * (number_observations_node / self.sample_size)
            level_node_count += 1

            # Check termination conditions
            terminal_due_to_samples = number_observations_node == self.minimal_samples_split
            terminal_due_to_depth = self.maximal_depth is not None and level >= self.maximal_depth

            # If the terminal condition is satisfied, the node is not added to the next level queue.
            if terminal_due_to_samples or terminal_due_to_depth:
                self.node.is_terminal = True
                continue

            else:
                split_variable, split_value, left_indices, right_indices = self._find_best_split_parent(indices=parent_indices)

                # ILeft and right child node
                self.node.left_child = self.Node()
                self.node.right_child = self.Node()
                self.node.left_child.node_prediction = np.mean(self.response[left_indices])
                self.node.right_child.node_prediction = np.mean(self.response[right_indices])
                self.node.set_params(split_value, split_variable)
                self.level_indices.extend([left_indices, right_indices])

                # Add child nodes to the next level queue
                next_level_queue.put((self.node.left_child, left_indices, level + 1))
                next_level_queue.put((self.node.right_child, right_indices, level + 1))

                # Store observations for each node
                current_level_observations[self.node.left_child] = len(self.response[left_indices])
                current_level_observations[self.node.right_child] = len(self.response[right_indices])

                if len(left_indices) == self.minimal_samples_split:
                    if level not in self.terminal_indices:
                        self.terminal_indices[level] = []
                    self.terminal_indices[level].append(left_indices)

                if len(right_indices) == self.minimal_samples_split:
                    if level not in self.terminal_indices:
                        self.terminal_indices[level] = []
                    self.terminal_indices[level].append(right_indices)


        # Update the residuals, theoretical quantities, and queue for the next level
        self.residuals = np.append(self.residuals, level_mse_sum)
        self._update_theoretical_quantities(current_level_observations, level)
        self.queue = next_level_queue


    def _find_best_split_parent(self, indices):
        """
        Finds the best split based on impurity reduction.

        **Parameters**

        *indices*: ``list``. Indices of samples to be split.
        """
        best_impurity = float("inf")
        best_split = (None, None, [], [])
        range_variables = range(self.dimension)

        # Iterate through the possible variable/split combinations
        for variable in range_variables:
            thresholds = np.unique(self.design[indices, variable])
            for split_threshold in thresholds:
                left_node_indices = [i for i in indices if self.design[i, variable] <= split_threshold]
                right_node_indices = [i for i in indices if self.design[i, variable] > split_threshold]

                if len(left_node_indices) == 0 or len(right_node_indices) == 0:
                    continue
                impurity = (
                            len(self.design[left_node_indices]) * self._impurity(left_node_indices)
                            + len(self.design[right_node_indices]) * self._impurity(right_node_indices)
                           ) / self.sample_size

                # Update the impurity and choice of variable/split
                if impurity < best_impurity:
                    best_impurity = impurity
                    best_split = (variable, split_threshold, np.array(left_node_indices), np.array(right_node_indices))

        return best_split

    def predict(self, design: pd.DataFrame | np.ndarray, depth: int) -> np.array:
        """
        Predicts target values for the given design data using the decision tree at the specified depth.

        **Parameters**

        *design*: ``pd.DataFrame or np.ndarray``. Input design matrix for predictions.

        *depth*: ``int``. Depth level of the tree to use for predictions. If 0, returns the unconditional mean.
        """

        if isinstance(design, pd.DataFrame):
            design = design.to_numpy()

        # Handle depth = 0 case by returning the unconditional mean
        if depth == 0:
            return np.repeat(np.mean(self.response), design.shape[0])
        if depth > self.maximal_depth:
            warnings.warn("Depth can not exceed max_depth. Returning None.", category=UserWarning)
            return None

        predictions = []
        for sample in design:
            current_node = self.regression_tree  # Start at the root
            current_depth = 1

            # Traverse the tree based on stored nodes up to the specified depth
            while not current_node.is_terminal and current_depth <= depth:
                # Find the current node's split criteria based on the level structure
                split_feature = current_node.variable
                split_threshold = current_node.split_threshold

                # Decide which child node to move to based on split criteria
                if sample[split_feature] <= split_threshold:
                    current_node = current_node.left_child
                else:
                    current_node = current_node.right_child

                current_depth += 1

            # When reaching the terminal node or max depth, take the prediction value
            predictions.append(current_node.node_prediction)

        return np.array(predictions)

    def _update_theoretical_quantities(self, current_level_observations, level):
        """
        Updates theoretical quantities (bias, variance, risk) at each level.

        **Parameters**

        *current_level_observations*: ``dict``. Observations at the current tree level.

        *level*: ``int``. Current level of the tree.
        """
        # Exit the function if empty
        if not self.level_indices:
            return

        # Processing of block matrix:
        self._block_matrix_processing(current_level_observations, level)
        if self.true_signal is not None and self.true_noise_vector is not None:
            if self.block_matrix[level].shape[0] == self.sample_size:
                self.new_bias2, self.new_variance = self._get_bias_and_variance(self.indices_processed[level],
                                                                                self.block_matrix[level])
                self.bias2 = np.append(self.bias2, self.new_bias2)
                self.variance = np.append(self.variance, self.new_variance)
                self.new_risk = self.new_bias2 + self.new_variance
                self.risk = np.append(self.risk, self.new_risk)

            else:
                self.new_bias2, self.new_variance = self._get_bias_and_variance(self.indices_complete,
                                                                                self.block_matrices_full[level])
                self.bias2 = np.append(self.bias2, self.new_bias2)
                self.variance = np.append(self.variance, self.new_variance)
                self.new_risk = self.new_bias2 + self.new_variance
                self.risk = np.append(self.risk, self.new_risk)

    def get_discrepancy_stop(self, critical_value, max_depth = None):
        """
         Finds the first generation where the discrepancy principle is met.

         **Parameters**

         *critical_value*: ``float``. Threshold for discrepancy-based stopping.

         *max_depth*: ``int or None``. Maximum depth for the tree if it has not been grown yet.
         """

        # If no iteration done before, grow the tree until max_depth
        if self.residuals.size == 0:
            self.maximal_depth = max_depth
            while not self.queue.empty():
                self._grow_one_iteration()

        if np.any(self.residuals <= critical_value):
            # argmax takes the first instance of True in the true-false array
            early_stopping_index = np.argmax(self.residuals <= critical_value)
            return early_stopping_index
        else:
            warnings.warn("Early stopping index not found. Returning None.", category=UserWarning)
            return None

    def get_balanced_oracle(self, max_depth = None):
        """
        Computes the balanced oracle iteration based on the bias and variance.

        **Parameters**

        *max_depth*: ``int or None``. Maximum depth for the tree if it has not been grown yet.
        """

        # If no iteration done before, grow the tree until max_depth
        if self.residuals.size == 0:
            self.maximal_depth = max_depth
            while not self.queue.empty():
                self._grow_one_iteration()

        if np.any(self.bias2 <= self.variance):
            # argmax takes the first instance of True in the true-false array
            balanced_oracle = np.argmax(self.bias2 <= self.variance)
            return balanced_oracle

        else:
            warnings.warn(
                "Balanced oracle not found. Returning None.", category=UserWarning
            )
            return None

    def _block_matrix_processing(self, current_level_observations, level):
        # Process observations and indices after completing the level
        self.block_matrix[level] = self._process_level_observations(current_level_observations)
        self.indices_processed[level] = np.concatenate(self.level_indices)

        if self.block_matrix[level].shape[0] < self.sample_size:
            indices_pre_append = self.indices_processed[level]
            filtered_indices = {k: v for k, v in self.terminal_indices.items() if k != level}

            # Check if filtered_indices is None or empty
            if not filtered_indices:
                print("No indices to concatenate.")
                return

            self.block_matrices_full[level] = self.append_block_matrix(self.block_matrix[level],
                                                                            filtered_indices)
            indices_append = np.concatenate(
                [idx for level in range(1, level) for idx in self.terminal_indices.get(level, [])])
            self.indices_complete = np.append(indices_pre_append, indices_append)

    def _impurity(self, indices) -> float:
        """
        Computes the mean squared error impurity for the response at the given set of indices.

        **Parameters**

        *indices*: ``list``. Indices of samples to compute impurity on.
        """
        response_node = self.response[indices]
        mse = np.sum((response_node -  np.mean(response_node)) ** 2) / len(response_node)
        return mse

    def append_block_matrix(self, existing_matrix, filtered_indices):
        """
            Appends new block matrices to an existing block-diagonal
            matrix to create an expanded block-diagonal matrix.

            This method constructs a larger block-diagonal matrix by appending
            new blocks derived from `filtered_indices`
            to an `existing_matrix`. Each new block corresponds to the size of
            index arrays provided in `filtered_indices`
            and contains entries that are the reciprocal of the block size (1/size).

            Parameters:
                existing_matrix (np.ndarray or None): The existing block-diagonal matrix
                to which new blocks will be appended.
                    - If `None` or an empty array, a new block-diagonal matrix is created from the new blocks alone.
                filtered_indices (dict): A dictionary where each key corresponds to a level
                or identifier, and each value
                    is a list of NumPy arrays. Each array represents indices of data points, and its size determines
                    the dimensions of the corresponding block matrix.

            Returns:
                np.ndarray: A new block-diagonal matrix that includes the existing matrix and the newly appended blocks.

            """
        elements_count = {key: [arr.size for arr in value] for key, value in filtered_indices.items()}

        # Collect all block sizes and create each block
        block_matrices = []
        total_new_block_size = 0

        # Iterate over each key-value pair to create each block matrix
        for sizes in elements_count.values():
            for size in sizes:
                # Create a block of size 'size x size' with entries 1/size
                block_matrix = np.full((size, size), 1 / size, dtype=float)
                block_matrices.append(block_matrix)
                total_new_block_size += size

        # Check if there is anything to append
        if total_new_block_size == 0:
            return existing_matrix

        # Create a large matrix that will include the existing and all new blocks
        if existing_matrix is None or existing_matrix.size == 0:
            # If no existing matrix, simply concatenate all new blocks
            new_block_matrix = block_matrices[0]
            for block in block_matrices[1:]:
                new_block_matrix = np.block([
                    [new_block_matrix, np.zeros((new_block_matrix.shape[0], block.shape[1]))],
                    [np.zeros((block.shape[0], new_block_matrix.shape[1])), block]
                ])
            return new_block_matrix
        else:
            # Existing matrix is present; calculate its dimension
            existing_dim = existing_matrix.shape[0]
            # Create a full matrix to hold both existing and new blocks
            full_matrix = np.zeros((existing_dim + total_new_block_size, existing_dim + total_new_block_size))
            full_matrix[:existing_dim, :existing_dim] = existing_matrix

            # Place new blocks starting from the bottom-right of the existing matrix
            start_dim = existing_dim
            for block in block_matrices:
                end_dim = start_dim + block.shape[0]
                full_matrix[start_dim:end_dim, start_dim:end_dim] = block
                start_dim = end_dim  # Update the starting dimension for the next block

            return full_matrix

    def _get_bias_and_variance(self, indices, block_matrix):
        """
           Calculates the bias squared and variance for a given iteration level to
           determine the balanced oracle iteration.

           Parameters:
               indices (np.ndarray): An array of indices corresponding to the data points being considered.
               block_matrix (np.ndarray): A matrix used in the computation of bias and variance.
           Returns:
               tuple:
                   balanced_oracle_iteration (int or None): The current `level` if `bias2` is less than
                   or equal to `variance`;
                       otherwise, `None`.
                   bias2 (float): The computed bias squared value.
                   variance (float): The computed variance value.

           """

        # Squared Bias:
        bias2 = np.mean(((np.eye(indices.shape[0]) - block_matrix) @ self.true_signal[indices]) ** 2)
        # Variance:
        variance = np.mean((block_matrix @ self.true_noise_vector[indices]) ** 2)

        return bias2, variance

    def _process_level_observations(self, observations):
        """
            Constructs a block-diagonal matrix from observations at a specific level of the tree.

            This method processes the observations collected from nodes at a particular level during
            the tree-growing process and creates a block-diagonal matrix. Each block corresponds to
            a node and is a square matrix of size equal to the number of observations at that node.

            Parameters:
                observations (dict): A dictionary where each key is a node object, and each value is
                    an integer representing the number of observations at that node.

            Returns:
                np.ndarray or None:
                    - If observations are provided and valid blocks are created, returns a NumPy ndarray
                      representing the assembled block-diagonal matrix.
                    - If there are no observations (empty dictionary), returns `None`.
                    - If no valid block matrices are created (e.g., all counts are zero), returns an empty array.

            """


        block_matrices = []
        dimensions = []

        # Iterate over each node's observations to create individual block matrices
        for node, count in observations.items():
            if count > 0:
                entry = 1 / count  #
                block_matrix = np.full((count, count), entry)  # Create a block matrix for the node
                block_matrices.append(block_matrix)
                dimensions.append(count)

        # Combine the block matrices into a single large block matrix
        total_dim = sum(dimensions)
        full_matrix = np.zeros((total_dim, total_dim))

        # Populate the full matrix with individual block matrices
        current_position = 0
        for i, block in enumerate(block_matrices):
            dim = dimensions[i]
            full_matrix[current_position:current_position + dim, current_position:current_position + dim] = block
            current_position += dim

        return full_matrix



