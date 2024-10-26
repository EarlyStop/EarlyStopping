from __future__ import annotations
from typing import Tuple
import numpy as np
import pandas as pd
from queue import Queue
import warnings

def clone_tree(node):
    """
       Creates a deep copy of a binary tree rooted at the given `node`.

       This function recursively clones each node and its children, ensuring that the new tree
       is a completely separate object from the original. All attributes and properties of each
       node are copied over to their corresponding cloned nodes.

       Parameters:
           node (Node): The root node of the tree or subtree to clone.

       Returns:
           Node: The root node of the cloned tree or subtree. If the input `node` is `None`, returns `None`.

       """

    if node is None:
        return None

    # Create a new node and copy properties
    cloned_node = DecisionTreeRegressor.Node()
    cloned_node.set_params(node.split_threshold, node.variable)
    cloned_node.is_terminal = node.is_terminal
    cloned_node.node_prediction = node.node_prediction
    cloned_node.design_and_response  = node.design_and_response

    # Recursively clone children
    left_cloned = clone_tree(node.get_left_node())
    right_cloned = clone_tree(node.get_right_node())
    cloned_node.set_children(left_cloned, right_cloned)

    # Set the node as terminal if it has no children
    if left_cloned is None and right_cloned is None:
        cloned_node.is_terminal = True


    return cloned_node

class DecisionTreeRegressor():

    class Node:
        def __init__(self):
            self.split_threshold = None  # This can be a float or None
            self.variable = None         # This can be an int or None
            self._left = None            # This can be a Node or None
            self._right = None           # This can be a Node or None
            self.is_terminal = False     # Boolean indicating if the node is terminal
            self.node_prediction = None  # Prediction at this node (float or None)
            self.design_and_response = None  # This can be a NumPy array or None

        def set_params(self, split_threshold: float, variable: int) -> None:
            """ Set split and variable parameters for this node. """
            self.split_threshold = split_threshold
            self.variable = variable

        def get_params(self):
            """ Get the split and variable parameters for this node. """
            return (self.split_threshold, self.variable)

        def set_children(self, left, right) -> None:
            """ Set left and right child nodes for the current node. """
            self._left = left
            self._right = right

        def get_left_node(self):
            """ Get the left child node. """
            return self._left

        def get_right_node(self):
            """ Get the right child node. """
            return self._right


    def __init__(self,
                 design: np.array = None,
                 response: np.array = None,
                 min_samples_split: int = 2,
                 true_signal: np.array = None,
                 true_noise_vector: np.array = None):
        """
        Initializer
        Inputs:
            true_noise_vector      -> true noise
            min_samples_split -> minimum number of samples required to split a node
        [...]
        """
        self.regression_tree = None
        self.true_signal = true_signal
        self.minimal_samples_split = min_samples_split
        self.true_noise_vector = true_noise_vector
        self.trees_at_each_level = {}  # Dictionary to store tree copies at each level
        self.design = design
        self.response = response

        # Parameters of the model
        self.sample_size = self.design.shape[0]

        # For theoretical quantities:
        self.observations_per_level = {}  # Dictionary to store observations count per level
        self.indices_per_level = {}
        self.block_matrix = {}
        self.indices_processed = {}


    def _find_best_split(self, design_and_response: np.array) -> Tuple[int, float] | None:
        """
        Protected function to find the best split for a node

        Input:
            design_and_response -> data to find the best split for
        Output:
        Output:
            Tuple containing the index of the variable and the value to split on,
            or None if no valid split is found
        """
        impurity_node = None
        best_variable = None
        best_split_threshold = None

        # Iterate through the possible variable/split combinations
        for variable in range(design_and_response.shape[1] - 1):
            for split_threshold in np.unique(design_and_response[:, variable]):
                # Split the dataset
                left_node_mask = design_and_response[:, variable] <= split_threshold
                right_node_mask = ~left_node_mask
                left_design_and_response = design_and_response[left_node_mask]
                right_design_and_response = design_and_response[right_node_mask]

                # Ensure non-empty arrays
                if (left_design_and_response.shape[0]>= self.minimal_samples_split and
                        right_design_and_response.shape[0]>= self.minimal_samples_split):
                    # Calculate the impurity
                    impurity_candidate = (left_design_and_response.shape[0] / self.sample_size) * self._impurity(left_design_and_response) + \
                         (right_design_and_response.shape[0] / self.sample_size) * self._impurity(right_design_and_response)

                    # Update the impurity and choice of variable/split
                    if impurity_node is None or impurity_candidate < impurity_node:
                        impurity_node = impurity_candidate
                        best_variable = variable
                        best_split_threshold = split_threshold

        # Return the best variable and split
        if best_variable is not None and best_split_threshold is not None:
            return best_variable, best_split_threshold
        else:
            return None

    def __grow_regression_tree_breadth_first(self, node: Node, current_indices) -> None:

        """
            Grows a regression tree using a breadth-first approach.

            This method constructs a regression tree by expanding nodes level by level,
            starting from the root node. It uses a queue to manage the nodes at each level
            and applies splitting criteria to decide whether to split a node or make it terminal.

            Parameters:
                node (Node): The root node of the tree or subtree to grow.
                design_and_response (np.array): A NumPy array containing both the design (features)
                    and the response (target variable). Each row corresponds to a sample, and columns
                    include both features and the response variable.
                current_indices (np.array): An array of indices corresponding to the samples
                    in `design_and_response`. These indices help track the position of samples
                    in the original dataset throughout the tree-growing process.

            Returns:
                None: The tree is grown in place, and various attributes are updated to reflect
                the state of the tree at each level.

            """

        self.balanced_oracle_iteration = None
        self.queue = Queue()
        self.level = 1
        self.current_indices = current_indices
        self.queue.put((node, self.design_and_response, self.current_indices, self.level))

        self.terminal_indices = {}  # Dictionary to store indices for each terminal node
        self.block_matrices_full = {}
        self.indices_complete_all = {}

        self.bias2 = np.array([])
        self.variance = np.array([])
        self.risk = np.array([])

        self.indices_collect= {}
        self.block_matrix_collect = {}

        self.balanced_oracle_iteration_collect = {}
        self.residuals = np.array([])

        while not self.queue.empty():
            self.level_mse_sum = 0  # Initialize the sum of MSE for the current level
            self.level_node_count = 0  # Initialize the count of nodes for the current level
            self.next_level_queue = Queue()  # Queue for storing nodes of the next level
            self.level_indices = []  # List to store all indices for this level

            self.current_level_observations = {}  # Dictionary to store the number of observations for the current level

            # Process all nodes at the current level (= perform 'one iteration'):
            self._grow_one_iteration()

            # Store quantities at this level
            if self.level_indices:
                # Processing of block matrix:
                self._block_matrix_processing()
                # Get the bias2 and the variance:
                self._get_theoretical_quantities()

            self.residuals = np.append(self.residuals, self.level_mse_sum)

            # After processing the current level, store the tree state
            self.trees_at_each_level[self.level] = clone_tree(self.regression_tree)

            # Prepare for the next level of nodes
            self.queue = self.next_level_queue

    def _get_theoretical_quantities(self):
        if self.true_signal is not None and self.true_noise_vector is not None:
            if self.block_matrix[self.level].shape[0] == self.design_and_response.shape[0]:
                self.new_bias2, self.new_variance = self._get_bias_and_variance(self.indices_processed[self.level],
                                                              self.block_matrix[self.level], self.level)
                self.balanced_oracle_iteration_collect[self.level] = self.balanced_oracle_iteration

                self.bias2 = np.append(self.bias2, self.new_bias2)
                self.variance = np.append(self.variance, self.new_variance)
                self.risk = np.append(self.risk, self.level_mse_sum)

                self.indices_collect[self.level] = self.indices_processed[self.level]
                self.block_matrix_collect[self.level] = self.block_matrix[self.level]
            else:
                self.new_bias2, self.new_variance = self._get_bias_and_variance(self.indices_complete,
                                                              self.block_matrices_full[self.level], self.level)
                self.balanced_oracle_iteration_collect[self.level] = self.balanced_oracle_iteration

                self.bias2 = np.append(self.bias2, self.new_bias2)
                self.variance = np.append(self.variance, self.new_variance)
                self.risk = np.append(self.risk, self.level_mse_sum)

                self.indices_collect[self.level] = self.indices_complete
                self.block_matrix_collect[self.level] = self.block_matrices_full[self.level]

    def _block_matrix_processing(self):
        self.indices_per_level[self.level] = self.level_indices
        # Store the observations data for the entire level
        self.observations_per_level[self.level] = self.current_level_observations
        # Process observations after completing the level
        self.block_matrix[self.level] = self._process_level_observations(self.current_level_observations)
        # Process indices after completing the level
        self.indices_processed[self.level] = np.concatenate(self.level_indices)

        if self.block_matrix[self.level].shape[0] < self.design_and_response.shape[0]:
            indices_pre_append = self.indices_processed[self.level]
            filtered_indices = {k: v for k, v in self.terminal_indices.items() if k != self.level}

            # Check if filtered_indices is None or empty
            if not filtered_indices:
                print("No indices to concatenate.")
                # Handle the case where there is nothing to concatenate
                return

            self.block_matrices_full[self.level] = self.append_block_matrix(self.block_matrix[self.level],
                                                                            filtered_indices)
            indices_append = np.concatenate(
                [idx for self.level in range(1, self.level) for idx in self.terminal_indices.get(self.level, [])])

            self.indices_complete = np.append(indices_pre_append, indices_append)
            self.indices_complete_all[self.level] = self.indices_complete

    def _grow_one_iteration(self):

        # Process all nodes at the current level
        for _ in range(self.queue.qsize()):
            self.node, self.design_and_response_queue, self.current_indices, self.level = self.queue.get()

            # Always calculate and update node_prediction, not just for terminal nodes
            self.node.node_prediction = self._node_prediction_value(self.design_and_response_queue)
            self.node.design_and_matrix = self.design_and_response_queue

            # Calculate MSE for the current node and update level statistics
            node_mse = self._impurity(self.design_and_response_queue)
            self.level_mse_sum += node_mse * (self.design_and_response_queue.shape[0] / self.sample_size)
            self.level_node_count += 1

            # Check termination conditions
            terminal_due_to_samples = self.design_and_response_queue.shape[0] <= (self.minimal_samples_split * 2)
            terminal_due_to_depth = self.maximal_depth is not None and self.level >= self.maximal_depth

            if terminal_due_to_samples or terminal_due_to_depth:
                self.node.node_prediction = self._node_prediction_value(self.design_and_response_queue)
                self.node.is_terminal = True
                continue

            else:
                split_params = self._find_best_split(self.design_and_response_queue)
                if split_params is not None:
                    split_variable, split_value = split_params
                    left_node_mask = self.design_and_response_queue[:, split_variable] <= split_value
                    right_node_mask = ~left_node_mask

                    # Create left and right child nodes and set current node parameters
                    self.left_node, self.right_node = self.Node(), self.Node()
                    self.node.set_params(split_value, split_variable)
                    self.node.set_children(self.left_node, self.right_node)

                    # Update indices for left and right children
                    self.left_indices = self.current_indices[left_node_mask]
                    self.right_indices = self.current_indices[right_node_mask]

                    self.left_design_and_response, self.right_design_and_response = (self.design_and_response_queue[left_node_mask],
                                                                           self.design_and_response_queue[right_node_mask])
                    if self.left_design_and_response.size > 0 and self.right_design_and_response.size > 0:
                        self.level_indices.extend([self.left_indices, self.right_indices])  # Storing global indices

                        # Assign data to child nodes before recursion or further processing
                        self.left_node.design_and_response = self.left_design_and_response
                        self.right_node.design_and_response = self.right_design_and_response
                        self.left_node.node_prediction = self._node_prediction_value(self.left_design_and_response)
                        self.right_node.node_prediction = self._node_prediction_value(self.right_design_and_response)

                    # Add child nodes to the next level queue for further exploration
                    self.next_level_queue.put(
                        (self.left_node, self.left_design_and_response, self.left_indices, self.level + 1))
                    self.next_level_queue.put(
                        (self.right_node, self.right_design_and_response, self.right_indices, self.level + 1))

                    # Store observations for each node
                    self.current_level_observations[self.left_node] = self.left_design_and_response.shape[0]
                    self.current_level_observations[self.right_node] = self.right_design_and_response.shape[0]

                    if self.left_design_and_response.shape[0] <= (
                            self.minimal_samples_split * 2):  # No further splits possible.
                        if self.level not in self.terminal_indices:
                            self.terminal_indices[self.level] = []
                        self.terminal_indices[self.level].append(self.left_indices)

                    if self.right_design_and_response.shape[0] <= (
                            self.minimal_samples_split * 2):  # No further splits possible.
                        if self.level not in self.terminal_indices:
                            self.terminal_indices[self.level] = []
                        self.terminal_indices[self.level].append(self.right_indices)

                else:
                    # If no valid split is found, make current node a leaf
                    self.node.node_prediction = self._node_prediction_value(self.design_and_response_queue)
                    self.node.is_terminal = True  # Mark as terminal

    def get_discrepancy_stop(self, critical_value):
        """Returns early stopping index based on discrepancy principle up to max_iteration

        **Parameters**

        *critical_value*: ``float``. The critical value for the discrepancy principle. The algorithm stops when
        :math: `\\Vert Y - A \hat{f}^{(m)} \\Vert^{2} \leq \\kappa^{2},`
        where :math: `\\kappa` is the critical value.

        **Returns**

        *early_stopping_index*: ``int``. The first iteration at which the discrepancy principle is satisfied.
        (None is returned if the stopping index is not found.)
        """
        if np.any(self.residuals<=critical_value):
            # argmax takes the first instance of True in the true-false array
            early_stopping_index = np.argmax(self.residuals <= critical_value)
            return early_stopping_index
        else:
            warnings.warn("Early stopping index not found. Returning None.", category=UserWarning)
            return None

    def get_balanced_oracle(self):
        """Returns strong balanced oracle if found up to max_iteration.

        **Parameters**

        *max_iteration*: ``int``. The maximum number of total iterations to be considered.

        **Returns**

        *strong_balanced_oracle*: ``int``. The first iteration at which the strong bias is smaller than the strong variance.
        """

        if np.any(self.bias2 <= self.variance):
            # argmax takes the first instance of True in the true-false array
            balanced_oracle = np.argmax(self.bias2 <= self.variance)
            return balanced_oracle

        else:
            warnings.warn(
                "Balanced oracle not found. Returning None.", category=UserWarning
            )
            return None

    def _impurity(self, design_and_response_input: np.array) -> float:

        # compute the mean target for the node
        response_node_mean = np.mean(design_and_response_input[:, -1])
        # compute the mean squared error wrt the mean
        mse = np.sum((design_and_response_input[:, -1] - response_node_mean) ** 2) / design_and_response_input.shape[0]
        # return results
        return (mse)

    def _node_prediction_value(self, design_and_response: np.array) -> float:
        """
        Protected function to compute the value at a leaf node

        Input:
            design_and_response -> data to compute the leaf value
        Output:
            Mean of design_and_response
        """
        return (np.mean(design_and_response[:, -1]))

    def append_block_matrix(self, existing_matrix, filtered_indices):
        """
            Appends new block matrices to an existing block-diagonal matrix to create an expanded block-diagonal matrix.

            This method constructs a larger block-diagonal matrix by appending new blocks derived from `filtered_indices`
            to an `existing_matrix`. Each new block corresponds to the size of index arrays provided in `filtered_indices`
            and contains entries that are the reciprocal of the block size (1/size).

            Parameters:
                existing_matrix (np.ndarray or None): The existing block-diagonal matrix to which new blocks will be appended.
                    - If `None` or an empty array, a new block-diagonal matrix is created from the new blocks alone.
                filtered_indices (dict): A dictionary where each key corresponds to a level or identifier, and each value
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

    def _get_bias_and_variance(self, indices, block_matrix, level):
        """
           Calculates the bias squared and variance for a given iteration level to determine the balanced oracle iteration.

           Parameters:
               indices (np.ndarray): An array of indices corresponding to the data points being considered.
               block_matrix (np.ndarray): A matrix used in the computation of bias and variance.
               level (int): The current iteration level in the tree-growing process.

           Returns:
               tuple:
                   balanced_oracle_iteration (int or None): The current `level` if `bias2` is less than or equal to `variance`;
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

        if len(observations) == 0:
            return None  # If there are no observations, nothing to process

        # Initialize the list for storing block matrices and their dimensions
        block_matrices = []
        dimensions = []

        # Iterate over each node's observations to create individual block matrices
        for node, count in observations.items():
            if count > 0:
                entry = 1 / count  # Calculate the entry value
                block_matrix = np.full((count, count), entry)  # Create a block matrix for the node
                block_matrices.append(block_matrix)
                dimensions.append(count)

        # Combine the block matrices into a single large block matrix
        if block_matrices:
            total_dim = sum(dimensions)
            full_matrix = np.zeros((total_dim, total_dim))  # Initialize the full block matrix

            # Populate the full matrix with individual block matrices
            current_position = 0
            for i, block in enumerate(block_matrices):
                dim = dimensions[i]
                full_matrix[current_position:current_position + dim, current_position:current_position + dim] = block
                current_position += dim

            return full_matrix  # Return the assembled block matrix
        else:
            print("No valid block matrices were created.")
            return np.array([])  # Return an empty array if no blocks were created

    def __traverse(self, node, design_row):
        """
           Recursively traverses the decision tree to make a prediction for a single data point.

           Parameters:
               node (Node): The current node in the decision tree.
               design_row (array-like): A single sample's feature values.
                   - Should be indexed such that `design_row[f]` accesses the feature at index `f`.

           Returns:
               float: The predicted value obtained from the terminal node corresponding to the input sample.

           """
        # if node is None or node.is_terminal:
        #     return node.node_prediction
        #
        if node.is_terminal:
            return node.node_prediction

        s, f = node.get_params()
        if design_row[f] <= s:
            return self.__traverse(node.get_left_node(), design_row)
        else:
            return self.__traverse(node.get_right_node(), design_row)

    def iterate(self, design: pd.DataFrame | np.ndarray, response: pd.Series | np.ndarray | pd.DataFrame, max_depth: int = None,) -> None:
        """
        Train the CART model

        Inputs:
            design -> input set of predictor variables
            response -> input set of labels
            max_depth -> maximum number in generations. The number of splits conducted is max_depth - 1.

        """
        # Convert pandas DataFrame/Series to numpy array if necessary
        if isinstance(design, pd.DataFrame):
            design = design.to_numpy()
        if isinstance(response, (pd.Series, pd.DataFrame)):
            response = response.to_numpy()

        self.maximal_depth = max_depth
        self.response = response.reshape(-1, 1)
        # prepare the input data
        self.design_and_response = np.concatenate((design, self.response), axis=1)
        # set the root node of the tree
        self.regression_tree = self.Node()
        # build the tree
        self.initial_indices = np.arange(self.sample_size)  # Initial indices for the entire dataset
        self.__grow_regression_tree_breadth_first(node=self.regression_tree, current_indices=self.initial_indices)  # Initial indices for the entire dataset

    def predict(self, design: pd.DataFrame | np.ndarray, depth: int) -> np.array:
        """
        Predicts target values for the given design data using the decision tree at the specified depth.

        Parameters:
            design (pd.DataFrame or np.ndarray): The input feature data for which predictions are to be made.
                - Each row represents a sample.
                - Each column represents a feature.
            depth (int): The depth of the tree to use for making predictions.
                - If `depth` is 1, the function returns the unconditional mean of the response variable.

        Returns:
            np.array: An array of predicted values corresponding to each input sample.

        """
        if depth <= self.maximal_depth:
            if isinstance(design, pd.DataFrame):
                design = design.to_numpy()
            # Unconditional mean prediction for depth=0
            if depth == 0:
                return np.repeat(np.mean(self.response), design.shape[0])
            else:
                tree = self.trees_at_each_level[depth] # self.trees_at_each_level does not include unconditional mean
                predictions = []
                for r in range(design.shape[0]):
                    predictions.append(self.__traverse(tree, design[r, :]))
                return np.array(predictions)
        else:
            warnings.warn("'Depth' can not exceed 'max_depth'. Returning None.", category=UserWarning)
            return None


