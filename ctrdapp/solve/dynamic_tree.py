from scipy import spatial


class DynamicTree:
    """Stores tree as multiple KDtrees to allow for fast nearest neighbor search

    The list of nodes are stored separately from the KDtrees, which contain only
    the insertion data for each node. There is an array that links the indices
    of the KDtree values to the node list. The multiple KDtrees allow for
    dynamic insertion of new nodes, while maintaining efficient nearest neighbor
    search. A NN query is O(log n * log n) and amortized insertion is
    O(log n * log n / n).

    This static-to-dynamic transformation structure is taken from  J. L. Bentley
    and J. B. Saxe*. Decomposable searching problems I: Static-to-dynamic
    transformation. J. Algorithms 1(4):301–358, 1980., as described in Jeff
    Erickson's (University of Illinois at Urbana-Champagne) lecture on advanced
    data structures.
    """

    def __init__(self, dim, ins, rot, init_heuristic, init_g_curves):
        """
        Parameters
        ----------
        dim : int
            number of dimensions represented in the node data
        ins : list of float
            initial insertion values
        rot : list of float
            initial rotation values
        init_heuristic : Heuristic
            Cost function and information for this node
        init_g_curves : list of list of 4x4 arrays
            List of SE3 g arrays for each tube
        """
        self.tube_num = dim

        init_insert_indices = [len(init_g_curves[0]) - 1] * self.tube_num
        first_node = Node(ins, rot, dim, init_heuristic, init_g_curves, init_insert_indices)
        ins_list = [ins]
        first_kd = spatial.KDTree(ins_list)

        self.nodes = [first_node]
        """list of node: initialized with first node, filled by insert method"""
        self.map2nodes = [0]
        """list of int: holds references to node location in nodes.
        Position in list matches position in KDtree, arranged from smallest to
        largest (each with size 2^i)."""
        self.kdtrees = [first_kd]
        """list of KDtree: list to hold KDtrees which have size (2^i). 
        i.e. sizes in position in list: [1, 2, 4, 8, 16]"""
        self.solution = []  # use if not test (= true) to check for empty list
        """list of int: gives the indices of the successful nodes."""

    def nearest_neighbor(self, x):
        """Finds the nearest neighbor in the tree to the given point

        Parameters
        ----------
        x : list of float
            the given point

        Returns
        -------
        list of float
            array giving the insertion values of the nearest neighbor
        int
            the index of the nearest neighbor (as stored in nodes array)
        list of float
            array giving the insertion values of the neighbor's parent
        """

        min_dist = float('inf')
        loc_best = None
        i_best = None
        for i in range(0, len(self.kdtrees)):
            if self.kdtrees[i] is not None:
                [this_dist, this_loc] = self.kdtrees[i].query(x)
                if this_dist < min_dist:
                    min_dist = this_dist
                    loc_best = this_loc
                    i_best = i
        final_ind = 2**i_best + loc_best - 1
        nodes_ind = self.map2nodes[final_ind]
        final_node = self.nodes[nodes_ind]

        parent_ind = final_node.parent
        if parent_ind is not None:
            parent_node = self.nodes[parent_ind]
        else:  # first node has no parent
            parent_node = final_node
        return final_node.insertion, nodes_ind, parent_node.insertion

    def insert(self, ins, rot, parent, heuristic, g_curves, insert_indices):
        """Inserts the new point into the tree with the given data and parent

        Parameters
        ----------
        ins : list of float
            The insertion values for the point to be added to the tree
        rot : list of float
            The rotation values for the point to be added to the tree
        parent : int
            The index of the parent node for this new node
        heuristic : Heuristic
            The cost structure and information for this new node
        g_curves : list of list of 4x4 array
            SE3 arrays for this node where g_curves[tube #][index] = 4x4 SE3
        insert_indices : list of int
            corresponds to index for "origin" SE3 array for each g_curve tube

        Returns
        -------
        VOID
        """

        heuristic.calculate_cost_from_parent(self.nodes[parent].heuristic)
        new_node = Node(ins, rot, self.tube_num, heuristic, g_curves, insert_indices, parent)
        self.nodes.append(new_node)
        this_node_index = len(self.nodes) - 1
        self.nodes[parent].children.append(this_node_index)

        # Find the first empty KDtree
        k = 0
        while k < len(self.kdtrees) and self.kdtrees[k] is not None:
            k = k+1
        new_data = []

        # collect data from all previous (smaller) KDtrees
        for j in range(0, 2**k-1):
            new_data.append(self.nodes[self.map2nodes[j]].insertion)
            DynamicTree._assign(self.map2nodes, j+2**k-1, self.map2nodes[j])
            self.map2nodes[j] = None
        new_data.append(ins)
        DynamicTree._assign(self.map2nodes, 2**(k+1)-2, this_node_index)

        # build new KDtree in the empty position
        DynamicTree._assign(self.kdtrees, k, spatial.KDTree(new_data))

        # delete previous KDtrees
        for i in range(0, k):
            self.kdtrees[i] = None

    def get_costs(self, child_ind):
        """Retrieves list of costs from child to root parent

        Parameters
        ----------
        child_ind : int
            index of node to start traversal

        Returns
        -------
        list of float
            costs of each node from the given index to the root
        """

        i = child_ind
        costs = []
        while True:
            costs.append(self.nodes[i].get_cost())
            if i == 0:
                break
            i = self.nodes[i].parent
        return costs

    def get_tube_data(self, child_ind):
        """Retrieves list of data (insertion/rotation) from **child to root**

        Parameters
        ----------
        child_ind : int
            index of node to start traversal

        Returns
        -------
        list of list of float
            list of the insertion data from child to root
        list of list of float
            list of the rotation data from child to root
        insert_indices : list of list of int
            corresponds to index for "origin" SE3 array for each g_curve tube
        """

        i = child_ind
        insertion = [self.nodes[i].insertion]
        rotation = [self.nodes[i].rotation]
        insert_indices = [self.nodes[i].insert_indices]
        while i != 0:
            i = self.nodes[i].parent
            insertion.append(self.nodes[i].insertion)
            rotation.append(self.nodes[i].rotation)
            insert_indices.append(self.nodes[i].insert_indices)

        return insertion, rotation, insert_indices

    def get_tube_curves(self, child_ind):
        """Retrieves list of g_curves from **child to root**

        Parameters
        ----------
        child_ind : int
            index of node to start traversal

        Returns
        -------
        list of list of list of 4x4 arrays
            list of the curve data from child to root
        """

        i = child_ind
        g_out = [self.nodes[i].g_curves]
        while i != 0:
            i = self.nodes[i].parent
            g_out.append(self.nodes[i].g_curves)

        return g_out

    @staticmethod
    def _assign(this_list, ind, val):
        """Adds an item to the end of a list or places it at the given index

        If index is

        Parameters
        ----------
        this_list : list of any
        ind : int
            index at which to insert the new value
        val : any
            new value to insert into the list
        """
        if len(this_list) <= ind:
            this_list.append(val)
            if len(this_list) < ind:
                raise UserWarning("Index given is beyond list size (+1)")
        else:
            this_list[ind] = val


class Node:
    """Each node holds the configuration data, as well as cost and tree refs."""

    def __init__(self, insertion, rotation, dim, heuristic, g_curves, insert_indices, parent=None):
        """
        Parameters
        ----------
        insertion : list of float
            insertion values
        rotation: list of float
            rotation values
        dim : int
            number of dimensions tree has for error checking
        heuristic : Heuristic
            Heuristic object containing cost information
        g_curves : list of list of 4x4 arrays
            g_curve[tube num][index] = 4x4 SE3 array
        insert_indices : list of int
            corresponds to index for "origin" SE3 array for each g_curve tube
        parent : int
            index of parent in node list (default is none)

        Raises
        ------
        Value Error
            If length of insertion or rotation does not match the given dim
        """
        if len(insertion) != dim or len(rotation) != dim:
            raise ValueError(f"Data size does not match given tube number ({dim})")
        self.insertion = insertion
        self.rotation = rotation
        self.parent = parent
        self.heuristic = heuristic
        self.g_curves = g_curves
        self.insert_indices = insert_indices
        self.children = []
        """list of int: positions of children in nodes list, set as children are inserted"""

    def get_cost(self):
        return self.heuristic.get_cost()
