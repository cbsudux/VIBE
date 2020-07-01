import numpy as np

inv = np.linalg.inv


class Node:
    """
    Not thread safe
    """
    def __init__(self, name):
        self.name = name
        self.parent = None
        self.childs = []

        self.transform = np.eye(4) # local
        self.callback = None

    # CALLBACK SETTERS
    def RegisterCallback(self, callback):
        self.callback = callback

    def UnRegisterCallback(self):
        self.callback = None


    # PARENT SETTER METHODS
    def SetParent(self, node):
        if self.GetParent() is node:
            return

        cb = self.callback
        self.callback = None  # temporarily disable so that other setter funcs don't invoke

        if self.HasParent():
            self.parent.RemoveChildNode(self)

        if node is None:
            self.parent = None
        else:
            assert isinstance(node, Node)
            self.parent = node
            node.AddChild(self)

        self.node_parent_recursion_check()

        self.callback = cb
        if self.callback:
            self.callback(self)

    def node_parent_recursion_check(self, parent_nodes=None):
        if parent_nodes is None:
            parent_nodes = []
        if self.HasParent():
            if self.parent not in parent_nodes:
                parent_nodes.append(self.parent)
            else:
                raise ValueError("Recursive Node parent hierarchy detected")
            self.parent.node_parent_recursion_check(parent_nodes)

    def RemoveParent(self):
        self.SetParent(None)

    # PARENT GETTER METHODS
    def HasParent(self):
        return self.GetParent() is not None

    def GetParent(self):
        return self.parent

    def IsNodeParent(self, parent_node):
        return self.GetParent() is parent_node


    # CHILD SETTER METHODS
    def AddChild(self, node):
        assert isinstance(node, Node)

        cb = self.callback
        self.callback = None  # temporarily disable so that other setter funcs don't invoke

        changed = False
        if not node.IsNodeParent(self):
            node.SetParent(self)
            changed = True
        if not self.IsNodeInChildren(node):
            self.childs.append(node)
            changed = True

        self.callback = cb

        if changed and self.callback:
            self.callback(self)

    def RemoveChildNode(self, child_node):
        idx = self.GetChildNodeIndex(child_node)
        return self.RemoveChildByIndex(idx)

    def RemoveChildByName(self, name):
        for ix, node in enumerate(self.childs):
            if node.name == name:
                self.RemoveChildByIndex(ix)
                return True
        return False

    def RemoveChildByIndex(self, index):
        if index >= len(self.childs) or index < 0:
            return False
        node = self.childs.pop(index)
        node.RemoveParent()
        if self.callback:
            self.callback(self)
        return True

    def RemoveAllChilds(self):
        if len(self.childs) == 0:
            return 

        for node in self.childs:
            node.RemoveParent()
        self.childs = []
        if self.callback:
            self.callback(self)

    # CHILD GETTER METHODS
    def GetChildCount(self):
        return len(self.childs)

    def GetChild(self, index):
        if index >= self.GetChildCount() or index < 0:
            return None
        return self.childs[index]

    def GetChildByName(self, name):
        for node in self.childs:
            if node.name == name:
                return node
        return None

    def GetChildNodeIndex(self, child_node):
        for ix, node in enumerate(self.childs):
            if node is child_node:
                return ix
        return -1

    def IsNodeInChildren(self, child_node):
        return self.GetChildNodeIndex(child_node) >= 0


    # TRANSFORM SETTER METHODS
    def SetTransform(self, transform, local=True):
        assert transform.shape == (4,4)
        if local or self.parent is None:
            self.transform = transform.copy()
        else:
            p_transform = self.parent.EvaluateTransform(local=False)
            LT = np.dot(inv(p_transform), transform)
            self.transform = LT

        # if self.callback:
        #     self.callback(self)

    def SetLocalTransform(self, transform):
        self.SetTransform(transform, local=True)

    def SetGlobalTransform(self, transform):
        self.SetTransform(transform, local=False)

    # TRANSFORM GETTER METHODS
    def EvaluateTransform(self, local=True):
        if local or self.parent is None:
            return self.transform.copy()
        else:
            p_transform = self.parent.EvaluateTransform(local=False)
            return np.dot(p_transform, self.transform)

    def EvaluateGlobalTransform(self):
        return self.EvaluateTransform(local=False)

    def EvaluateLocalTransform(self):
        return self.EvaluateTransform(local=True)


    def __repr__(self):
        parent_str = "None" if self.parent is None else self.parent.name
        return "Node instance (Name: %s, Parent: %s, Childs: %d)"%(self.name, parent_str, self.GetChildCount())


class NodeSkeleton:
    def __init__(self):
        self.nodes = []
        self.parents = np.empty(0, dtype=np.int32)

        # self.global_transforms_cache = []

    def SetNodes(self, nodes):
        assert isinstance(nodes, list)
        self.nodes = nodes
        self.populate_parents()
        self.register_node_callbacks()

    def register_node_callbacks(self):
        def callback(n):
            p_idx = self.GetNodeIndex(n.GetParent())
            n_idx = self.GetNodeIndex(n)
            # prev_p_idx = self.parents[n_idx]
            self.parents[n_idx] = p_idx

        for node in self.nodes:
            node.RegisterCallback(callback)

    def populate_parents(self):
        N = self.GetNodeCount()
        if N == 0:
            return 

        self.parents = -np.ones(N, dtype=np.int32)
        # self.global_transforms_cache = np.zeros((N, 4, 4), dtype=np.float32)
        for ix, node in enumerate(self.nodes):
            # self.global_transforms_cache[ix] = node.EvaluateGlobalTransform()
            node_parent = node.GetParent()
            if node_parent is not None:
                self.parents[ix] = self.GetNodeIndex(node_parent)

    def GetNodeCount(self):
        return len(self.nodes)

    def GetParents(self):
        return self.parents.copy()

    def GetNodeIndex(self, node_in):
        if node_in is not None:
            for ix, node in enumerate(self.nodes):
                if node_in is node:
                    return ix
        return -1

    def GetNodeByIndex(self, index):
        assert 0 <= index < self.GetNodeCount()
        return self.nodes[index]

    def EvaluateNodeGlobalTransforms(self):
        return self.EvaluateNodeTransforms(local=False)

    def EvaluateNodeLocalTransforms(self):
        return self.EvaluateNodeTransforms(local=True)

    def EvaluateNodeTransforms(self, local=False):
        N = self.GetNodeCount()
        GT = np.zeros((N, 4, 4), dtype=np.float32)
        if local:
            for ix, node in enumerate(self.nodes):
                GT[ix] = node.EvaluateLocalTransform()
        else:
            # to avoid recomputing global parent transforms
            visited = np.zeros(N, dtype=np.bool)
            for ix, node in enumerate(self.nodes):
                if visited[ix]:
                    continue
                p_ix = self.parents[ix]
                M = node.EvaluateLocalTransform() 
                if p_ix >= 0:
                    if not visited[p_ix]:
                        GT[p_ix] = self.nodes[p_ix].EvaluateGlobalTransform()
                        visited[p_ix] = True
                    M = np.dot(GT[p_ix], M)
                GT[ix] = M
                visited[ix] = True
        return GT


if __name__ == '__main__':
    # from transforms3d.euler import euler2mat, mat2euler

    def check_np_is_same(x, x2, eps=1e-4):
        return np.all(np.abs(np.asarray(x) - np.asarray(x2)) < eps)

    def assert_same(x, x2, eps=1e-4):
        assert check_np_is_same(x, x2, eps)

    def get_random_4x4():
        M = np.eye(4)
        M[:3,:3] = np.random.random(size=(3,3))
        M[:3,3] = np.random.random(size=3)
        return M

    def unit_test_Node_parenting():
        global III 
        III = 0

        def incr(node):
            global III
            III += 1
            # print(node)

        node1 = Node("node1")
        node2 = Node("node2")

        node1.RegisterCallback(incr)
        node1.AddChild(node2)

        assert III == 1
        assert node1.GetChildCount() == 1
        assert node1.IsNodeInChildren(node2) and node1.GetChild(0) is node2
        assert node2.IsNodeParent(node1) and node2.GetParent() is node1

        node2.RemoveParent()
        assert III == 2
        assert node2.GetParent() is None and node1.GetChildCount() == 0

        node3 = Node("node3")
        node1.SetParent(node3)
        node2.SetParent(node3)
        assert III == 3
        assert node2.GetParent() is node1.GetParent() is node3
        assert node3.GetChildCount() == 2
        assert node3.GetChild(node3.GetChildNodeIndex(node1)) is node1
        assert node3.GetChild(node3.GetChildNodeIndex(node2)) is node2
        assert node3.GetChildNodeIndex(node3) == -1

        node2.SetParent(node1)
        assert III == 4
        assert node1.GetChildCount() == 1 and node1.GetChild(0) is node2
        assert node3.GetChildCount() == 1 and node3.GetChild(0) is node1
        assert node3.GetChild(1) is None
        assert node2.GetParent().GetParent() is node3

        ret = node1.RemoveChildByIndex(0)
        assert III == 5
        assert ret
        assert node2.GetParent() is None and node1.GetChildCount() == 0

        node3.RemoveAllChilds()
        assert III == 6
        assert node3.GetChildCount() == 0 and node1.parent is None

        # raise and handle recursive node parenting exception
        try:
            node4 = Node("node4")
            node1.SetParent(node2)
            node2.SetParent(node3)
            node3.SetParent(node4)
            node4.SetParent(node1)
            node1.EvaluateGlobalTransform()
        except ValueError:
            pass
        print("PASSED unit_test_Node_parenting()")


    def unit_test_Node_transforms():
        node1 = Node("node1")
        node2 = Node("node2")
        node3 = Node("node3")

        node1.AddChild(node2)
        node2.AddChild(node3)
        assert node1.IsNodeInChildren(node2)
        assert node2.IsNodeParent(node1)
        assert node3.GetParent().GetParent() is node1

        LM1 = get_random_4x4()
        LM2 = get_random_4x4()
        LM3 = get_random_4x4()

        M1 = LM1.copy()
        M2 = np.dot(M1, LM2)
        M3 = np.dot(M2, LM3)
        
        node1.SetLocalTransform(LM1)
        node2.SetLocalTransform(LM2)
        node3.SetLocalTransform(LM3)
        
        node2_T = node2.EvaluateGlobalTransform()
        node3_T = node3.EvaluateGlobalTransform()
        
        assert_same(M2, node2_T)
        assert_same(M3, node3_T)

        node2.SetGlobalTransform(M2)
        node2_LT = node2.EvaluateLocalTransform()
        assert_same(LM2, node2_LT)
        node2_T = node2.EvaluateGlobalTransform()
        assert_same(M2, node2_T)

        assert not check_np_is_same(node2_T, node2_LT)

        node1.RemoveChildNode(node2)
        assert not node1.IsNodeInChildren(node2)
        assert node2.parent is None

        M2 = LM2.copy()
        node2_T = node2.EvaluateGlobalTransform()
        node2_LT = node2.EvaluateLocalTransform()
        assert_same(node2_LT, node2_T)

        M3 = np.dot(LM2, LM3)
        assert_same(M3, node3.EvaluateGlobalTransform())

        print("PASSED unit_test_Node_transforms()")

    def unit_test_Skeleton():

        sk = NodeSkeleton()
        node1 = Node("node1")
        node2 = Node("node2")
        node3 = Node("node3")

        node1.AddChild(node2)
        node2.AddChild(node3)
        assert node3.GetParent().GetParent() is node1

        nodes = [node1, node2, node3]
        sk.SetNodes(nodes)
        assert_same(sk.parents, [-1,0,1])

        LM1 = get_random_4x4()
        LM2 = get_random_4x4()
        LM3 = get_random_4x4()

        node1.SetLocalTransform(LM1)
        node2.SetLocalTransform(LM2)
        node3.SetLocalTransform(LM3)

        M1 = LM1.copy()
        M2 = np.dot(M1, LM2)
        M3 = np.dot(M2, LM3)

        # standard transforms test
        GT = sk.EvaluateNodeGlobalTransforms()
        assert_same(GT[0], M1)
        assert_same(GT[1], M2)
        assert_same(GT[2], M3)

        LT = sk.EvaluateNodeLocalTransforms()
        assert_same(LT[0], LM1)
        assert_same(LT[1], LM2)
        assert_same(LT[2], LM3)

        # change transform of nodes
        LM2 = get_random_4x4()
        LM3 = get_random_4x4()
        M2 = np.dot(M1, LM2)
        M3 = np.dot(M2, LM3)
        node2.SetLocalTransform(LM2)
        node3.SetGlobalTransform(M3)
        GT = sk.EvaluateNodeGlobalTransforms()

        assert_same(GT[1], M2)
        assert_same(GT[2], M3)

        # rearrange nodes
        nodes = [node3, node1, node2]
        sk.SetNodes(nodes)
        assert_same(sk.parents, [2,-1,1])

        GT = sk.EvaluateNodeGlobalTransforms()
        assert_same(GT[0], M3)
        assert_same(GT[1], M1)
        assert_same(GT[2], M2)

        # adjust node parenting/childs
        node1.RemoveAllChilds()
        node2.RemoveAllChilds()
        assert_same(sk.parents, [-1,-1,-1])

        GT = sk.EvaluateNodeGlobalTransforms()
        assert_same(GT[0], node3.EvaluateLocalTransform())
        assert_same(GT[1], node1.EvaluateLocalTransform())
        assert_same(GT[2], node2.EvaluateLocalTransform())

        print("PASSED unit_test_Skeleton()")


    unit_test_Node_parenting()
    unit_test_Node_transforms()
    unit_test_Skeleton()
