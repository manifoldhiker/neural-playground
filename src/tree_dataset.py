import random
import collections
from torch.utils.data import IterableDataset, DataLoader
import torch

from .tree import TreeNode


def random_binary_tree(n):
    node_names = list(range(n))
    random.shuffle(node_names)

    def _gen(n):
        if n == 0:
            return None
    
        node_val = node_names.pop()
    
        if n == 1:
            return TreeNode(val=node_val)
        else:
            left_size = random.randint(0, n-1)
            right_size = n - 1 - left_size
            left_subtree = _gen(left_size)
            right_subtree = _gen(right_size)
            return TreeNode(val=node_val, left=left_subtree, right=right_subtree)
        
    return _gen(n)


def inorder_knuth(root):
    # print('Tribute to Knuth.')
    # print("(1969). Fundamental Algorithms. The Art of Computer Programming. Vol. 1")

    res = []
    def Visit(P):
        if P.left is None and P.right is None:
            res.append(P.val)

    # T1. [Initialize.] Set stack A empty, and set link variable P <- T
    A = []
    P = root
    while True:
        # T2. [P = NULL?] if P == NULL, go to step T4.
        if P is None:
            # T4. [P <= Stack] If stack A is empty, algorithm terminates
            if len(A)==0: break
            # Otherwise, set P <= A
            P = A.pop()

            # T5. [Visit P.] Visit NODE(P).
            Visit(P)
            # Then, set P <- RLINK(P) and return to T2.
            P = P.right
        else:
            # T3. [Stack <= P.] (Now P points to a nonempty binary tree that is to be traversed.)
            # Set A <= P; That is, push the P onto the stack A
            A.append(P)
            # Then, set P <- LLINK(P) and return to T2.
            P = P.left
    return res

def get_leaf_vals(root): return inorder_knuth(root)


def tree_to_edges(root):
    """ Convert a binary tree to a list of edges. """
    edges = []
    def traverse(node):
        if node.left:
            edges.append((node.val, node.left.val))
            traverse(node.left)
        if node.right:
            edges.append((node.val, node.right.val))
            traverse(node.right)
    traverse(root)
    return edges

def edges_to_tree(edges):
    """ Create a binary tree from a list of edges. """
    if not edges:
        return None
    
    # Create nodes and find the root
    nodes = {}
    children = set()
    
    for parent, child in edges:
        if parent not in nodes:
            nodes[parent] = TreeNode(parent)
        if child not in nodes:
            nodes[child] = TreeNode(child)
        if child in nodes:
            if nodes[parent].left is None:
                nodes[parent].left = nodes[child]
            else:
                nodes[parent].right = nodes[child]
        children.add(child)
        
    # Find the root (node not in children set)
    root_node = next(node for node in nodes if node not in children)
    return nodes[root_node]


def get_parent_dict(root):
    parent = dict()
    stack = [(root)]
    
    while stack:
        root = stack.pop()
        if root is None: continue
        
        if root.left:
            parent[root.left.val] = root.val
            stack.append(root.left)
        if root.right:
            parent[root.right.val] = root.val
            stack.append(root.right)
            
    return parent


def from_root_to_node(tree, node_val):
    parent = get_parent_dict(tree)
    path = []
    
    while node_val is not None:
        path.append(node_val)
        node_val = parent.get(node_val)
        
    path = list(reversed(path))
    return path




def edges_to_tokens(edges): return [token for from_node, to_node in edges for token in (str(from_node),f'→{to_node}', ',')][:-1]


def tree2tokens(tree):
    edges = tree_to_edges(tree)
    random.shuffle(edges)
    
    edge_tokens = edges_to_tokens(edges)
    root = tree.val
    leafs = get_leaf_vals(tree)

    goal = random.choice(leafs)

    input_tokens = [*edge_tokens, '|', goal, ':', root]
    
    target_path = from_root_to_node(tree, goal)[1:]
    path_tokens = [f'→{node_val}' for node_val in target_path]
    
    return input_tokens, path_tokens


N_NODES_DEFAULT = 5

def generate_datapoint(n_nodes=N_NODES_DEFAULT):
    tree = random_binary_tree(n_nodes)
    return tree2tokens(tree)


def input_tokens_to_tree(input_tokens):
    idx = input_tokens.index('|')
    edge_tokens = ''.join(input_tokens[:idx])
    edges = [edge.split('→') for edge in edge_tokens.split(',')]
    tree = edges_to_tree(edges)
    return tree


def parse_input_idx(input_idx, tokenizer):
    input_tokens = tokenizer.detokenize(input_idx)

    input_tokens = [t for t in input_tokens if t != PAD_TOKEN]
    idx = input_tokens.index('|')
    edge_tokens = ''.join(input_tokens[:idx])
    edges = [edge.split('→') for edge in edge_tokens.split(',')]
    tree = edges_to_tree(edges)

    output_tokens = input_tokens[idx+1:]

    goal_node = output_tokens[0]
    root_node = output_tokens[2]
    path = output_tokens[3:]

    return {
        'tree': tree,
        'goal': goal_node,
        'root': root_node,
        'path': path,
    }


PAD_TOKEN = '<PAD>'

class MyTreeTokenizer:
    special_tokens = [PAD_TOKEN, ',', '|', ':']


    def __init__(self, n_nodes):
        self.n_nodes = n_nodes

        self.MAX_SEQ_LEN = self.n_nodes * 4 + 4

        tokens = [t for n in range(n_nodes) for t in [str(n), f'→{n}']]
        tokens = self.special_tokens+tokens
        self.idx2token = {i:str(token) for i,token in enumerate(tokens)}
        
        self.token2idx = {t:i for i,t in self.idx2token.items()}
        
    def tokenize(self, s):
        return [self.token2idx[str(t)] for t in s]

    def detokenize(self, idx):
        return [self.idx2token[int(i)] for i in idx]


    def __call__(self, s): return self.tokenize(s)


class TreeDataset(IterableDataset):
    def __init__(self, n_nodes=5):
        self.n_nodes = n_nodes
        self.tokenizer = MyTreeTokenizer(n_nodes)
    
    def __iter__(self):
        while True: 
            prompt_tokens, path_tokens = generate_datapoint(self.n_nodes)
            prompt_idx = self.tokenizer(prompt_tokens)
            path_idx = self.tokenizer(path_tokens)

            input_tokens = prompt_idx + path_idx
            pad_len = self.tokenizer.MAX_SEQ_LEN - len(input_tokens)
    
            input_idx = torch.tensor(input_tokens + [0] * pad_len)
            # pad_mask = torch.zeros(self.tokenizer.MAX_SEQ_LEN)
            # pad_mask[:len(input_idx)] = 1
            
            task_mask = torch.zeros(self.tokenizer.MAX_SEQ_LEN, dtype=torch.bool)
            task_mask[len(prompt_idx):len(input_tokens)] = True
            
            yield {
                'input_idx': input_idx,
                'task_mask': task_mask,
                # 'pad_mask': pad_mask,
            }
