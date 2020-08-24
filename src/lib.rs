use std::cell::RefCell;
use rand::prelude::*;

use serde::{Deserialize, Serialize};

#[derive(Debug)]
struct Server<T: Clone> {
    inner: RefCell<Vec<T>>,
    length: usize,
}

impl<T: Clone> Server<T> {
    fn new<F>(length: usize, default: &F) -> Server<T>
        where F: Fn() -> T {
        let mut vs: Vec<T> = Vec::new();
        for _ in 0..length {
            vs.push(default());
        }
        Server {
            inner: RefCell::new(vs),
            length,
        }
    }
    fn read(&self, i: usize) -> Option<T> {
        let x = self.inner.borrow().get(i)?.clone();
        Some(x)
    }

    fn write(&self, i: usize, x: T) {
        self.inner.borrow_mut()[i] = x;
    }
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
struct Block {
    inner: [u8; 32],
    empty: bool,
    leaf: LeafIdx,
    u: Idx,
}

impl Block {
    fn empty() -> Block {
        Self::empty_with_u(Idx::max_value())
    }

    fn empty_with_u(u: Idx) -> Block {
        Block {
            inner: rand::random(),
            empty: true,
            leaf: 0,
            u,
        }
    }

    fn new(u: Idx, value: [u8; 32], leaf: LeafIdx) -> Block {
        Block {
            inner: value,
            empty: false,
            leaf,
            u
        }
    }
}

type Idx = usize;
type LeafIdx = usize;

trait ORAM {
    fn read_and_remove(&self, u: Idx) -> Result<Block, String>;
    fn add(&self, block: &Block) -> Result<(), String>;
    fn pop(&self) -> Result<Block, String>;
    fn capacity(&self) -> usize;

    fn read(&self, u: Idx) -> Result<Block, String> {
        let block = self.read_and_remove(u)?;
        self.add(&block)?;
        Ok(block)
    }

    fn write(&self, block: &Block) -> Result<(), String> {
        self.read_and_remove(block.u)?;
        self.add(block)
    }
}

#[derive(Debug)]
struct TrivialORAM {
    key: [u8; 32],
    nonce: [u8; 32],
    server: Server<String>,
}

impl TrivialORAM {
    fn new(n: usize) -> TrivialORAM {
        let f = || serde_json::to_string(&Block::empty()).unwrap();
        let server = Server::new(n, &f);
        TrivialORAM::with_server(server)
    }

    fn with_leaf_generator<F>(n: usize, leaf_generator: &F) -> TrivialORAM
        where F : Fn() -> LeafIdx {
        let f = || {
            let b = Block {
                inner: rand::random(),
                empty: true,
                leaf: leaf_generator(),
                u: Idx::max_value(),
            };
            serde_json::to_string(&b).unwrap()
        };
        let server = Server::new(n, &f);
        TrivialORAM::with_server(server)
    }

    fn with_server(server: Server<String>) -> TrivialORAM {
        TrivialORAM {
            key: [0; 32],
            nonce: [0; 32],
            server,
        }
    }

    fn enc(&self, block: &Block) -> Result<String, String> {
        // TODO use proper encryption
        match serde_json::to_string(block) {
            Ok(v) => Ok(v),
            Err(e) => Err(e.to_string()),
        }
    }

    fn dec(&self, block: &String) -> Result<Block, String> {
        match serde_json::from_str(block) {
            Ok(v) => Ok(v),
            Err(e) => Err(e.to_string()),
        }
    }

    fn dump_data(&self) -> Vec<Block> {
        self.server.inner.borrow() .iter().map(|s| {
            self.dec(s).unwrap()
        }).collect()
    }
}

impl ORAM for TrivialORAM {
    fn read_and_remove(&self, u: Idx) -> Result<Block, String> {
        let mut output = Ok(Block::empty());
        for i in 0..self.server.length {
            let block = self.server.read(i).unwrap();
            let block: Block = self.dec(&block)?;
            if block.u == u {
                let mut empty_block = Block::empty();
                empty_block.leaf = block.leaf;
                self.server.write(i, self.enc(&empty_block)?);
                output = Ok(block);
            } else {
                self.server.write(i, self.enc(&block)?);
            }
        }
        output
    }
    fn add(&self, block: &Block) -> Result<(), String> {
        let mut written = false;
        for i in 0..self.server.length {
            let block_i: Block = self.dec(&self.server.read(i).unwrap())?;
            if block_i.empty && !written {
                self.server.write(i, self.enc(&block)?);
                written = true;
            } else {
                self.server.write(i, self.enc(&block_i)?);
            }
        }
        Ok(())
    }

    fn pop(&self) -> Result<Block, String> {
        let write_empty = |r: &Result<Block, String>| {
            match r {
                Ok(x) => x.empty,
                Err(_) => true,
            }
        };

        let mut output = Err(String::from("no output"));
        for i in 0..self.server.length {
            let block = self.server.read(i).unwrap();
            let block: Block = self.dec(&block)?;
            if block.empty {
                self.server.write(i, self.enc(&block)?);
                if write_empty(&output) {
                    output = Ok(block);
                }
            } else {
                let mut empty_block = Block::empty();
                empty_block.leaf = block.leaf;
                self.server.write(i, self.enc(&empty_block)?);
                output = Ok(block);
            }
        }
        output
    }

    fn capacity(&self) -> usize {
        self.server.length
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
enum Direction {
    Left,
    Right,
}

type Path = Vec<Direction>;

fn leaf_to_path(l: LeafIdx, depth: usize) -> Path {
    let mut l = l;
    let mut out: Path = Vec::new();
    for _ in 0..depth {
        out.push(if l % 2 == 0 { Direction::Left } else { Direction::Right });
        l = l >> 1;
    }
    out
}

fn path_to_leaf(path: &Path) -> LeafIdx {
    let mut l: LeafIdx = 0;
    for x in path.iter().rev() {
        if *x == Direction::Left {
            l = l << 1;
        } else {
            l = (l << 1) + 1;
        }
    }
    l
}

fn random_path(depth: usize) -> (LeafIdx, Path) {
    let leaf = rand::random::<LeafIdx>();
    // TODO should we mask the bits in leaf that are not needed?
    let path = leaf_to_path(leaf, depth);
    (leaf, path)
}

fn extend_path_rand(path: &Path, total_length: usize) -> (LeafIdx, Path) {
    if path.len() >= total_length {
        return (path_to_leaf(&path), path.clone());
    }
    let (_, extra) = random_path(total_length - path.len());
    let mut new_path = path.clone();
    new_path.extend(extra);
    (path_to_leaf(&new_path), new_path)
}

type Bucket = TrivialORAM;

#[derive(Debug)]
struct TreeNode {
    bucket: Bucket,
    left: Option<Box<TreeNode>>,
    right: Option<Box<TreeNode>>,
}

impl TreeNode {
    fn new(bucket: Bucket) -> TreeNode {
        TreeNode {
            bucket,
            left: None,
            right: None,
        }
    }
}

struct TreeORAM {
    depth: usize,
    root: Box<TreeNode>,
    state: RefCell<Option<LeafIdx>>,
    position_map: Box<dyn ORAM>, // mapping of block index and leaf
    eviction_rate: usize,
}

fn log2usize(x: usize) -> usize {
    let mut i: usize = 0;
    let mut n: usize = x;
    loop {
        n = n >> 1;
        if n > 0 {
            i += 1;
        } else {
            break;
        }
    }
    i
}

impl TreeORAM {
    fn new(depth: usize) -> TreeORAM {
        let bucket_size = std::cmp::max(1, depth); // bucket size should be O(log(N)), so using the depth should be ok

        let f = |path: &Path| {
            let leaf_gen = || {
                let (leaf, _) = extend_path_rand(path, depth);
                leaf
            };
            Bucket::with_leaf_generator(bucket_size, &leaf_gen)
        };
        let boxed_tree = TreeORAM::create_tree(depth, &f).unwrap();

        let leaves_count = 2usize.pow(depth as u32);
        let position_map = TrivialORAM::new(leaves_count);

        TreeORAM {
            depth,
            root: boxed_tree,
            state: RefCell::new(None),
            position_map: Box::new(position_map),
            eviction_rate: 2,
        }
    }

    fn create_tree<F>(depth: usize, node_constructor: &F) -> Option<Box<TreeNode>>
        where F: Fn(&Path) -> Bucket {
        TreeORAM::create_tree_rec(depth, &Vec::new(), node_constructor)
    }

    fn create_tree_rec<F>(depth: usize, current_path: &Path, node_constructor: &F) -> Option<Box<TreeNode>>
        where F: Fn(&Path) -> Bucket {
        if depth == 0 {
            // we define height 0 to be a tree with one node
            let node = TreeNode::new(node_constructor(current_path));
            Some(Box::new(node))
        } else {
            let mut node = TreeNode::new(node_constructor(current_path));
            let mut path_left = current_path.clone(); path_left.push(Direction::Left);
            let mut path_right = current_path.clone(); path_right.push(Direction::Right);
            node.left = TreeORAM::create_tree_rec(depth - 1, &path_left, node_constructor);
            node.right = TreeORAM::create_tree_rec(depth - 1, &path_right, node_constructor);
            Some(Box::new(node))
        }
    }

    fn get_buckets_count(&self) -> usize {
        // n = 2^(d + 1) - 1
        (1 << (self.depth + 1)) - 1
    }

    fn get_n(&self) -> usize {
        1 << self.depth
    }

    fn count_nodes(&self) -> usize {
        fn rec(node: &TreeNode) -> usize {
            if node.left.is_none() && node.right.is_none() {
                1
            } else {
                1 + rec(node.left.as_ref().unwrap()) + rec(node.right.as_ref().unwrap())
            }
        }
        rec(&self.root)
    }

    fn sanity_check_paths(&self) -> bool {
        // NOTE: pop checks one block in the bucket, not all
        fn rec(node: &Box<TreeNode>, current_path: &Path, depth: usize) -> bool {
            if current_path.len() == depth {
                let b = node.bucket.pop().unwrap();
                leaf_to_path(b.leaf, depth) == *current_path
            } else {
                let b = node.bucket.pop().unwrap();
                let ok = leaf_to_path(b.leaf, current_path.len()) == *current_path;
                let mut left_path = current_path.clone(); left_path.push(Direction::Left);
                let mut right_path = current_path.clone(); right_path.push(Direction::Right);
                ok && rec(node.left.as_ref().unwrap(), &left_path, depth)
                    && rec(node.right.as_ref().unwrap(), &right_path, depth)
            }
        }
        rec(&self.root, &Vec::new(), self.depth)
    }

    fn evict(&self) -> Result<(), String> {
        for d in 0..(self.depth-1) {
            let s = self.all_buckets_at_depth(d);
            let a: Vec<&&TreeNode> = s.choose_multiple(&mut rand::thread_rng(), self.eviction_rate).collect();
            for node in a {
                let block = node.bucket.pop()?;
                let path = leaf_to_path(block.leaf, self.depth);
                let b = path[d]; // NOTE: but the paper says d+1?

                let empty_msg = String::from("empty");
                // println!("evicting {:?} {:?}, {:?}", path, b, block);
                // TODO write order leaks information
                match b {
                    Direction::Left => {
                        node.left.as_ref().ok_or(empty_msg.clone()).and_then(|x| x.bucket.write(&block))?;
                        node.right.as_ref().ok_or(empty_msg.clone()).and_then(|x| x.bucket.write(&Block::empty()))?;
                    },
                    Direction::Right => {
                        node.right.as_ref().ok_or(empty_msg.clone()).and_then(|x| x.bucket.write(&block))?;
                        node.left.as_ref().ok_or(empty_msg.clone()).and_then(|x| x.bucket.write(&Block::empty()))?;
                    }
                }
            }
        }
        Ok(())
    }

    fn all_buckets_at_depth(&self, depth: usize) -> Vec<&TreeNode> {
        fn rec(tree: &TreeNode, depth: usize) -> Vec<&TreeNode> {
            if depth == 0 {
                vec![tree]
            } else {
                let mut left = rec(tree.left.as_ref().unwrap(), depth-1);
                let right = rec(tree.right.as_ref().unwrap(), depth-1);
                left.extend(right);
                left
            }
        }
        rec(self.root.as_ref(), depth)
    }
}

fn encode_leaf(leaf: &LeafIdx) -> [u8; 32] {
    let mut out: [u8; 32] = [0; 32];
    out[..8].copy_from_slice(&leaf.to_be_bytes());
    out
}

fn decode_leaf(encoded: &[u8; 32]) -> LeafIdx {
    let mut tmp: [u8; 8] = [0; 8];
    tmp.copy_from_slice(&encoded[..8]);
    LeafIdx::from_be_bytes(tmp)
}

impl ORAM for TreeORAM {
    fn read_and_remove(&self, u: usize) -> Result<Block, String> {
        if u >= self.position_map.capacity() {
            return Err(String::from("invalid index"));
        }

        let (leaf_star, path_star) = random_path(self.depth);
        let leaf = decode_leaf(&self.position_map.read(u)?.inner);
        let path = leaf_to_path(leaf, self.depth);
        // println!("reading u: {:?}, leaf: {:?}, path: {:?}", u, leaf, path);

        self.position_map.write(&Block::new(u, encode_leaf(&leaf_star), 0))?;
        *self.state.borrow_mut() = Some(path_to_leaf(&path_star));

        let mut out = Block::empty();
        let mut current_node = self.root.as_ref();
        for i in 0..self.depth {
            let x = current_node.bucket.read_and_remove(u)?;
            if !x.empty && u == x.u {
                out = x;
            }
            if path[i] == Direction::Left {
                current_node = current_node.left.as_ref().unwrap();
            } else {
                current_node = current_node.right.as_ref().unwrap();
            }
        }
        // println!("read and remove {:?}", out);
        Ok(out)
    }

    fn add(&self, block: &Block) -> Result<(), String> {
        if let Some(leaf) = *self.state.borrow() {
            let mut new_block = block.clone();
            new_block.leaf = leaf;
            self.root.bucket.write(&new_block)?;
            self.evict()?;
            // TODO clear the state?
            Ok(())
        } else {
            Err(String::from("state is empty"))
        }
    }

    fn pop(&self) -> Result<Block, String> {
        unimplemented!()
    }

    fn capacity(&self) -> usize {
        self.get_n()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trivial_oram() {
        let oram_size = 10;
        let oram = TrivialORAM::new(oram_size);

        // test encoding
        for b in vec![Block::empty(), Block::new(1, rand::random(), rand::random())] {
            let encoded = oram.enc(&b).unwrap();
            let decoded = oram.dec(&encoded).unwrap();
            assert_eq!(b, decoded);
        }

        // test actual oram
        assert!(oram.pop().unwrap().empty);
        for i in 0..oram.server.length {
            assert!(oram.read(i).unwrap().empty);
        }

        {
            let block = Block::new(0, [1; 32], 0);
            assert_eq!(oram.write(&block).unwrap(), ());
            assert_eq!(oram.read(0).unwrap(), block);
        }
        assert!(!oram.pop().unwrap().empty);

        {
            let block = Block::new(1, [2; 32], 0);
            assert_eq!(oram.write(&block).unwrap(), ());
            assert_eq!(oram.read(1).unwrap(), block);
        }
        assert!(!oram.pop().unwrap().empty);

        {
            let block = Block::new(0, [3; 32], 0);
            assert_eq!(oram.write(&block).unwrap(), ());
            assert_eq!(oram.read(0).unwrap(), block);
        }
        assert!(!oram.pop().unwrap().empty);

        {
            assert_eq!(oram.write(&Block::empty_with_u(0)).unwrap(), ());
            assert_eq!(oram.write(&Block::empty_with_u(1)).unwrap(), ());
        }
        assert!(oram.pop().unwrap().empty);

        assert_eq!(oram.dump_data().len(), oram_size);
    }

    #[test]
    fn helper_functions() {
        assert_eq!(log2usize(2), 1);
        assert_eq!(log2usize(4), 2);
        assert_eq!(log2usize(128), 7);

        assert_eq!(leaf_to_path(1, 4), vec![Direction::Right, Direction::Left, Direction::Left, Direction::Left]);
        assert_eq!(leaf_to_path(2, 4), vec![Direction::Left, Direction::Right, Direction::Left, Direction::Left]);
        assert_eq!(leaf_to_path(3, 4), vec![Direction::Right, Direction::Right, Direction::Left, Direction::Left]);

        for leaf in 0..15 {
            assert_eq!(path_to_leaf(&leaf_to_path(leaf, 4)), leaf);
        }

        assert_eq!(decode_leaf(&encode_leaf(&0)), 0);
        for _ in 0..10 {
            let leaf: LeafIdx = rand::random();
            let encoded = encode_leaf(&leaf);
            let decoded = decode_leaf(&encoded);
            assert_eq!(leaf, decoded);
        }
    }

    fn generic_tree_oram_io(oram_depth: usize) {
        let leaves_count = 2usize.pow(oram_depth as u32);
        let tree = TreeORAM::new(oram_depth);

        // check all is empty
        for i in 0..tree.get_n() {
            assert!(tree.read(i).unwrap().empty);
        }

        // it doesn't matter what leaf is, it will be over written
        tree.write(&Block::new(0, [u8::max_value(); 32], 0)).unwrap();
        assert!(!tree.read(0).unwrap().empty);

        // check writing
        for u in vec![0, 1, 2, 8, leaves_count-1] {
            let block = Block::new(u, [u as u8; 32], 0);
            tree.write(&block).unwrap();
            let result = tree.read(u).unwrap();

            // we don't check the leaf because that's randomized
            assert!(!result.empty);
            assert_eq!(result.inner, block.inner);
            assert_eq!(result.u, block.u);
        }

        // read empty indices
        for u in vec![3, 4, 9, 10] {
            let result = tree.read(u).unwrap();
            assert!(result.empty);
        }

        // write to index that doesn't exist
        let bad_index_block = Block::new(leaves_count, [0; 32], 0);
        assert_eq!(tree.write(&bad_index_block).err().unwrap(), "invalid index");
    }

    #[test]
    fn tree_functions() {
        // sanity check
        for d in vec![0, 1, 4] {
            let tree = TreeORAM::new(d);
            assert_eq!(tree.count_nodes(), tree.get_buckets_count());
            assert_eq!(tree.position_map.capacity(), 1 << d);
            assert!(tree.sanity_check_paths());
        }

        // generic test
        generic_tree_oram_io(4);
        generic_tree_oram_io(5);
    }
}