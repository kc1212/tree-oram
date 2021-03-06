mod server;
mod oram;

use server::Server;
use oram::*;

use std::cell::RefCell;
use rand::prelude::*;

#[derive(Debug)]
struct TrivialORAM {
    block_size: usize,
    key: [u8; 32],
    nonce: [u8; 32],
    server: Server<String>,
}

impl TrivialORAM {
    fn new(n: usize, block_size: usize) -> TrivialORAM {
        let f = || serde_json::to_string(&Block::empty(block_size)).unwrap();
        let server = Server::new(n, &f);
        TrivialORAM {
            block_size,
            key: [0; 32],
            nonce: [0; 32],
            server,
        }
    }

    fn with_leaf_generator<F>(n: usize, block_size: usize, leaf_generator: &F) -> TrivialORAM
        where F : Fn() -> LeafIdx {
        let f = || {
            let b = Block {
                inner: vec![0; block_size],
                empty: true,
                leaf: leaf_generator(),
                u: BlockIdx::max_value(),
            };
            serde_json::to_string(&b).unwrap()
        };
        let server = Server::new(n, &f);
        TrivialORAM {
            block_size,
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

    fn pop(&self) -> Result<Block, String> {
        let write_empty = |r: &Result<Block, String>| {
            match r {
                Ok(x) => x.empty,
                Err(_) => true,
            }
        };

        let mut output = Err(String::from("no output"));
        for i in 0..self.server.get_length() {
            let block = self.server.read(i).unwrap();
            let block: Block = self.dec(&block)?;
            if block.empty {
                self.server.write(i, self.enc(&block)?);
                if write_empty(&output) {
                    output = Ok(block);
                }
            } else {
                let mut empty_block = Block::empty(self.block_size);
                empty_block.leaf = block.leaf;
                self.server.write(i, self.enc(&empty_block)?);
                output = Ok(block);
            }
        }
        output
    }

}

impl ORAM for TrivialORAM {
    fn read_and_remove(&self, u: BlockIdx) -> Result<Block, String> {
        let mut output = Ok(Block::empty(self.block_size));
        for i in 0..self.server.get_length() {
            let block = self.server.read(i).unwrap();
            let block: Block = self.dec(&block)?;
            if block.u == u {
                let mut empty_block = Block::empty(self.block_size);
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
        for i in 0..self.server.get_length() {
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

    fn get_capacity(&self) -> usize {
        self.server.get_length()
    }

    fn get_block_size(&self) -> usize {
        self.block_size
    }

    fn dump_data(&self) -> Vec<Block> {
        self.server.dump_data().iter().map(|s| {
            self.dec(s).unwrap()
        }).collect()
    }
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

struct PositionMap {
    leaf_count: usize,
    inner: Box<dyn ORAM>
}

impl PositionMap {
    fn new(leaf_count: usize, oram: Box<dyn ORAM>)  -> PositionMap {
        PositionMap {
            leaf_count,
            inner: oram,
        }
    }

    fn get_leaf_len_bytes(&self) -> usize {
        std::cmp::max(1, log2usize(self.leaf_count)/8)
    }

    fn encode_leaf(&self, leaf: LeafIdx) -> Vec<u8> {
        let mut out = leaf.to_le_bytes().to_vec();
        out.resize(self.get_leaf_len_bytes(), 0);
        out
    }

    fn decode_leaf(&self, mut encoded: Vec<u8>) -> LeafIdx {
        // TODO is there an easy to do all this?
        assert_eq!(encoded.len(), self.get_leaf_len_bytes());
        let mut tmp: [u8; 8] = [0; 8];
        encoded.resize(8, 0);
        tmp.copy_from_slice(&encoded);
        LeafIdx::from_le_bytes(tmp)
    }

    fn read_leaf(&self, id: usize) -> Result<LeafIdx, String> {
        if id >= self.leaf_count {
            Err(format!("leaf index is too big"))
        } else {
            let l = self.get_leaf_len_bytes();
            let v = match self.inner.read_u8(id*l, (id+1)*l) {
                Err(e) => if e == "empty block" {
                    Ok(vec![0; self.get_leaf_len_bytes()])
                } else {
                    Err(e)
                }
                x => x,
            }?;
            Ok(self.decode_leaf(v))
        }
    }

    fn write_leaf(&self, id: usize, leaf: LeafIdx) -> Result<(), String> {
        if leaf >= self.leaf_count {
            Err(format!("leaf index is too big"))
        } else if id >= self.leaf_count {
            Err(format!("id index is too big"))
        } else {
            self.inner.write_u8(id*self.get_leaf_len_bytes(), self.encode_leaf(leaf))
        }
    }

    fn get_capacity(&self) -> usize {
        self.leaf_count
    }
}

struct TreeORAM {
    depth: usize,
    root: Box<TreeNode>,
    state: RefCell<Option<LeafIdx>>,
    position_map: PositionMap, // mapping of block index and leaf
    eviction_rate: usize,
}

impl TreeORAM {
    // This implementation currently fixes c=2, from c and the depth we derive the block size.
    const C: usize = 2;

    fn new(depth: usize) -> TreeORAM {
        let bucket_size = std::cmp::max(1, depth); // bucket size should be O(log(N)), so using the depth should be ok
        let block_size = Self::compute_block_size(depth);

        let f = |path: &Path| {
            let leaf_gen = || {
                let (leaf, _) = extend_path_rand(path, depth);
                leaf
            };
            Bucket::with_leaf_generator(bucket_size, block_size, &leaf_gen)
        };
        let boxed_tree = TreeORAM::create_tree(depth, &f).unwrap();

        let leaves_count = 2usize.pow(depth as u32); // this is also the N
        let position_map = {
            let boxed : Box<dyn ORAM> = {
                if leaves_count > 4 {
                    // Here we recursively apply the tree ORAM.
                    let tree = TreeORAM::new(log2usize(leaves_count/Self::C));
                    assert_eq!(tree.get_capacity(), leaves_count/Self::C);
                    Box::new(tree)
                } else {
                    let trivial = TrivialORAM::new(
                        if leaves_count >= Self::C {leaves_count/Self::C} else {leaves_count},
                        block_size);
                    Box::new(trivial)
                }
            };
            let pmap = PositionMap::new(leaves_count, boxed);
            pmap
        };

        TreeORAM {
            depth,
            root: boxed_tree,
            state: RefCell::new(None),
            position_map,
            // NOTE: it seems when eviction rate is higher,
            // leaf ID are less likely to stomp on each other.
            // This might also be a bug, needs more investigation.
            eviction_rate: 4,
        }
    }

    fn create_tree<F>(depth: usize, node_constructor: &F) -> Option<Box<TreeNode>>
        where F: Fn(&Path) -> Bucket {
        Self::create_tree_rec(depth, &Vec::new(), node_constructor)
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
            node.left = Self::create_tree_rec(depth - 1, &path_left, node_constructor);
            node.right = Self::create_tree_rec(depth - 1, &path_right, node_constructor);
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
                let empty_block = Block::empty(self.get_block_size());
                // TODO write order leaks information
                match b {
                    Direction::Left => {
                        node.left.as_ref().ok_or(empty_msg.clone()).and_then(|x| x.bucket.write(&block))?;
                        node.right.as_ref().ok_or(empty_msg.clone()).and_then(|x| x.bucket.write(&empty_block))?;
                    },
                    Direction::Right => {
                        node.right.as_ref().ok_or(empty_msg.clone()).and_then(|x| x.bucket.write(&block))?;
                        node.left.as_ref().ok_or(empty_msg.clone()).and_then(|x| x.bucket.write(&empty_block))?;
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

    fn compute_block_size(depth: usize) -> usize {
        // We fix c=2, so 2 = B/(log N), B = 2*(log N) where N = 2^D, so B = 2*D
        // NOTE: block size is measured in bits in the paper, but here's we're measuring it as bytes.
        std::cmp::max(8, round_up(Self::C*depth, 8)/8)
    }
}

impl ORAM for TreeORAM {
    fn read_and_remove(&self, u: usize) -> Result<Block, String> {
        if u >= self.position_map.get_capacity() {
            return Err(String::from("invalid index"));
        }

        let (leaf_star, path_star) = random_path(self.depth);
        let leaf = self.position_map.read_leaf(u)?;
        let path = leaf_to_path(leaf, self.depth);

        self.position_map.write_leaf(u, leaf_star)?;
        *self.state.borrow_mut() = Some(path_to_leaf(&path_star));

        let mut out = Block::empty(self.get_block_size());
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

    fn get_capacity(&self) -> usize {
        self.get_n()
    }

    fn get_block_size(&self) -> usize {
        Self::compute_block_size(self.depth)
    }

    fn dump_data(&self) -> Vec<Block> {
        let mut blocks = Vec::new();
        for i in 0..self.get_capacity() {
            blocks.push(self.read(i).unwrap());
        }
        blocks
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trivial_oram() {
        let oram_size = 10;
        const BLOCK_SIZE: usize = 32;
        let oram = TrivialORAM::new(oram_size, BLOCK_SIZE);

        // test encoding
        let rand_value: [u8; BLOCK_SIZE] = rand::random();
        for b in vec![Block::empty(BLOCK_SIZE), Block::new(1, rand_value.to_vec(), rand::random())] {
            let encoded = oram.enc(&b).unwrap();
            let decoded = oram.dec(&encoded).unwrap();
            assert_eq!(b, decoded);
        }

        // test actual oram
        assert!(oram.pop().unwrap().empty);
        for i in 0..oram.server.get_length() {
            assert!(oram.read(i).unwrap().empty);
        }

        {
            let block = Block::new(0, vec![1; BLOCK_SIZE], 0);
            assert_eq!(oram.write(&block).unwrap(), ());
            assert_eq!(oram.read(0).unwrap(), block);

            // overwrite
            let block2 = Block::new(0, vec![2; BLOCK_SIZE], 0);
            assert_eq!(oram.write(&block2).unwrap(), ());
            assert_eq!(oram.read(0).unwrap(), block2);
        }
        assert!(!oram.pop().unwrap().empty);
        assert!(oram.pop().unwrap().empty);

        {
            let block = Block::new(1, vec![2; BLOCK_SIZE], 0);
            assert_eq!(oram.write(&block).unwrap(), ());
            assert_eq!(oram.read(1).unwrap(), block);
        }
        assert!(!oram.pop().unwrap().empty);
        assert!(oram.pop().unwrap().empty);

        {
            let block = Block::new(0, vec![3; BLOCK_SIZE], 0);
            assert_eq!(oram.write(&block).unwrap(), ());
            assert_eq!(oram.read(0).unwrap(), block);
        }
        assert!(!oram.pop().unwrap().empty);
        assert!(oram.pop().unwrap().empty);

        {
            assert_eq!(oram.write(&Block::empty_with_u(BLOCK_SIZE, 0)).unwrap(), ());
            assert_eq!(oram.write(&Block::empty_with_u(BLOCK_SIZE, 1)).unwrap(), ());
        }
        assert!(oram.pop().unwrap().empty);

        assert_eq!(oram.dump_data().len(), oram_size);
    }

    fn generic_tree_oram_io(oram_depth: usize) {
        let leaves_count = 2usize.pow(oram_depth as u32);
        let tree = TreeORAM::new(oram_depth);

        // test position map
        assert_eq!(tree.position_map.write_leaf(0, 1).unwrap(), ());
        assert_eq!(tree.position_map.read_leaf(0).unwrap(), 1);
        assert_eq!(tree.position_map.write_leaf(0, 2).unwrap(), ());
        assert_eq!(tree.position_map.read_leaf(0).unwrap(), 2);

        // check all is empty
        for i in 0..tree.get_capacity() {
            assert!(tree.read(i).unwrap().empty);
        }

        // it doesn't matter what leaf is, it will be over written
        assert_eq!(tree.write(&Block::new(0, vec![u8::max_value(); tree.get_block_size()], 0)).unwrap(), ());
        assert!(!tree.read(0).unwrap().empty);

        // check writing
        for u in vec![0, 1, 2, 8, leaves_count-1] {
            let block = Block::new(u, vec![u as u8; tree.get_block_size()], 0);
            assert_eq!(tree.write(&block).unwrap(), ());
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
        let bad_index_block = Block::new(leaves_count, vec![0; tree.get_block_size()], 0);
        assert_eq!(tree.write(&bad_index_block).err().unwrap(), "invalid index");

        // write with an invalid block size
        let bad_sized_block = Block::new(0, vec![0; tree.get_block_size()*2], 0);
        assert_eq!(tree.write(&bad_sized_block).err().unwrap(), "invalid block size");
    }

    #[test]
    fn tree_functions() {
        // sanity check
        for d in vec![0, 1, 4] {
            let tree = TreeORAM::new(d);
            assert_eq!(tree.count_nodes(), tree.get_buckets_count());
            assert!(tree.sanity_check_paths());
        }

        // generic test
        generic_tree_oram_io(4);
        generic_tree_oram_io(5);
    }

    fn generic_oram_u8_io(oram: &dyn ORAM) {
        assert_eq!(oram.write_u8(0, vec![42]).unwrap(), ());
        assert_eq!(oram.read_u8(0, 1).unwrap(), vec![42]);

        let b = oram.get_block_size();
        let start_idx = b/2;
        assert_eq!(oram.write_u8(start_idx, vec![42; b]).unwrap(), ());
        assert_eq!(oram.read_u8(start_idx, start_idx+b).unwrap(), vec![42; b]);
    }

    #[test]
    fn trivial_oram_u8_io() {
        let oram_size = 10;
        const BLOCK_SIZE: usize = 32;
        let oram = TrivialORAM::new(oram_size, BLOCK_SIZE);
        generic_oram_u8_io(&oram);
    }

    #[test]
    fn tree_oram_u8_io() {
        let tree2 = TreeORAM::new(4);
        generic_oram_u8_io(&tree2);
    }
}
