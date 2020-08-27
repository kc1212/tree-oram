
use serde::{Deserialize, Serialize};

pub fn log2usize(x: usize) -> usize {
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

pub fn round_up(x: usize, multiple: usize) -> usize {
    (x + multiple - 1) & (usize::max_value()-multiple+1)
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Direction {
    Left,
    Right,
}

pub type Path = Vec<Direction>;

pub fn leaf_to_path(l: LeafIdx, depth: usize) -> Path {
    let mut l = l;
    let mut out: Path = Vec::new();
    for _ in 0..depth {
        out.push(if l % 2 == 0 { Direction::Left } else { Direction::Right });
        l = l >> 1;
    }
    out
}

pub fn path_to_leaf(path: &Path) -> LeafIdx {
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

pub fn random_path(depth: usize) -> (LeafIdx, Path) {
    let long_leaf = rand::random::<LeafIdx>();
    // TODO should we mask the bits in leaf that are not needed?
    let path = leaf_to_path(long_leaf, depth);
    (path_to_leaf(&path), path)
}

pub fn extend_path_rand(path: &Path, total_length: usize) -> (LeafIdx, Path) {
    if path.len() >= total_length {
        return (path_to_leaf(&path), path.clone());
    }
    let (_, extra) = random_path(total_length - path.len());
    let mut new_path = path.clone();
    new_path.extend(extra);
    (path_to_leaf(&new_path), new_path)
}

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct Block {
    pub inner: Vec<u8>,
    pub empty: bool,
    pub leaf: LeafIdx,
    pub u: BlockIdx,
}

impl Block {
    pub fn empty(size: usize) -> Block {
        Self::empty_with_u(size, BlockIdx::max_value())
    }

    pub fn empty_with_u(size: usize, u: BlockIdx) -> Block {
        Block {
            inner: vec![0; size],
            empty: true,
            leaf: 0,
            u,
        }
    }

    pub fn new(u: BlockIdx, value: Vec<u8>, leaf: LeafIdx) -> Block {
        Block {
            inner: value,
            empty: false,
            leaf,
            u
        }
    }
}

pub type BlockIdx = usize;
pub type LeafIdx = usize;

pub trait ORAM {
    fn read_and_remove(&self, u: BlockIdx) -> Result<Block, String>;
    fn add(&self, block: &Block) -> Result<(), String>;
    fn get_capacity(&self) -> usize;
    fn get_block_size(&self) -> usize;
    fn dump_data(&self) -> Vec<Block>;

    fn read(&self, u: BlockIdx) -> Result<Block, String> {
        let block = self.read_and_remove(u)?;
        self.add(&block)?;

        if block.inner.len() != self.get_block_size() {
            Err(String::from("invalid block size"))
        } else {
            Ok(block)
        }
    }

    fn read_more(&self, start: BlockIdx, end: BlockIdx) -> Result<Vec<Block>, String> {
        let mut blocks = Vec::new();
        for i in start..end {
            blocks.push(self.read(i)?);
        }
        Ok(blocks)
    }

    fn write(&self, block: &Block) -> Result<(), String> {
        if block.inner.len() != self.get_block_size() {
            Err(String::from("invalid block size"))
        } else {
            self.read_and_remove(block.u)?;
            self.add(block)
        }
    }

    fn write_more(&self, blocks: &Vec<Block>) -> Result<(), String> {
        for block in blocks {
            self.write(block)?;
        }
        Ok(())
    }

    fn sanity_check_u8_io(&self, start: usize, end: usize) -> Result<(), String> {
        if end < start {
            return Err(String::from("invalid start/end"));
        }
        let total_bytes = self.get_block_size()*self.get_capacity();
        if end > total_bytes {
            return Err(format!("index out of range end: {} >= total: {}", end, total_bytes));
        }
        Ok(())
    }

    fn read_u8(&self, start: usize, end: usize) -> Result<Vec<u8>, String> {
        self.sanity_check_u8_io(start, end)?;

        let b = self.get_block_size();
        if end == start {
            return Ok(Vec::new());
        }

        let start_block = start/b;
        let end_block = if (end % b) == 0 {end/b} else {(end+b)/b};
        let blocks = self.read_more(start_block, end_block)?;

        let mut out = Vec::new();
        for i in start..end {
            let local_idx = i/b - start_block;
            let inner_idx = i%b;
            if blocks[local_idx].empty {
                return Err(String::from("empty block"));
            }
            if blocks[local_idx].u != i/b {
                return Err(String::from("invalid block index"));
            }
            out.push(blocks[local_idx].inner[inner_idx])
        }
        Ok(out)
    }

    fn write_u8(&self, start: usize, data: Vec<u8>) -> Result<(), String> {
        // NOTE: this function does a read_more and a write_more,
        // so it's not indistinguishable from read_u8.
        // We have to read the data back first because we don't want to change
        // the existing part of the block that is not overwritten.
        let end = start + data.len();
        self.sanity_check_u8_io(start, end)?;

        let b = self.get_block_size();
        let start_block = start/b;
        let end_block = if (end % b) == 0 {end/b} else {(end+b)/b};

        let mut blocks = self.read_more(start_block, end_block)?;

        for i in start..end {
            let local_idx = i/b - start_block;
            let inner_idx = i%b;
            if !blocks[local_idx].empty && blocks[local_idx].u != i/b {
                return Err(String::from("invalid block index"));
            }
            blocks[local_idx].inner[inner_idx] = data[i-start];
            blocks[local_idx].u = i/b;
            blocks[local_idx].empty = false;
        }
        self.write_more(&blocks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        assert_eq!(round_up(1, 8), 8);
        assert_eq!(round_up(7, 8), 8);
        assert_eq!(round_up(9, 8), 16);
    }
}
