use std::cell::RefCell;

#[derive(Debug)]
pub struct Server<T: Clone> {
    inner: RefCell<Vec<T>>,
}

impl<T: Clone> Server<T> {
    pub fn new<F>(length: usize, default: &F) -> Server<T>
        where F: Fn() -> T {
        let mut vs: Vec<T> = Vec::new();
        for _ in 0..length {
            vs.push(default());
        }
        Server {
            inner: RefCell::new(vs),
        }
    }

    pub fn read(&self, i: usize) -> Option<T> {
        let x = self.inner.borrow().get(i)?.clone();
        Some(x)
    }

    pub fn write(&self, i: usize, x: T) {
        self.inner.borrow_mut()[i] = x;
    }

    pub fn get_length(&self) -> usize {
        self.inner.borrow().len()
    }

    pub fn dump_data(&self) -> Vec<T>  {
        // TODO return an iterator is a better alternative
        self.inner.borrow().clone()
    }
}

