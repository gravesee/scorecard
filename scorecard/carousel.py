from typing import Generic, Sequence, TypeVar

T = TypeVar('T')

class Carousel(Generic[T]):
    def __init__(self, seq: Sequence[T]):
        self.seq = seq
        self.i = 0

    def next(self) -> T:
        self.i = (self.i + 1) % len(self.seq)
        return self.curr()
    
    def prev(self) -> T:
        self.i = (self.i - 1) % len(self.seq)
        return self.curr()
    
    def curr(self) -> T:
        return self.seq[self.i]
    
    @property
    def items(self) -> Sequence[T]:
        return self.seq
