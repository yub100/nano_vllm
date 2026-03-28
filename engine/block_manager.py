from collections import deque
import xxhash
import numpy as np

from engine.sequence import Sequence

# cache block
class Block:
    # block_id is physical index
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []
    
    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    # initialize
    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []

# manage cache block
# it is used by main process(gpu0), to generate block_table which can be shared by each process.
# all gpu/process are physical cached during model forwarding compution according the block_table.
class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    # prefix cache
    # first seqs: [A, B, C, D]
    # second seqs: [A, B, X, D]
    # A, B can cache, D can't
    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix_hash: int = -1):
        h = xxhash.xxh64()
        if prefix_hash != -1:
            h.update(prefix_hash.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()
    
    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block
    
    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        block = self.blocks[block_id]
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    # only used in prefill period
    # it's logic kvcache
    # Divide seq into blocks and allocate physical block, write the full block to hash_to_block_id.
    # Actually, the block store cache, due to q and k/v have a one-to-one relationship. Thus, can allocate cache for every token(q) indavance.
    def allocate(self, seq: Sequence):
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            # actually there are no blocks
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            
            # if key=h not exist, return -1
            block_id = self.hash_to_block_id.get(h, -1)

            # miss or dirty read
            # once cache_miss = True, all subsequent block will miss.
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True

            # allocate an cache is two step, if cache_miss is true,
            # we just allocate block, don't update cache.thus don't update num_cached_tokens
            if cache_miss:
                # obtain physical block to store kv
                block_id = self.free_block_ids[0]

                # obtain physical block
                block = self._allocate_block(block_id)
            else:
                # hit means this block already exist and have cache, so update num_cached_tokens
                seq.num_cached_tokens += self.block_size

                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)

            # if block is full
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id

            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    # seq' blocks havn't allocated, seq.statue is waiting
    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks
    
    # seq'blocks already allocate, seq.statue is running
    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        # fill the token_ids to block only when len(seq) % self.block_size == 0
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks - 1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1
