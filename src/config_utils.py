from typing import List, Tuple

def get_shards_split(n_prompts: int, n_shards: int | None = None) -> List[Tuple[int,int]]:
    """ More honest way to split prompts into shards.
    """

    if n_shards is None or n_shards == 1:
        return [(0, n_prompts)]

    shards_locs = []
    shard_size, rem = divmod(n_prompts, n_shards)
    prev = 0
    for i in range(n_shards):
        if i<rem:
            cur_size = shard_size+1
        else:
            cur_size = shard_size

        shards_locs.append((prev, prev+cur_size))
        prev += cur_size

    return shards_locs
